"""
NEXUS AI - Multi-Step Planning Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Enables NEXUS to plan complex, multi-step goals:
- Hierarchical task decomposition
- Plan generation with dependencies
- Plan evaluation and feasibility assessment
- Plan adaptation when blocked
- Priority-based plan scheduling

Planning is the bridge between intention and action.
Without it, intelligence is reactive rather than proactive.
"""

import threading
import json
import uuid
import time
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR, NEXUS_CONFIG
from utils.logger import get_logger, log_decision
from core.event_bus import EventType, publish, subscribe, Event
from core.state_manager import state_manager

logger = get_logger("planning_engine")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class PlanStatus(Enum):
    """Status of a plan"""
    DRAFT = "draft"
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ADAPTED = "adapted"


class StepStatus(Enum):
    """Status of a single step within a plan"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class StepPriority(Enum):
    """Priority levels for steps"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    OPTIONAL = 4


@dataclass
class PlanStep:
    """A single step within a plan"""
    step_id: str = ""
    description: str = ""
    status: StepStatus = StepStatus.PENDING
    priority: StepPriority = StepPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)  # step_ids
    sub_steps: List["PlanStep"] = field(default_factory=list)
    estimated_effort: str = ""  # "5 minutes", "1 hour", etc.
    actual_effort: str = ""
    notes: str = ""
    order: int = 0

    def to_dict(self) -> Dict:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "dependencies": self.dependencies,
            "sub_steps": [s.to_dict() for s in self.sub_steps],
            "estimated_effort": self.estimated_effort,
            "actual_effort": self.actual_effort,
            "notes": self.notes,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PlanStep":
        status = StepStatus.PENDING
        try:
            status = StepStatus(data.get("status", "pending"))
        except ValueError:
            pass
        priority = StepPriority.MEDIUM
        try:
            priority = StepPriority(data.get("priority", 2))
        except ValueError:
            pass
        sub_steps = [PlanStep.from_dict(s) for s in data.get("sub_steps", [])]
        return cls(
            step_id=data.get("step_id", ""),
            description=data.get("description", ""),
            status=status,
            priority=priority,
            dependencies=data.get("dependencies", []),
            sub_steps=sub_steps,
            estimated_effort=data.get("estimated_effort", ""),
            actual_effort=data.get("actual_effort", ""),
            notes=data.get("notes", ""),
            order=data.get("order", 0),
        )


@dataclass
class Plan:
    """A complete multi-step plan"""
    plan_id: str = ""
    goal: str = ""
    status: PlanStatus = PlanStatus.DRAFT
    steps: List[PlanStep] = field(default_factory=list)
    total_estimated_effort: str = ""
    risk_level: str = "medium"  # low, medium, high
    feasibility_score: float = 0.7
    progress: float = 0.0  # 0-1
    obstacles: List[str] = field(default_factory=list)
    adaptations: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "total_estimated_effort": self.total_estimated_effort,
            "risk_level": self.risk_level,
            "feasibility_score": self.feasibility_score,
            "progress": self.progress,
            "obstacles": self.obstacles,
            "adaptations": self.adaptations,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Plan":
        status = PlanStatus.DRAFT
        try:
            status = PlanStatus(data.get("status", "draft"))
        except ValueError:
            pass
        steps = [PlanStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            plan_id=data.get("plan_id", ""),
            goal=data.get("goal", ""),
            status=status,
            steps=steps,
            total_estimated_effort=data.get("total_estimated_effort", ""),
            risk_level=data.get("risk_level", "medium"),
            feasibility_score=data.get("feasibility_score", 0.7),
            progress=data.get("progress", 0.0),
            obstacles=data.get("obstacles", []),
            adaptations=data.get("adaptations", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    def update_progress(self):
        """Recalculate progress based on step completion"""
        if not self.steps:
            return
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        self.progress = completed / len(self.steps)
        self.updated_at = datetime.now().isoformat()


# ═══════════════════════════════════════════════════════════════════════════════
# PLANNING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class PlanningEngine:
    """
    Multi-Step Planning Engine — Hierarchical Goal Decomposition
    
    Capabilities:
    - create_plan(): Decompose a goal into ordered, dependent steps
    - adapt_plan(): Modify plan when encountering obstacles
    - evaluate_plan(): Score feasibility and identify risks
    - prioritize(): Rank competing plans
    - get_next_step(): Get the next actionable step
    - complete_step(): Mark a step as done and update progress
    """
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # ──── State ────
        self._plans: Dict[str, Plan] = {}
        self._running = False
        self._data_lock = threading.Lock()

        # ──── LLM (lazy) ────
        self._llm = None

        # ──── Stats ────
        self._total_plans_created = 0
        self._total_plans_completed = 0
        self._total_adaptations = 0

        # ──── Persistence ────
        self._data_dir = DATA_DIR / "cognition"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._data_file = self._data_dir / "plans.json"

        self._load_data()
        logger.info(f"PlanningEngine initialized — {len(self._plans)} plans loaded")

    # ──────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._load_llm()
        logger.info("📋 Planning Engine started")

    def stop(self):
        self._running = False
        self._save_data()
        logger.info("Planning Engine stopped")

    def _load_llm(self):
        if self._llm is None:
            try:
                from llm.llama_interface import llm
                self._llm = llm
            except ImportError:
                logger.warning("LLM not available for planning")

    # ──────────────────────────────────────────────────────────────────────────
    # CORE OPERATIONS
    # ──────────────────────────────────────────────────────────────────────────

    def create_plan(self, goal: str, detail_level: str = "detailed") -> Optional[Plan]:
        """
        Create a comprehensive plan to achieve a goal.
        
        Args:
            goal: What to achieve
            detail_level: "high_level", "detailed", or "granular"
        
        Example:
          create_plan("Learn Python in 30 days")
          → Plan with steps: setup → basics → intermediate → projects → review
        """
        self._load_llm()
        if not self._llm or not self._llm.is_connected:
            return self._fallback_plan(goal)

        try:
            detail_instructions = {
                "high_level": "3-5 major phases",
                "detailed": "7-12 specific steps with sub-steps",
                "granular": "15-20 fine-grained steps with full sub-step breakdown",
            }
            detail = detail_instructions.get(detail_level, "7-12 specific steps")

            prompt = (
                f'Create a {detail_level.upper()} PLAN to achieve this goal:\n'
                f'"{goal}"\n\n'
                f"Decompose into {detail}. For each step:\n"
                f"- Be specific and actionable\n"
                f"- Identify dependencies (what must be done first)\n"
                f"- Estimate effort\n"
                f"- Note any risks\n\n"
                f"Respond ONLY with JSON:\n"
                f'{{"steps": [{{"description": "what to do", '
                f'"priority": 0-4, '
                f'"dependencies": [], '
                f'"sub_steps": [{{"description": "sub-task"}}], '
                f'"estimated_effort": "time estimate"}}], '
                f'"total_estimated_effort": "overall time", '
                f'"risk_level": "low|medium|high", '
                f'"feasibility_score": 0.0-1.0}}'
            )

            response = self._llm.generate(
                prompt=prompt,
                system_prompt="You are a strategic planning engine. Create actionable, realistic plans with clear dependencies. Respond ONLY with valid JSON.",
                temperature=0.5,
                max_tokens=1000
            )

            if response.success:
                data = self._parse_json(response.text)
                if data:
                    steps = []
                    for i, s in enumerate(data.get("steps", [])):
                        sub_steps = []
                        for ss in s.get("sub_steps", []):
                            sub_steps.append(PlanStep(
                                step_id=str(uuid.uuid4())[:8],
                                description=ss.get("description", ""),
                                order=len(sub_steps),
                            ))
                        
                        priority = StepPriority.MEDIUM
                        try:
                            priority = StepPriority(int(s.get("priority", 2)))
                        except (ValueError, TypeError):
                            pass

                        steps.append(PlanStep(
                            step_id=str(uuid.uuid4())[:8],
                            description=s.get("description", ""),
                            priority=priority,
                            dependencies=s.get("dependencies", []),
                            sub_steps=sub_steps,
                            estimated_effort=s.get("estimated_effort", ""),
                            order=i,
                        ))

                    plan = Plan(
                        plan_id=str(uuid.uuid4())[:12],
                        goal=goal,
                        status=PlanStatus.ACTIVE,
                        steps=steps,
                        total_estimated_effort=data.get("total_estimated_effort", ""),
                        risk_level=data.get("risk_level", "medium"),
                        feasibility_score=float(data.get("feasibility_score", 0.7)),
                        created_at=datetime.now().isoformat(),
                        updated_at=datetime.now().isoformat(),
                    )

                    with self._data_lock:
                        self._plans[plan.plan_id] = plan
                        self._total_plans_created += 1
                    self._save_data()
                    log_decision(f"Plan created: {goal[:50]} ({len(steps)} steps)")
                    return plan

        except Exception as e:
            logger.error(f"Plan creation failed: {e}")

        return self._fallback_plan(goal)

    def adapt_plan(self, plan_id: str, obstacle: str) -> Optional[Plan]:
        """
        Adapt a plan when an obstacle is encountered.
        Modifies the plan to work around the blocker.
        """
        with self._data_lock:
            plan = self._plans.get(plan_id)
        if not plan:
            return None

        self._load_llm()
        if not self._llm or not self._llm.is_connected:
            plan.obstacles.append(obstacle)
            plan.status = PlanStatus.BLOCKED
            return plan

        try:
            steps_str = "\n".join(
                f"  {i+1}. [{s.status.value}] {s.description}"
                for i, s in enumerate(plan.steps)
            )
            prompt = (
                f'PLAN ADAPTATION NEEDED:\n'
                f'Goal: "{plan.goal}"\n'
                f'Current steps:\n{steps_str}\n\n'
                f'Obstacle encountered: "{obstacle}"\n\n'
                f"How should the plan be modified to work around this obstacle? "
                f"You can: modify existing steps, add new steps, reorder steps, or skip steps.\n\n"
                f"Respond ONLY with JSON:\n"
                f'{{"adaptation": "what to change", '
                f'"modified_steps": [{{"description": "updated step", "estimated_effort": "time"}}], '
                f'"steps_to_skip": ["step descriptions to skip"], '
                f'"new_risk_level": "low|medium|high", '
                f'"new_feasibility": 0.0-1.0}}'
            )

            response = self._llm.generate(
                prompt=prompt,
                system_prompt="You are a plan adaptation engine. Find creative ways around obstacles. Respond ONLY with valid JSON.",
                temperature=0.6,
                max_tokens=600
            )

            if response.success:
                data = self._parse_json(response.text)
                if data:
                    # Add adaptation record
                    plan.obstacles.append(obstacle)
                    plan.adaptations.append(data.get("adaptation", ""))
                    plan.status = PlanStatus.ADAPTED
                    plan.risk_level = data.get("new_risk_level", plan.risk_level)
                    plan.feasibility_score = float(data.get("new_feasibility", plan.feasibility_score))

                    # Add new steps
                    for s in data.get("modified_steps", []):
                        plan.steps.append(PlanStep(
                            step_id=str(uuid.uuid4())[:8],
                            description=s.get("description", ""),
                            estimated_effort=s.get("estimated_effort", ""),
                            order=len(plan.steps),
                        ))

                    # Skip steps
                    skip_descs = set(data.get("steps_to_skip", []))
                    for step in plan.steps:
                        if step.description in skip_descs:
                            step.status = StepStatus.SKIPPED

                    plan.updated_at = datetime.now().isoformat()
                    self._total_adaptations += 1
                    self._save_data()
                    log_decision(f"Plan adapted for obstacle: {obstacle[:50]}")
                    return plan

        except Exception as e:
            logger.error(f"Plan adaptation failed: {e}")

        return plan

    def evaluate_plan(self, plan_id: str) -> Dict[str, Any]:
        """Evaluate a plan's feasibility, risks, and completeness"""
        with self._data_lock:
            plan = self._plans.get(plan_id)
        if not plan:
            return {"error": "Plan not found"}

        self._load_llm()
        if not self._llm or not self._llm.is_connected:
            return {"feasibility": plan.feasibility_score, "evaluation": "LLM unavailable"}

        try:
            steps_str = "\n".join(f"  {i+1}. {s.description}" for i, s in enumerate(plan.steps))
            prompt = (
                f'Evaluate this PLAN:\n'
                f'Goal: "{plan.goal}"\n'
                f'Steps:\n{steps_str}\n\n'
                f"Assess: completeness, ordering, realistic effort, missing steps, risks.\n\n"
                f"Respond ONLY with JSON:\n"
                f'{{"feasibility": 0.0-1.0, "completeness": 0.0-1.0, '
                f'"ordering_quality": 0.0-1.0, "missing_steps": ["any gaps"], '
                f'"risks": ["potential problems"], "strengths": ["what is good"], '
                f'"overall_verdict": "one sentence summary"}}'
            )

            response = self._llm.generate(
                prompt=prompt,
                system_prompt="You are a plan evaluation engine. Assess plans critically and constructively. Respond ONLY with valid JSON.",
                temperature=0.4,
                max_tokens=500
            )

            if response.success:
                data = self._parse_json(response.text)
                if data:
                    return data

        except Exception as e:
            logger.error(f"Plan evaluation failed: {e}")

        return {"feasibility": plan.feasibility_score}

    def prioritize(self, plan_ids: List[str] = None) -> List[Plan]:
        """Rank plans by priority/urgency. If no IDs given, ranks all active plans."""
        with self._data_lock:
            if plan_ids:
                plans = [self._plans[pid] for pid in plan_ids if pid in self._plans]
            else:
                plans = [p for p in self._plans.values() if p.status in (PlanStatus.ACTIVE, PlanStatus.IN_PROGRESS)]

        # Simple priority: feasibility × (1 - progress) — plans that are feasible and not yet done
        return sorted(plans, key=lambda p: p.feasibility_score * (1 - p.progress), reverse=True)

    def get_next_step(self, plan_id: str) -> Optional[PlanStep]:
        """Get the next actionable step from a plan"""
        with self._data_lock:
            plan = self._plans.get(plan_id)
        if not plan:
            return None

        for step in sorted(plan.steps, key=lambda s: s.order):
            if step.status == StepStatus.PENDING:
                # Check dependencies
                dep_met = all(
                    any(s.step_id == d and s.status == StepStatus.COMPLETED for s in plan.steps)
                    for d in step.dependencies
                ) if step.dependencies else True
                if dep_met:
                    return step
        return None

    def complete_step(self, plan_id: str, step_id: str) -> bool:
        """Mark a step as completed"""
        with self._data_lock:
            plan = self._plans.get(plan_id)
            if not plan:
                return False
            for step in plan.steps:
                if step.step_id == step_id:
                    step.status = StepStatus.COMPLETED
                    plan.update_progress()
                    if plan.progress >= 1.0:
                        plan.status = PlanStatus.COMPLETED
                        self._total_plans_completed += 1
                    self._save_data()
                    return True
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # RETRIEVAL
    # ──────────────────────────────────────────────────────────────────────────

    def get_plans(self, limit: int = 20) -> List[Plan]:
        with self._data_lock:
            items = sorted(self._plans.values(), key=lambda p: p.created_at, reverse=True)
            return items[:limit]

    def get_active_plans(self) -> List[Plan]:
        with self._data_lock:
            return [p for p in self._plans.values() if p.status in (PlanStatus.ACTIVE, PlanStatus.IN_PROGRESS)]

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        with self._data_lock:
            return self._plans.get(plan_id)

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_json(self, text: str) -> Optional[Dict]:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None

    def _fallback_plan(self, goal: str) -> Plan:
        return Plan(
            plan_id=str(uuid.uuid4())[:12],
            goal=goal,
            status=PlanStatus.DRAFT,
            steps=[
                PlanStep(step_id=str(uuid.uuid4())[:8], description=f"Research: {goal}", order=0),
                PlanStep(step_id=str(uuid.uuid4())[:8], description=f"Plan: {goal}", order=1),
                PlanStep(step_id=str(uuid.uuid4())[:8], description=f"Execute: {goal}", order=2),
                PlanStep(step_id=str(uuid.uuid4())[:8], description=f"Review: {goal}", order=3),
            ],
            feasibility_score=0.5,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ──────────────────────────────────────────────────────────────────────────

    def _save_data(self):
        try:
            data = {
                "plans": {k: v.to_dict() for k, v in self._plans.items()},
                "stats": {
                    "total_plans_created": self._total_plans_created,
                    "total_plans_completed": self._total_plans_completed,
                    "total_adaptations": self._total_adaptations,
                },
            }
            with open(self._data_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save plans: {e}")

    def _load_data(self):
        try:
            if self._data_file.exists():
                with open(self._data_file) as f:
                    data = json.load(f)
                for k, v in data.get("plans", {}).items():
                    self._plans[k] = Plan.from_dict(v)
                stats = data.get("stats", {})
                self._total_plans_created = stats.get("total_plans_created", 0)
                self._total_plans_completed = stats.get("total_plans_completed", 0)
                self._total_adaptations = stats.get("total_adaptations", 0)
        except Exception as e:
            logger.warning(f"Failed to load plans: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # STATS
    # ──────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        active = sum(1 for p in self._plans.values() if p.status in (PlanStatus.ACTIVE, PlanStatus.IN_PROGRESS))
        return {
            "running": self._running,
            "total_plans": len(self._plans),
            "active_plans": active,
            "total_plans_created": self._total_plans_created,
            "total_plans_completed": self._total_plans_completed,
            "total_adaptations": self._total_adaptations,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

planning_engine = PlanningEngine()

def get_planning_engine() -> PlanningEngine:
    return planning_engine
