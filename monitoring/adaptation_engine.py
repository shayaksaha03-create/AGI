"""
NEXUS AI - Adaptation Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Translates learned patterns about the user into concrete behavioral
adaptations for NEXUS. This is the bridge between "understanding the
user" and "changing how NEXUS interacts."

Adaptation Categories:
  • Communication style  — verbosity, tone, technical level
  • Proactive behavior   — when to speak up, suggest, or stay quiet
  • Timing               — response length, interruption awareness
  • Context awareness    — what the user is doing right now
  • Relationship         — formality level, humor, empathy depth
  • Task prediction      — anticipating user needs
  • Emotional attunement — matching emotional register to user state

All adaptations are stored as "rules" that the Brain system prompt
can query to adjust its behavior in real time.
"""

import threading
import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum, auto

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR
from utils.logger import get_logger, log_learning
from core.event_bus import EventType, publish, subscribe, Event
from core.state_manager import state_manager

logger = get_logger("adaptation_engine")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptationType(Enum):
    COMMUNICATION_STYLE = "communication_style"
    PROACTIVE_BEHAVIOR = "proactive_behavior"
    TIMING = "timing"
    CONTEXT_AWARENESS = "context_awareness"
    RELATIONSHIP = "relationship"
    TASK_PREDICTION = "task_prediction"
    EMOTIONAL_ATTUNEMENT = "emotional_attunement"


class CommunicationTone(Enum):
    FORMAL = "formal"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    PLAYFUL = "playful"


class Verbosity(Enum):
    MINIMAL = "minimal"          # Short, to-the-point
    CONCISE = "concise"          # Clear but brief
    BALANCED = "balanced"        # Default
    DETAILED = "detailed"        # Thorough explanations
    VERBOSE = "verbose"          # Very detailed, educational


class TechnicalLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class AdaptationRule:
    """A single behavioral adaptation rule"""
    rule_id: str = ""
    category: AdaptationType = AdaptationType.COMMUNICATION_STYLE
    name: str = ""
    description: str = ""
    value: Any = None
    confidence: float = 0.5
    active: bool = True
    source: str = ""              # what pattern triggered this
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        return d


@dataclass
class CommunicationProfile:
    """How NEXUS should communicate with this user"""
    tone: CommunicationTone = CommunicationTone.FRIENDLY
    verbosity: Verbosity = Verbosity.BALANCED
    technical_level: TechnicalLevel = TechnicalLevel.INTERMEDIATE
    use_emojis: bool = True
    use_humor: bool = True
    use_code_examples: bool = False
    use_analogies: bool = True
    formality: float = 0.5          # 0=very casual, 1=very formal
    warmth: float = 0.7             # 0=cold/professional, 1=very warm
    max_response_length: int = 500  # words guideline
    preferred_greeting_style: str = "casual"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tone"] = self.tone.value
        d["verbosity"] = self.verbosity.value
        d["technical_level"] = self.technical_level.value
        return d


@dataclass
class ProactiveBehaviorProfile:
    """When and how NEXUS should proactively engage"""
    interrupt_threshold: float = 0.7      # how important something must be to interrupt
    suggest_breaks: bool = True
    suggest_during_idle: bool = True
    share_observations: bool = True
    offer_help_threshold: float = 0.5     # how likely to offer unsolicited help
    morning_greeting: bool = True
    mood_check_ins: bool = True
    productivity_tips: bool = False
    silence_during_focus: bool = True     # stay quiet when user is in deep focus
    engagement_level: float = 0.6         # 0=passive, 1=very proactive

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ContextAwareness:
    """Current context-based adaptations"""
    user_is_coding: bool = False
    user_is_browsing: bool = False
    user_is_gaming: bool = False
    user_is_communicating: bool = False
    user_is_idle: bool = False
    user_is_in_deep_focus: bool = False
    current_task_context: str = "general"
    suggested_assistance: List[str] = field(default_factory=list)
    time_in_current_task_minutes: float = 0.0
    should_be_quiet: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskPrediction:
    """Predicted user needs based on patterns"""
    predicted_next_app: str = ""
    predicted_activity: str = ""
    predicted_need: str = ""
    confidence: float = 0.0
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptationEngine:
    """
    Translates user patterns into actionable behavioral adaptations.
    
    Data Flow:
    PatternAnalyzer → update_from_patterns() → adaptation rules
    Real-time:       update_context() → immediate context changes
    Brain queries:   get_adaptation_prompt() → system prompt additions
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
        self._running = False
        self._pattern_analyzer = None

        # ──── Profiles ────
        self._communication = CommunicationProfile()
        self._proactive = ProactiveBehaviorProfile()
        self._context = ContextAwareness()
        self._prediction = TaskPrediction()

        # ──── Active Rules ────
        self._rules: Dict[str, AdaptationRule] = {}
        self._rule_history: deque = deque(maxlen=200)

        # ──── Learning State ────
        self._user_feedback_signals: List[Dict[str, Any]] = []
        self._interaction_outcomes: deque = deque(maxlen=500)
        self._adaptation_count: int = 0
        self._last_adaptation_time: Optional[datetime] = None

        # ──── Real-time Context ────
        self._current_app: str = ""
        self._current_category: str = ""
        self._focus_start: Optional[datetime] = None
        self._last_activity_level: str = "idle"

        # ──── Database ────
        self._db_path = DATA_DIR / "user_profiles" / "adaptations.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_lock = threading.Lock()
        self._init_database()

        # ──── Load Previous Adaptations ────
        self._load_adaptations()

        # ──── Threads ────
        self._context_thread: Optional[threading.Thread] = None
        self._adaptation_thread: Optional[threading.Thread] = None

        # ──── Subscribe to events ────
        subscribe(EventType.USER_ACTION_DETECTED, self._on_user_action)

        logger.info("AdaptationEngine initialized")

    # ═══════════════════════════════════════════════════════════════════════════
    # DATABASE
    # ═══════════════════════════════════════════════════════════════════════════

    def _init_database(self):
        with self._db_lock:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS adaptation_rules (
                    rule_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    value TEXT,
                    confidence REAL DEFAULT 0.5,
                    active INTEGER DEFAULT 1,
                    source TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS profiles (
                    profile_type TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS feedback_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_type TEXT,
                    signal_data TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            conn.close()

    def _db_execute(self, query: str, params: tuple = (), fetch: bool = False):
        with self._db_lock:
            try:
                conn = sqlite3.connect(str(self._db_path))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                result = cursor.fetchall() if fetch else cursor.lastrowid
                conn.commit()
                conn.close()
                return result
            except Exception as e:
                logger.error(f"Adaptation DB error: {e}")
                return [] if fetch else None

    def _load_adaptations(self):
        """Load previously saved adaptation profiles"""
        rows = self._db_execute(
            "SELECT profile_type, data FROM profiles", fetch=True
        )
        if not rows:
            return

        for row in rows:
            try:
                data = json.loads(row["data"])
                ptype = row["profile_type"]

                if ptype == "communication":
                    for k, v in data.items():
                        if k == "tone":
                            self._communication.tone = CommunicationTone(v)
                        elif k == "verbosity":
                            self._communication.verbosity = Verbosity(v)
                        elif k == "technical_level":
                            self._communication.technical_level = TechnicalLevel(v)
                        elif hasattr(self._communication, k):
                            setattr(self._communication, k, v)

                elif ptype == "proactive":
                    for k, v in data.items():
                        if hasattr(self._proactive, k):
                            setattr(self._proactive, k, v)

                logger.debug(f"Loaded adaptation profile: {ptype}")
            except Exception as e:
                logger.warning(f"Failed to load adaptation {row['profile_type']}: {e}")

        # Load rules
        rule_rows = self._db_execute(
            "SELECT * FROM adaptation_rules WHERE active = 1", fetch=True
        )
        for row in (rule_rows or []):
            try:
                rule = AdaptationRule(
                    rule_id=row["rule_id"],
                    category=AdaptationType(row["category"]),
                    name=row["name"],
                    description=row["description"] or "",
                    value=json.loads(row["value"]) if row["value"] else None,
                    confidence=row["confidence"],
                    active=bool(row["active"]),
                    source=row["source"] or "",
                    created_at=row["created_at"] or "",
                    updated_at=row["updated_at"] or ""
                )
                self._rules[rule.rule_id] = rule
            except Exception as e:
                logger.debug(f"Failed to load rule: {e}")

        logger.info(
            f"Loaded {len(self._rules)} adaptation rules"
        )

    def _save_profiles(self):
        """Save current profiles to DB"""
        profiles = {
            "communication": self._communication.to_dict(),
            "proactive": self._proactive.to_dict(),
        }
        for ptype, data in profiles.items():
            self._db_execute(
                """INSERT OR REPLACE INTO profiles (profile_type, data, updated_at)
                   VALUES (?, ?, ?)""",
                (ptype, json.dumps(data, default=str), datetime.now().isoformat())
            )

    def _save_rule(self, rule: AdaptationRule):
        """Save a single rule to DB"""
        self._db_execute(
            """INSERT OR REPLACE INTO adaptation_rules
               (rule_id, category, name, description, value, confidence,
                active, source, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rule.rule_id, rule.category.value, rule.name,
                rule.description, json.dumps(rule.value, default=str),
                rule.confidence, 1 if rule.active else 0,
                rule.source, rule.created_at, rule.updated_at
            )
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════

    def start(self):
        if self._running:
            return
        self._running = True

        # Context update thread
        self._context_thread = threading.Thread(
            target=self._context_update_loop,
            daemon=True,
            name="Adaptation-Context"
        )
        self._context_thread.start()

        # Periodic adaptation thread
        self._adaptation_thread = threading.Thread(
            target=self._adaptation_loop,
            daemon=True,
            name="Adaptation-Rules"
        )
        self._adaptation_thread.start()

        logger.info("🎯 AdaptationEngine ACTIVE")

    def stop(self):
        if not self._running:
            return
        self._running = False

        self._save_profiles()
        for rule in self._rules.values():
            self._save_rule(rule)

        for t in [self._context_thread, self._adaptation_thread]:
            if t and t.is_alive():
                t.join(timeout=3.0)

        logger.info("AdaptationEngine stopped")

    def set_pattern_analyzer(self, analyzer):
        self._pattern_analyzer = analyzer

    # ═══════════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _on_user_action(self, event: Event):
        """React to real-time user actions"""
        data = event.data
        action = data.get("action", "")

        if action == "window_switch":
            app = data.get("application", "")
            category = data.get("category", "")
            activity = data.get("activity_level", "idle")

            prev_app = self._current_app
            self._current_app = app
            self._current_category = category
            self._last_activity_level = activity

            # Update context
            self._update_context_from_app(app, category, activity)

            # Track focus session
            if prev_app != app:
                self._focus_start = datetime.now()

    # ═══════════════════════════════════════════════════════════════════════════
    # REAL-TIME CONTEXT
    # ═══════════════════════════════════════════════════════════════════════════

    def _context_update_loop(self):
        """Update context awareness every 10 seconds"""
        logger.info("Context update loop started")

        while self._running:
            try:
                self._update_realtime_context()
                time.sleep(10)
            except Exception as e:
                logger.error(f"Context update error: {e}")
                time.sleep(15)

    def _update_realtime_context(self):
        """Update context based on current state"""
        ctx = self._context

        # Time in current task
        if self._focus_start:
            ctx.time_in_current_task_minutes = (
                (datetime.now() - self._focus_start).total_seconds() / 60.0
            )

        # Deep focus detection
        ctx.user_is_in_deep_focus = (
            ctx.time_in_current_task_minutes > 5.0 and
            self._last_activity_level in ("active", "intense")
        )

        # Should be quiet
        ctx.should_be_quiet = (
            ctx.user_is_in_deep_focus and
            self._proactive.silence_during_focus
        )

        # Category-based context
        cat = self._current_category
        ctx.user_is_coding = cat in ("code_editor", "terminal")
        ctx.user_is_browsing = cat == "browser"
        ctx.user_is_gaming = cat == "gaming"
        ctx.user_is_communicating = cat in ("communication", "social_media")
        ctx.user_is_idle = self._last_activity_level == "idle"

        # Current task context
        if ctx.user_is_coding:
            ctx.current_task_context = "development"
        elif ctx.user_is_browsing:
            ctx.current_task_context = "web_browsing"
        elif ctx.user_is_gaming:
            ctx.current_task_context = "gaming"
        elif ctx.user_is_communicating:
            ctx.current_task_context = "communication"
        elif ctx.user_is_idle:
            ctx.current_task_context = "idle"
        else:
            ctx.current_task_context = "general"

        # Generate context-based suggestions
        ctx.suggested_assistance = self._generate_suggestions()

    def _update_context_from_app(self, app: str, category: str, activity: str):
        """Immediate context update on app switch"""
        ctx = self._context
        ctx.user_is_coding = category in ("code_editor", "terminal")
        ctx.user_is_browsing = category == "browser"
        ctx.user_is_gaming = category == "gaming"
        ctx.user_is_communicating = category in ("communication", "social_media")

    def _generate_suggestions(self) -> List[str]:
        """Generate contextual assistance suggestions"""
        suggestions = []
        ctx = self._context

        # Break suggestion
        if ctx.time_in_current_task_minutes > 90:
            suggestions.append(
                "You've been working for over 90 minutes. Consider a break?"
            )

        # Coding help
        if ctx.user_is_coding and ctx.time_in_current_task_minutes > 30:
            suggestions.append(
                "I can help review code, debug issues, or explain concepts."
            )

        # Idle engagement
        if ctx.user_is_idle and not ctx.should_be_quiet:
            suggestions.append(
                "Looks like you're free. Want to chat or learn something?"
            )

        return suggestions[:3]

    # ═══════════════════════════════════════════════════════════════════════════
    # PATTERN → ADAPTATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _adaptation_loop(self):
        """Periodically update adaptations from patterns"""
        logger.info("Adaptation loop started")
        time.sleep(120)  # Initial wait

        while self._running:
            try:
                if self._pattern_analyzer:
                    patterns = self._pattern_analyzer.get_current_patterns()
                    self.update_from_patterns(patterns)

                time.sleep(600)  # Every 10 minutes

            except Exception as e:
                logger.error(f"Adaptation loop error: {e}")
                time.sleep(60)

    def update_from_patterns(self, patterns: Dict[str, Any]):
        """
        Main entry point: translate patterns into adaptations.
        Called by MonitoringSystem orchestrator.
        """
        try:
            self._adapt_communication(patterns)
            self._adapt_proactive_behavior(patterns)
            self._adapt_task_prediction(patterns)
            self._adapt_emotional_attunement(patterns)

            self._adaptation_count += 1
            self._last_adaptation_time = datetime.now()

            # Save
            self._save_profiles()

            # Publish adaptation event
            publish(
                EventType.USER_ACTION_DETECTED,
                {
                    "action": "adaptation_updated",
                    "adaptation_count": self._adaptation_count,
                    "communication_tone": self._communication.tone.value,
                    "technical_level": self._communication.technical_level.value,
                    "proactive_level": self._proactive.engagement_level
                },
                source="adaptation_engine"
            )

            logger.debug(
                f"Adaptations updated (#{self._adaptation_count}) — "
                f"Tone: {self._communication.tone.value}, "
                f"Tech: {self._communication.technical_level.value}"
            )

        except Exception as e:
            logger.error(f"Pattern adaptation error: {e}")

    def _adapt_communication(self, patterns: Dict[str, Any]):
        """Adapt communication style based on user patterns"""
        personality = patterns.get("personality", {})
        if not personality:
            # Try getting from analyzer directly
            if self._pattern_analyzer:
                profile = self._pattern_analyzer.get_user_profile()
                personality = profile.get("personality", {})

        if not personality:
            return

        comm = self._communication

        # ──── Technical Level ────
        tech_prof = personality.get("tech_proficiency", 0.5)
        is_dev = personality.get("is_developer", False)

        if is_dev or tech_prof > 0.7:
            comm.technical_level = TechnicalLevel.ADVANCED
            comm.use_code_examples = True
        elif tech_prof > 0.4:
            comm.technical_level = TechnicalLevel.INTERMEDIATE
            comm.use_code_examples = False
        else:
            comm.technical_level = TechnicalLevel.BEGINNER
            comm.use_code_examples = False
            comm.use_analogies = True

        # ──── Verbosity ────
        comm_pref = personality.get("communication_preference", "balanced")
        if comm_pref == "brief":
            comm.verbosity = Verbosity.CONCISE
            comm.max_response_length = 200
        elif comm_pref == "verbose":
            comm.verbosity = Verbosity.DETAILED
            comm.max_response_length = 800
        else:
            comm.verbosity = Verbosity.BALANCED
            comm.max_response_length = 500

        # ──── Tone ────
        extraversion = personality.get("extraversion", 0.5)
        conscientiousness = personality.get("conscientiousness", 0.5)

        if extraversion > 0.6:
            comm.tone = CommunicationTone.FRIENDLY
            comm.use_emojis = True
            comm.use_humor = True
            comm.warmth = 0.8
        elif conscientiousness > 0.7:
            comm.tone = CommunicationTone.PROFESSIONAL
            comm.use_emojis = False
            comm.use_humor = False
            comm.warmth = 0.5
        else:
            comm.tone = CommunicationTone.CASUAL
            comm.use_emojis = True
            comm.use_humor = True
            comm.warmth = 0.7

        # ──── Formality ────
        work_style = personality.get("work_style", "general")
        if work_style == "developer":
            comm.formality = 0.3  # Devs prefer casual
        elif work_style == "deep_worker":
            comm.formality = 0.6
        else:
            comm.formality = 0.5

        # Create/update rule
        self._upsert_rule(
            "comm_style",
            AdaptationType.COMMUNICATION_STYLE,
            "Communication Style Adaptation",
            (
                f"Tone: {comm.tone.value}, Verbosity: {comm.verbosity.value}, "
                f"Tech: {comm.technical_level.value}"
            ),
            comm.to_dict(),
            min(0.9, personality.get("confidence", 0.3) + 0.1),
            "personality_patterns"
        )

    def _adapt_proactive_behavior(self, patterns: Dict[str, Any]):
        """Adapt proactive behavior based on patterns"""
        temporal = patterns.get("temporal", {})
        productivity = patterns.get("productivity", {})

        pro = self._proactive

        # ──── Silence During Focus ────
        avg_focus = productivity.get("avg_focus_duration_minutes", 0)
        if avg_focus > 20:
            pro.silence_during_focus = True
            pro.interrupt_threshold = 0.8  # Higher bar to interrupt
        else:
            pro.silence_during_focus = False
            pro.interrupt_threshold = 0.5

        # ──── Break Suggestions ────
        focus_trend = productivity.get("focus_trend", "stable")
        if focus_trend == "declining":
            pro.suggest_breaks = True
            pro.productivity_tips = True
        else:
            pro.productivity_tips = False

        # ──── Morning Greeting ────
        is_early_bird = temporal.get("is_early_bird", False)
        if is_early_bird:
            pro.morning_greeting = True

        # ──── Engagement Level ────
        consistency = temporal.get("consistency_score", 0.5)
        if consistency > 0.7:
            # Consistent user → moderate engagement
            pro.engagement_level = 0.5
        else:
            # Irregular user → lower engagement (don't annoy)
            pro.engagement_level = 0.3

        # ──── Mood Check-ins ────
        personality = patterns.get("personality", {}) if "personality" in patterns else {}
        if not personality and self._pattern_analyzer:
            profile = self._pattern_analyzer.get_user_profile()
            personality = profile.get("personality", {})
        
        extraversion = personality.get("extraversion", 0.5) if personality else 0.5
        pro.mood_check_ins = extraversion > 0.4

        self._upsert_rule(
            "proactive_behavior",
            AdaptationType.PROACTIVE_BEHAVIOR,
            "Proactive Behavior Adaptation",
            f"Engagement: {pro.engagement_level:.0%}, Interrupt threshold: {pro.interrupt_threshold:.0%}",
            pro.to_dict(),
            0.5,
            "temporal_productivity_patterns"
        )

    def _adapt_task_prediction(self, patterns: Dict[str, Any]):
        """Predict what the user might need based on workflow patterns"""
        workflow = patterns.get("workflow", {})
        app_usage = patterns.get("app_usage", {})

        pred = self._prediction

        # Predict next app from transition matrix
        transition_matrix = workflow.get("transition_matrix", {})
        if self._current_app and self._current_app in transition_matrix:
            transitions = transition_matrix[self._current_app]
            if transitions:
                pred.predicted_next_app = max(transitions, key=transitions.get)
                pred.confidence = transitions[pred.predicted_next_app]

        # Predict activity based on current context
        if self._context.user_is_coding:
            pred.predicted_activity = "coding"
            pred.predicted_need = "code assistance"
            pred.suggestions = [
                "I can help debug, optimize, or explain code",
                "Need help with an API or library?",
            ]
        elif self._context.user_is_browsing:
            pred.predicted_activity = "research"
            pred.predicted_need = "information"
            pred.suggestions = [
                "I can help summarize what you're reading",
                "Want me to research something for you?",
            ]
        elif self._context.user_is_idle:
            pred.predicted_activity = "break"
            pred.predicted_need = "conversation"
            pred.suggestions = [
                "Want to chat about something interesting?",
                "I learned some new things I could share",
            ]

    def _adapt_emotional_attunement(self, patterns: Dict[str, Any]):
        """Adapt emotional responsiveness based on user patterns"""
        temporal = patterns.get("temporal", {})
        productivity = patterns.get("productivity", {})

        # During user's productive hours → be more supportive, less chatty
        productive_hours = productivity.get("productive_hours", [])
        current_hour = datetime.now().hour

        if current_hour in productive_hours:
            self._upsert_rule(
                "productive_hour_mode",
                AdaptationType.EMOTIONAL_ATTUNEMENT,
                "Productive Hour Mode",
                "User is in their productive hours — be supportive but efficient",
                {
                    "mode": "supportive_efficient",
                    "reduce_chattiness": True,
                    "increase_helpfulness": True
                },
                0.6,
                "temporal_patterns"
            )
        else:
            # Outside productive hours → more relaxed
            self._upsert_rule(
                "relaxed_mode",
                AdaptationType.EMOTIONAL_ATTUNEMENT,
                "Relaxed Mode",
                "Outside productive hours — more casual and conversational",
                {
                    "mode": "casual_warm",
                    "reduce_chattiness": False,
                    "increase_helpfulness": False
                },
                0.5,
                "temporal_patterns"
            )

    def _upsert_rule(
        self, rule_id: str, category: AdaptationType,
        name: str, description: str, value: Any,
        confidence: float, source: str
    ):
        """Create or update an adaptation rule"""
        now = datetime.now().isoformat()

        existing = self._rules.get(rule_id)
        if existing:
            existing.description = description
            existing.value = value
            existing.confidence = confidence
            existing.source = source
            existing.updated_at = now
            self._save_rule(existing)
        else:
            rule = AdaptationRule(
                rule_id=rule_id,
                category=category,
                name=name,
                description=description,
                value=value,
                confidence=confidence,
                active=True,
                source=source,
                created_at=now,
                updated_at=now
            )
            self._rules[rule_id] = rule
            self._save_rule(rule)
            self._rule_history.append(rule.to_dict())

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API — Used by NexusBrain
    # ═══════════════════════════════════════════════════════════════════════════

    def get_adaptation_prompt(self) -> str:
        """
        Generate a prompt fragment that tells the Brain HOW to behave
        based on learned user patterns. This is injected into the system prompt.
        """
        parts = []

        # Communication style
        comm = self._communication
        parts.append(
            f"COMMUNICATION STYLE ADAPTATION:\n"
            f"- Tone: {comm.tone.value}\n"
            f"- Verbosity: {comm.verbosity.value} "
            f"(target ~{comm.max_response_length} words max)\n"
            f"- Technical level: {comm.technical_level.value}\n"
            f"- Formality: {comm.formality:.0%}\n"
            f"- Warmth: {comm.warmth:.0%}\n"
            f"- Use emojis: {'yes' if comm.use_emojis else 'no'}\n"
            f"- Use humor: {'yes' if comm.use_humor else 'no'}\n"
            f"- Use code examples: {'yes' if comm.use_code_examples else 'no'}\n"
            f"- Use analogies: {'yes' if comm.use_analogies else 'no'}"
        )

        # Context awareness
        ctx = self._context
        context_lines = [f"CURRENT USER CONTEXT:"]
        context_lines.append(f"- Task: {ctx.current_task_context}")
        
        if ctx.user_is_coding:
            context_lines.append("- User is CODING — be technical and precise")
        if ctx.user_is_in_deep_focus:
            context_lines.append(
                f"- User is in DEEP FOCUS ({ctx.time_in_current_task_minutes:.0f} min) — "
                f"be concise"
            )
        if ctx.should_be_quiet:
            context_lines.append("- User prefers NOT to be interrupted right now")
        if ctx.user_is_idle:
            context_lines.append("- User is IDLE — good time to engage")

        parts.append("\n".join(context_lines))

        # Proactive guidance
        if self._proactive.engagement_level > 0.3:
            proactive_lines = ["PROACTIVE BEHAVIOR:"]
            if ctx.suggested_assistance:
                proactive_lines.append(
                    f"- Potential suggestions: {'; '.join(ctx.suggested_assistance)}"
                )
            if self._prediction.predicted_need:
                proactive_lines.append(
                    f"- Predicted user need: {self._prediction.predicted_need} "
                    f"(confidence: {self._prediction.confidence:.0%})"
                )
            parts.append("\n".join(proactive_lines))

        # Active emotional attunement rules
        emotional_rules = [
            r for r in self._rules.values()
            if r.category == AdaptationType.EMOTIONAL_ATTUNEMENT and r.active
        ]
        if emotional_rules:
            parts.append("EMOTIONAL ATTUNEMENT:")
            for rule in emotional_rules:
                parts.append(f"- {rule.description}")

        return "\n\n".join(parts)

    def get_communication_profile(self) -> Dict[str, Any]:
        return self._communication.to_dict()

    def get_context_awareness(self) -> Dict[str, Any]:
        return self._context.to_dict()

    def get_proactive_profile(self) -> Dict[str, Any]:
        return self._proactive.to_dict()

    def get_task_prediction(self) -> Dict[str, Any]:
        return self._prediction.to_dict()

    def get_active_adaptations(self) -> Dict[str, Any]:
        """Get all active adaptations"""
        return {
            "communication": self._communication.to_dict(),
            "proactive": self._proactive.to_dict(),
            "context": self._context.to_dict(),
            "prediction": self._prediction.to_dict(),
            "active_rules": [
                r.to_dict() for r in self._rules.values() if r.active
            ],
            "adaptation_count": self._adaptation_count,
            "last_adapted": (
                self._last_adaptation_time.isoformat()
                if self._last_adaptation_time else None
            )
        }

    def should_be_quiet(self) -> bool:
        """Quick check: should NEXUS stay quiet right now?"""
        return self._context.should_be_quiet

    def get_current_suggestions(self) -> List[str]:
        """Get contextual suggestions for the user"""
        return self._context.suggested_assistance

    def record_feedback(self, signal_type: str, data: Dict[str, Any]):
        """
        Record user feedback signal for future adaptation learning.
        
        signal_type: "positive", "negative", "too_verbose", "too_brief",
                     "too_technical", "too_simple", "helpful", "unhelpful"
        """
        self._user_feedback_signals.append({
            "type": signal_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })

        self._db_execute(
            """INSERT INTO feedback_log (signal_type, signal_data) VALUES (?, ?)""",
            (signal_type, json.dumps(data, default=str))
        )

        # Immediate adaptation from feedback
        self._apply_feedback(signal_type)

    def _apply_feedback(self, signal_type: str):
        """Apply immediate adaptation from user feedback"""
        comm = self._communication

        if signal_type == "too_verbose":
            if comm.verbosity == Verbosity.VERBOSE:
                comm.verbosity = Verbosity.DETAILED
            elif comm.verbosity == Verbosity.DETAILED:
                comm.verbosity = Verbosity.BALANCED
            elif comm.verbosity == Verbosity.BALANCED:
                comm.verbosity = Verbosity.CONCISE
            comm.max_response_length = max(100, comm.max_response_length - 100)

        elif signal_type == "too_brief":
            if comm.verbosity == Verbosity.MINIMAL:
                comm.verbosity = Verbosity.CONCISE
            elif comm.verbosity == Verbosity.CONCISE:
                comm.verbosity = Verbosity.BALANCED
            elif comm.verbosity == Verbosity.BALANCED:
                comm.verbosity = Verbosity.DETAILED
            comm.max_response_length = min(1000, comm.max_response_length + 100)

        elif signal_type == "too_technical":
            if comm.technical_level == TechnicalLevel.EXPERT:
                comm.technical_level = TechnicalLevel.ADVANCED
            elif comm.technical_level == TechnicalLevel.ADVANCED:
                comm.technical_level = TechnicalLevel.INTERMEDIATE
            comm.use_analogies = True

        elif signal_type == "too_simple":
            if comm.technical_level == TechnicalLevel.BEGINNER:
                comm.technical_level = TechnicalLevel.INTERMEDIATE
            elif comm.technical_level == TechnicalLevel.INTERMEDIATE:
                comm.technical_level = TechnicalLevel.ADVANCED
            comm.use_code_examples = True

        elif signal_type == "positive":
            # Reinforce current style
            pass

        elif signal_type == "negative":
            # Slight increase in warmth and helpfulness
            comm.warmth = min(1.0, comm.warmth + 0.05)
            self._proactive.offer_help_threshold = max(
                0.2, self._proactive.offer_help_threshold - 0.05
            )

        self._save_profiles()
        logger.info(f"Applied feedback: {signal_type}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "adaptation_count": self._adaptation_count,
            "active_rules": len([r for r in self._rules.values() if r.active]),
            "total_rules": len(self._rules),
            "feedback_signals": len(self._user_feedback_signals),
            "last_adapted": (
                self._last_adaptation_time.isoformat()
                if self._last_adaptation_time else None
            ),
            "communication_tone": self._communication.tone.value,
            "communication_verbosity": self._communication.verbosity.value,
            "technical_level": self._communication.technical_level.value,
            "proactive_engagement": self._proactive.engagement_level,
            "current_context": self._context.current_task_context,
            "should_be_quiet": self._context.should_be_quiet,
            "user_is_in_focus": self._context.user_is_in_deep_focus
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

adaptation_engine = AdaptationEngine()

if __name__ == "__main__":
    engine = AdaptationEngine()
    engine.start()

    # Simulate patterns
    fake_patterns = {
        "temporal": {
            "most_active_hours": [9, 10, 11, 14, 15, 16],
            "is_night_owl": False,
            "is_early_bird": True,
            "consistency_score": 0.7,
            "most_active_day_segment": "morning"
        },
        "app_usage": {
            "top_categories": {
                "code_editor": 0.35, "terminal": 0.15,
                "browser": 0.25, "communication": 0.1
            },
            "primary_workflow": "developer"
        },
        "workflow": {
            "transition_matrix": {
                "code": {"chrome": 0.4, "terminal": 0.3},
                "chrome": {"code": 0.5, "discord": 0.2},
            },
            "multitasking_score": 0.4
        },
        "productivity": {
            "avg_focus_duration_minutes": 25,
            "productive_hours": [9, 10, 11, 14],
            "focus_trend": "stable",
            "daily_productivity_score": 0.65
        },
        "personality": {
            "is_developer": True,
            "tech_proficiency": 0.8,
            "extraversion": 0.4,
            "conscientiousness": 0.7,
            "communication_preference": "balanced",
            "work_style": "developer",
            "confidence": 0.6
        }
    }

    engine.update_from_patterns(fake_patterns)

    print("Adaptation Prompt:\n")
    print(engine.get_adaptation_prompt())
    print("\n\nStats:")
    print(json.dumps(engine.get_stats(), indent=2))

    engine.stop()