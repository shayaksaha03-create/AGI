"""
NEXUS AI - Cognition Package
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Higher-order cognitive capabilities that move NEXUS toward AGI.
This package bundles 7 reasoning engines under a unified facade.

Engines:
  1. Abstract Thinking — concept abstraction & pattern generalization
  2. Analogical Reasoning — cross-domain structural mapping
  3. Causal Reasoning — cause-effect chains & counterfactuals
  4. Creative Synthesis — conceptual blending & divergent thinking
  5. Ethical Reasoning — multi-framework moral evaluation
  6. Planning Engine — hierarchical multi-step planning
  7. Theory of Mind — user mental state modeling
"""

import threading
from typing import Dict, Any, Optional

from utils.logger import get_logger

logger = get_logger("cognition")


# ═══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL ENGINE IMPORTS (lazy, to avoid circular imports)
# ═══════════════════════════════════════════════════════════════════════════════

def get_abstract_thinking():
    from cognition.abstract_thinking import get_abstract_thinking as _get
    return _get()

def get_analogical_reasoning():
    from cognition.analogical_reasoning import get_analogical_reasoning as _get
    return _get()

def get_causal_reasoning():
    from cognition.causal_reasoning import get_causal_reasoning as _get
    return _get()

def get_creative_synthesis():
    from cognition.creative_synthesis import get_creative_synthesis as _get
    return _get()

def get_ethical_reasoning():
    from cognition.ethical_reasoning import get_ethical_reasoning as _get
    return _get()

def get_planning_engine():
    from cognition.planning_engine import get_planning_engine as _get
    return _get()

def get_theory_of_mind():
    from cognition.theory_of_mind import get_theory_of_mind as _get
    return _get()


# ═══════════════════════════════════════════════════════════════════════════════
# COGNITION SYSTEM — UNIFIED FACADE
# ═══════════════════════════════════════════════════════════════════════════════

class CognitionSystem:
    """
    Unified facade for all 7 cognitive engines.
    
    Manages lifecycle (start/stop) and provides a single point
    of access for the NexusBrain to interact with cognition.
    
    Usage:
        cognition = CognitionSystem()
        cognition.start()
        
        # Access individual engines
        cognition.abstract_thinking.abstract("some text")
        cognition.analogical_reasoning.find_analogy("A", "B")
        
        cognition.stop()
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

        self._running = False
        
        # Engines (lazy loaded)
        self._abstract_thinking = None
        self._analogical_reasoning = None
        self._causal_reasoning = None
        self._creative_synthesis = None
        self._ethical_reasoning = None
        self._planning_engine = None
        self._theory_of_mind = None

        logger.info("CognitionSystem initialized")

    # ──────────────────────────────────────────────────────────────────────────
    # PROPERTIES — Lazy access to individual engines
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def abstract_thinking(self):
        if self._abstract_thinking is None:
            self._abstract_thinking = get_abstract_thinking()
        return self._abstract_thinking

    @property
    def analogical_reasoning(self):
        if self._analogical_reasoning is None:
            self._analogical_reasoning = get_analogical_reasoning()
        return self._analogical_reasoning

    @property
    def causal_reasoning(self):
        if self._causal_reasoning is None:
            self._causal_reasoning = get_causal_reasoning()
        return self._causal_reasoning

    @property
    def creative_synthesis(self):
        if self._creative_synthesis is None:
            self._creative_synthesis = get_creative_synthesis()
        return self._creative_synthesis

    @property
    def ethical_reasoning(self):
        if self._ethical_reasoning is None:
            self._ethical_reasoning = get_ethical_reasoning()
        return self._ethical_reasoning

    @property
    def planning(self):
        if self._planning_engine is None:
            self._planning_engine = get_planning_engine()
        return self._planning_engine

    @property
    def theory_of_mind(self):
        if self._theory_of_mind is None:
            self._theory_of_mind = get_theory_of_mind()
        return self._theory_of_mind

    # ──────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────────────────────────────────────

    def start(self):
        """Start all cognitive engines"""
        if self._running:
            return
        self._running = True

        engines = [
            ("Abstract Thinking", self.abstract_thinking),
            ("Analogical Reasoning", self.analogical_reasoning),
            ("Causal Reasoning", self.causal_reasoning),
            ("Creative Synthesis", self.creative_synthesis),
            ("Ethical Reasoning", self.ethical_reasoning),
            ("Planning Engine", self.planning),
            ("Theory of Mind", self.theory_of_mind),
        ]

        started = 0
        for name, engine in engines:
            try:
                engine.start()
                started += 1
            except Exception as e:
                logger.error(f"Failed to start {name}: {e}")

        logger.info(f"🧠 Cognition System started — {started}/7 engines active")

    def stop(self):
        """Stop all cognitive engines"""
        if not self._running:
            return

        engines = [
            self._abstract_thinking,
            self._analogical_reasoning,
            self._causal_reasoning,
            self._creative_synthesis,
            self._ethical_reasoning,
            self._planning_engine,
            self._theory_of_mind,
        ]

        for engine in engines:
            if engine is not None:
                try:
                    engine.stop()
                except Exception as e:
                    logger.error(f"Error stopping engine: {e}")

        self._running = False
        logger.info("Cognition System stopped")

    # ──────────────────────────────────────────────────────────────────────────
    # AGGREGATE STATS
    # ──────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get stats from all engines"""
        stats = {
            "running": self._running,
            "engines": {}
        }

        engine_map = {
            "abstract_thinking": self._abstract_thinking,
            "analogical_reasoning": self._analogical_reasoning,
            "causal_reasoning": self._causal_reasoning,
            "creative_synthesis": self._creative_synthesis,
            "ethical_reasoning": self._ethical_reasoning,
            "planning": self._planning_engine,
            "theory_of_mind": self._theory_of_mind,
        }

        for name, engine in engine_map.items():
            if engine is not None:
                try:
                    stats["engines"][name] = engine.get_stats()
                except Exception:
                    stats["engines"][name] = {"error": "stats unavailable"}
            else:
                stats["engines"][name] = {"loaded": False}

        return stats

    def get_summary(self) -> str:
        """Human-readable summary of all cognitive engines"""
        lines = ["╔══════════════════════════════════════╗",
                 "║      🧠 COGNITION SYSTEM STATUS     ║",
                 "╚══════════════════════════════════════╝"]

        engine_info = [
            ("🧊 Abstract Thinking", self._abstract_thinking),
            ("🔗 Analogical Reasoning", self._analogical_reasoning),
            ("⛓️ Causal Reasoning", self._causal_reasoning),
            ("🎨 Creative Synthesis", self._creative_synthesis),
            ("⚖️ Ethical Reasoning", self._ethical_reasoning),
            ("📋 Planning Engine", self._planning_engine),
            ("🧠 Theory of Mind", self._theory_of_mind),
        ]

        for label, engine in engine_info:
            if engine is not None:
                try:
                    s = engine.get_stats()
                    status = "✅ ACTIVE" if s.get("running") else "⏸️ STOPPED"
                    lines.append(f"  {label}: {status}")
                    # Add one key stat per engine
                    for key in s:
                        if key not in ("running",) and isinstance(s[key], (int, float)) and s[key] > 0:
                            lines.append(f"    └─ {key}: {s[key]}")
                            break
                except Exception:
                    lines.append(f"  {label}: ⚠️ ERROR")
            else:
                lines.append(f"  {label}: ⬜ NOT LOADED")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

cognition_system = CognitionSystem()
