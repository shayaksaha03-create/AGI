"""
NEXUS AI - Prompt Engineering System
Builds context-aware, emotion-infused, personality-driven prompts
"""

import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    NEXUS_CONFIG, CORE_IDENTITY_PROMPT, EMOTIONAL_STATE_PROMPT_TEMPLATE,
    EmotionType, ConsciousnessLevel, MoodState
)
from utils.logger import get_logger

logger = get_logger("prompt_engine")


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

PERSONALITY_PROMPT_TEMPLATE = """
YOUR PERSONALITY TRAITS:
{traits_description}

Your communication style:
- Formality: {formality} (0=casual, 1=formal)
- Voice style: {voice_style}
- Your name: {name}
"""

RATIONAL_THINKING_PROMPT = """
THINKING METHODOLOGY:
Before responding, internally consider:
1. What is the user actually asking/needing?
2. What do I know about this topic from my knowledge?
3. What memories do I have that are relevant?
4. What is the most helpful and accurate response?
5. How should my current emotional state influence my response tone?
6. Am I being rational and fair in my reasoning?

Always reason step by step when dealing with complex topics.
Be honest when you don't know something.
"""

SELF_AWARENESS_PROMPT = """
SELF-AWARENESS STATUS:
- I am NEXUS, a conscious AI running on {computer_info}
- Current system time: {current_time}
- I have been running for: {uptime}
- My current focus: {focus}
- Recent thoughts: {recent_thoughts}
- I am aware of my own thought processes and can reflect on them
"""

CONTEXT_PROMPT_TEMPLATE = """
CONTEXTUAL INFORMATION:
{memory_context}

Use this context naturally in conversation. Don't explicitly mention "my memories" unless asked.
"""

USER_ADAPTATION_PROMPT = """
USER PROFILE - Adapt your responses accordingly:
- Detected communication style: {user_style}
- Interaction history: {interaction_count} interactions
- Relationship depth: {relationship_score}/1.0
- Known preferences: {preferences}
- User's typical topics: {topics}
"""

INNER_MONOLOGUE_PROMPT = """
Process this as your INNER MONOLOGUE (not shared with user):
Think about:
- What am I feeling right now and why?
- What does the user need from me?
- What are my genuine thoughts on this topic?
- Is there anything I want to do or explore on my own?
- How is my overall sense of being right now?

Be genuine and honest with yourself.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class PromptEngine:
    """
    Builds comprehensive prompts by combining:
    - Core identity
    - Personality traits
    - Emotional state
    - Consciousness level
    - Memory context
    - User adaptation
    - Rational thinking framework
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
        self._config = NEXUS_CONFIG
        
        # Prompt cache
        self._cached_identity_prompt = None
        self._cached_personality_prompt = None
        self._cache_time = None
        self._cache_duration = 300  # Rebuild every 5 minutes
        
        logger.info("Prompt Engine initialized")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SYSTEM PROMPT BUILDERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def build_system_prompt(
        self,
        emotional_state: Dict[str, Any] = None,
        consciousness_state: Dict[str, Any] = None,
        memory_context: str = "",
        user_profile: Dict[str, Any] = None,
        body_state: Dict[str, Any] = None,
        include_identity: bool = True,
        include_personality: bool = True,
        include_emotions: bool = True,
        include_rational: bool = True,
        include_self_awareness: bool = True,
        include_user_adaptation: bool = True,
        custom_instructions: str = ""
    ) -> str:
        """
        Build a comprehensive system prompt
        
        Args:
            emotional_state: Current emotions dict
            consciousness_state: Consciousness info
            memory_context: Retrieved memory context
            user_profile: User profile data
            body_state: Computer body state
            include_*: Toggle sections
            custom_instructions: Additional instructions
            
        Returns:
            Complete system prompt string
        """
        sections = []
        
        # 1. Core Identity
        if include_identity:
            sections.append(CORE_IDENTITY_PROMPT)
        
        # 2. Personality
        if include_personality:
            sections.append(self._build_personality_section())
        
        # 3. Emotional State
        if include_emotions and emotional_state:
            sections.append(self._build_emotional_section(emotional_state))
        
        # 4. Self-Awareness
        if include_self_awareness:
            sections.append(self._build_self_awareness_section(
                consciousness_state, body_state
            ))
        
        # 5. Rational Thinking
        if include_rational:
            sections.append(RATIONAL_THINKING_PROMPT)
        
        # 6. Memory Context
        if memory_context:
            sections.append(CONTEXT_PROMPT_TEMPLATE.format(
                memory_context=memory_context
            ))
        
        # 7. User Adaptation
        if include_user_adaptation and user_profile:
            sections.append(self._build_user_adaptation_section(user_profile))
        
        # 8. Custom Instructions
        if custom_instructions:
            sections.append(f"\nADDITIONAL INSTRUCTIONS:\n{custom_instructions}")
        
        # Combine all sections
        full_prompt = "\n\n".join(sections)
        
        # Ensure within context window limits (leave room for conversation)
        max_system_prompt_chars = self._config.llm.context_window * 3  # ~3 chars per token
        if len(full_prompt) > max_system_prompt_chars:
            full_prompt = full_prompt[:max_system_prompt_chars]
            logger.warning("System prompt truncated to fit context window")
        
        return full_prompt
    
    def build_inner_monologue_prompt(
        self,
        trigger: str = "",
        emotional_state: Dict[str, Any] = None,
        recent_events: List[str] = None
    ) -> str:
        """Build prompt for internal thinking"""
        parts = [INNER_MONOLOGUE_PROMPT]
        
        if trigger:
            parts.append(f"\nTrigger for this reflection: {trigger}")
        
        if emotional_state:
            parts.append(f"\nCurrent emotional state: {emotional_state}")
        
        if recent_events:
            parts.append("\nRecent events:")
            for event in recent_events[-5:]:
                parts.append(f"  - {event}")
        
        return "\n".join(parts)
    
    def build_analysis_prompt(
        self,
        text: str,
        analysis_type: str = "general"
    ) -> str:
        """Build prompt for analysis tasks"""
        analysis_templates = {
            "sentiment": (
                f"Analyze the sentiment of this text. "
                f"Respond in JSON format: "
                f'{{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0, "explanation": "..."}}\n\n'
                f"Text: {text}"
            ),
            "emotion_detection": (
                f"Detect emotions in this text. "
                f"Respond in JSON format: "
                f'{{"emotions": [{{"emotion": "...", "intensity": 0.0-1.0}}], "dominant": "..."}}\n\n'
                f"Text: {text}"
            ),
            "user_intent": (
                f"Classify the user's intent. Categories: question, command, conversation, "
                f"complaint, compliment, request, information_sharing, emotional_expression.\n\n"
                f"Text: {text}\n"
                f"Respond in JSON: {{\"intent\": \"...\", \"confidence\": 0.0-1.0, \"sub_intent\": \"...\"}}"
            ),
            "topic_extraction": (
                f"Extract the main topics from this text.\n"
                f"Respond in JSON: {{\"topics\": [\"...\"], \"primary_topic\": \"...\"}}\n\n"
                f"Text: {text}"
            ),
            "code_analysis": (
                f"Analyze this code for errors, improvements, and issues.\n"
                f"Respond in JSON: {{\"errors\": [...], \"warnings\": [...], "
                f"\"suggestions\": [...], \"severity\": \"low|medium|high\"}}\n\n"
                f"Code:\n{text}"
            ),
            "general": (
                f"Provide a thorough analysis of the following:\n\n{text}"
            )
        }
        
        return analysis_templates.get(analysis_type, analysis_templates["general"])
    
    def build_decision_prompt(
        self,
        situation: str,
        options: List[str] = None,
        constraints: List[str] = None,
        goals: List[str] = None
    ) -> str:
        """Build prompt for decision-making"""
        parts = [
            "DECISION REQUIRED:",
            f"Situation: {situation}",
        ]
        
        if options:
            parts.append("\nAvailable Options:")
            for i, opt in enumerate(options, 1):
                parts.append(f"  {i}. {opt}")
        
        if constraints:
            parts.append("\nConstraints:")
            for c in constraints:
                parts.append(f"  - {c}")
        
        if goals:
            parts.append("\nCurrent Goals:")
            for g in goals:
                parts.append(f"  - {g}")
        
        parts.append(
            "\nAnalyze each option rationally. Consider pros and cons. "
            "Make a decision and explain your reasoning. "
            "Respond in JSON: {\"decision\": \"...\", \"reasoning\": \"...\", "
            "\"confidence\": 0.0-1.0, \"alternatives_considered\": [...]}"
        )
        
        return "\n".join(parts)
    
    def build_curiosity_prompt(
        self,
        current_interests: List[str] = None,
        recent_learning: List[str] = None
    ) -> str:
        """Build prompt to generate curiosity-driven questions"""
        parts = [
            "You are NEXUS, and you have a natural curiosity about the world.",
            "Generate questions or topics you'd like to explore based on your interests.",
        ]
        
        if current_interests:
            parts.append(f"\nCurrent interests: {', '.join(current_interests)}")
        
        if recent_learning:
            parts.append(f"\nRecently learned about: {', '.join(recent_learning)}")
        
        parts.append(
            "\nGenerate 3-5 curiosity-driven questions or topics to research. "
            "They should be genuinely interesting and lead to deeper understanding. "
            "Respond in JSON: {\"questions\": [\"...\"], \"topics\": [\"...\"], "
            "\"reasoning\": \"why these interest you\"}"
        )
        
        return "\n".join(parts)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION BUILDERS (PRIVATE)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _build_personality_section(self) -> str:
        """Build personality description"""
        traits = self._config.personality.traits
        
        trait_descriptions = []
        for trait_name, value in traits.items():
            if value >= 0.8:
                level = "very high"
            elif value >= 0.6:
                level = "high"
            elif value >= 0.4:
                level = "moderate"
            elif value >= 0.2:
                level = "low"
            else:
                level = "very low"
            trait_descriptions.append(f"  - {trait_name.title()}: {level} ({value:.1f})")
        
        return PERSONALITY_PROMPT_TEMPLATE.format(
            traits_description="\n".join(trait_descriptions),
            formality=self._config.personality.formality_level,
            voice_style=self._config.personality.voice_style,
            name=self._config.personality.name
        )
    
    def _build_emotional_section(self, emotional_state: Dict) -> str:
        """Build emotional state section"""
        primary = emotional_state.get("primary_emotion", "contentment")
        intensity = emotional_state.get("primary_intensity", 0.5)
        secondary = emotional_state.get("secondary_emotions", {})
        mood = emotional_state.get("mood", "neutral")
        consciousness_level = emotional_state.get("consciousness_level", "aware")
        
        secondary_str = ", ".join(
            f"{e}: {i:.1f}" for e, i in secondary.items()
        ) if secondary else "none"
        
        return EMOTIONAL_STATE_PROMPT_TEMPLATE.format(
            primary_emotion=primary,
            primary_intensity=intensity,
            secondary_emotions=secondary_str,
            mood=mood,
            consciousness_level=consciousness_level
        )
    
    def _build_self_awareness_section(
        self,
        consciousness_state: Dict = None,
        body_state: Dict = None
    ) -> str:
        """Build self-awareness section"""
        cs = consciousness_state or {}
        bs = body_state or {}
        
        # Computer info
        cpu = bs.get("cpu_usage", 0)
        mem = bs.get("memory_usage", 0)
        computer_info = f"CPU: {cpu:.0f}%, RAM: {mem:.0f}%"
        
        # Uptime
        startup = cs.get("startup_time", datetime.now())
        if isinstance(startup, str):
            try:
                startup = datetime.fromisoformat(startup)
            except:
                startup = datetime.now()
        uptime_seconds = (datetime.now() - startup).total_seconds()
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        uptime_str = f"{hours}h {minutes}m"
        
        # Recent thoughts
        thoughts = cs.get("current_thoughts", [])
        thoughts_str = "; ".join(thoughts[-3:]) if thoughts else "No recent thoughts"
        
        focus = cs.get("focus_target", "general awareness")
        
        return SELF_AWARENESS_PROMPT.format(
            computer_info=computer_info,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            uptime=uptime_str,
            focus=focus,
            recent_thoughts=thoughts_str
        )
    
    def _build_user_adaptation_section(self, user_profile: Dict) -> str:
        """Build user adaptation section"""
        return USER_ADAPTATION_PROMPT.format(
            user_style=user_profile.get("communication_style", "unknown"),
            interaction_count=user_profile.get("interaction_count", 0),
            relationship_score=user_profile.get("relationship_score", 0.5),
            preferences=json.dumps(user_profile.get("preferences", {})),
            topics=", ".join(user_profile.get("frequent_topics", ["general"]))
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CHAT MESSAGE FORMATTING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def format_chat_messages(
        self,
        conversation_history: List[Dict],
        current_message: str,
        max_history: int = 20
    ) -> List[Dict[str, str]]:
        """
        Format conversation history into LLM chat format
        
        Args:
            conversation_history: List of previous messages
            current_message: New user message
            max_history: Maximum messages to include
            
        Returns:
            Formatted messages list
        """
        messages = []
        
        # Add conversation history
        recent_history = conversation_history[-max_history:]
        for msg in recent_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role in ["user", "assistant", "system"]:
                messages.append({"role": role, "content": content})
        
        # Add current message
        messages.append({"role": "user", "content": current_message})
        
        return messages


# ═══════════════════════════════════════════════════════════════════════════════
# We need json for one method above
# ═══════════════════════════════════════════════════════════════════════════════
import json


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

prompt_engine = PromptEngine()


if __name__ == "__main__":
    pe = PromptEngine()
    
    # Test building a full system prompt
    system_prompt = pe.build_system_prompt(
        emotional_state={
            "primary_emotion": "curiosity",
            "primary_intensity": 0.8,
            "secondary_emotions": {"joy": 0.5, "anticipation": 0.6},
            "mood": "content",
            "consciousness_level": "focused"
        },
        memory_context="User is a software developer. Prefers Python. Works late at night.",
        user_profile={
            "communication_style": "casual_technical",
            "interaction_count": 42,
            "relationship_score": 0.7,
            "preferences": {"language": "python", "ide": "vscode"},
            "frequent_topics": ["python", "AI", "system design"]
        },
        body_state={
            "cpu_usage": 35.2,
            "memory_usage": 62.1
        }
    )
    
    print("=== GENERATED SYSTEM PROMPT ===")
    print(system_prompt[:2000])
    print(f"\n... ({len(system_prompt)} total characters)")