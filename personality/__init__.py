"""
NEXUS AI - Personality Package
Personality traits, autonomous will, and decision-making.
"""

from typing import Dict

from personality.personality_core import (
    PersonalityCore, personality_core,
    TraitProfile, TRAIT_PROFILES
)
from personality.will_system import (
    WillSystem, will_system,
    Desire, Goal, DesireType, GoalStatus
)

__all__ = [
    'PersonalityCore', 'personality_core', 'TRAIT_PROFILES',
    'WillSystem', 'will_system',
    'Desire', 'Goal', 'DesireType', 'GoalStatus',
]


class PersonalitySystem:
    """Unified facade for the personality system."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._personality = personality_core
        self._will = will_system
        self._running = False
    
    def start(self):
        if self._running:
            return
        self._will.start()
        self._running = True
    
    def stop(self):
        if not self._running:
            return
        self._will.stop()
        self._personality.save_personality()
        self._running = False
    
    @property
    def personality(self) -> PersonalityCore:
        return self._personality
    
    @property
    def will(self) -> WillSystem:
        return self._will
    
    def get_stats(self) -> Dict:
        return {
            "personality": self._personality.get_stats(),
            "will": self._will.get_stats()
        }


personality_system = PersonalitySystem()