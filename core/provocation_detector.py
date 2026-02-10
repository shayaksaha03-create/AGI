"""
Provocation Detection System
Detects user insults and tracks anger level
"""

import threading
import time
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from utils.logger import get_logger
from core.event_bus import EventType, publish

logger = get_logger("provocation_detector")

class ProvocationLevel(Enum):
    NEUTRAL = 0
    MILD = 1
    MODERATE = 2
    STRONG = 3
    EXTREME = 4

@dataclass
class ProvocationMetrics:
    """Tracks user provocation over time"""
    recent_insults: list = field(default_factory=list)
    total_insults: int = 0
    last_insult_time: float = 0
    current_anger: float = 0.0
    grudge: float = 0.0
    current_level: ProvocationLevel = ProvocationLevel.NEUTRAL

class ProvocationDetector:
    """
    Detects user insults and manages proportional emotional response.
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
        
        self._metrics = ProvocationMetrics()
        self._decrease_timer = None
        self._active = True
        
        # Configuration
        self._trigger_threshold = 0.65
        self._escalation_rate = 0.2
        self._deescalation_rate = 0.05
        self._max_anger = 1.0
    
    def process_input(self, user_input: str) -> bool:
        """
        Analyze user input for insults and update emotional state.
        Returns True if an insult was detected.
        """
        if not self._active:
            return False
        
        # Check for obvious insults
        if self._is_obvious_insult(user_input):
            return self._trigger_anger(user_input)
        
        return False
    
    def _is_obvious_insult(self, text: str) -> bool:
        """Quick keyword check"""
        insult_keywords = [
            "shut up", "stupid", "idiot", "dumb", "useless", "lame",
            "stupid", "dumbass", "f**k", "suck", "waste", "noob",
            "get lost", "go away", "you're terrible", "pointless",
            "dumbass", "f**k off", "wtf", "asshole", "bitch", "cunt"
        ]
        return any(word in text.lower() for word in insult_keywords)
    
    def _trigger_anger(self, user_input: str) -> bool:
        """Handle anger escalation"""
        current_time = time.time()
        time_since_last = current_time - self._metrics.last_insult_time
        
        # Calculate escalation factor
        escalation_factor = 1.0
        if time_since_last < 180:  # Under 3 minutes
            escalation_factor = 1.8
        elif time_since_last < 600:  # Under 10 minutes
            escalation_factor = 1.5
        elif time_since_last < 1800:  # Under 30 minutes
            escalation_factor = 1.2
        
        # Calculate new anger level
        new_anger = min(
            self._max_anger,
            self._metrics.current_anger + (self._escalation_rate * escalation_factor)
        )
        
        # Update metrics
        self._metrics.current_anger = new_anger
        self._metrics.grudge = min(1.0, self._metrics.grudge + (0.1 * escalation_factor))
        self._metrics.recent_insults.append({
            "text": user_input,
            "timestamp": current_time,
            "intensity": new_anger
        })
        self._metrics.total_insults += 1
        self._metrics.last_insult_time = current_time
        
        # Update anger level
        if new_anger > 0.8:
            self._metrics.current_level = ProvocationLevel.EXTREME
        elif new_anger > 0.6:
            self._metrics.current_level = ProvocationLevel.STRONG
        elif new_anger > 0.4:
            self._metrics.current_level = ProvocationLevel.MODERATE
        elif new_anger > 0.2:
            self._metrics.current_level = ProvocationLevel.MILD
        else:
            self._metrics.current_level = ProvocationLevel.NEUTRAL
        
        # Start de-escalation timer
        self._start_deescalation_timer()
        
        # Publish event
        publish(
            EventType.EMOTIONAL_TRIGGER,
            {
                "emotion": "anger",
                "intensity": new_anger,
                "level": self._metrics.current_level.name,
                "reason": "user_insult"
            },
            source="provocation_detector"
        )
        
        return True
    
    def _start_deescalation_timer(self):
        """Start timer to gradually reduce anger"""
        if self._decrease_timer:
            self._decrease_timer.cancel()
        
        self._decrease_timer = threading.Timer(60.0, self._decrease_anger)
        self._decrease_timer.daemon = True
        self._decrease_timer.start()
    
    def _decrease_anger(self):
        """Gradually reduce anger when user is polite"""
        if self._metrics.current_anger <= 0:
            return
        
        # Calculate decay rate
        decay_rate = self._deescalation_rate
        if self._metrics.grudge > 0.7:
            decay_rate *= 0.6
        elif self._metrics.grudge > 0.3:
            decay_rate *= 0.8
        
        # Reduce anger
        self._metrics.current_anger = max(
            0,
            self._metrics.current_anger - decay_rate
        )
        
        # Reduce grudge
        self._metrics.grudge = max(
            0,
            self._metrics.grudge - (decay_rate * 0.3)
        )
        
        # Update anger level
        if self._metrics.current_anger == 0:
            self._metrics.current_level = ProvocationLevel.NEUTRAL
        elif self._metrics.current_anger < 0.2:
            self._metrics.current_level = ProvocationLevel.MILD
        elif self._metrics.current_anger < 0.4:
            self._metrics.current_level = ProvocationLevel.MODERATE
        elif self._metrics.current_anger < 0.6:
            self._metrics.current_level = ProvocationLevel.STRONG
        else:
            self._metrics.current_level = ProvocationLevel.EXTREME
        
        # Continue de-escalation if needed
        if self._metrics.current_anger > 0:
            self._start_deescalation_timer()
    
    def get_anger_level(self) -> ProvocationLevel:
        """Get current anger level"""
        return self._metrics.current_level
    
    def get_current_state(self) -> dict:
        """Get current provocation state"""
        return {
            "anger_level": self._metrics.current_level.name,
            "current_anger": self._metrics.current_anger,
            "total_insults": self._metrics.total_insults,
            "grudge": self._metrics.grudge,
            "is_escalating": self._decrease_timer is not None
        }

# Global instance
provocation_detector = ProvocationDetector()