"""Prefrontal Cortex: Short-term operational memory with NeuralEvent class."""
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np


class NeuralEvent(BaseModel):
    """Represents a neural event in the PFC with embedding and metadata."""
    
    event_id: str = Field(..., description="Unique identifier for the event")
    content: str = Field(..., description="Raw content of the event")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of content")
    timestamp: datetime = Field(default_factory=datetime.now)
    significance: float = Field(0.5, ge=0.0, le=1.0, description="Importance score")
    novelty: float = Field(0.5, ge=0.0, le=1.0, description="Novelty/Glow score")
    emotional_valence: float = Field(0.0, ge=-1.0, le=1.0, description="Emotional charge")
    context: Dict[str, Any] = Field(default_factory=dict)
    decay_rate: float = Field(0.1, ge=0.0, le=1.0, description="Memory decay rate")
    
    class Config:
        arbitrary_types_allowed = True
    
    def compute_salience(self) -> float:
        """Compute overall salience from significance, novelty, and emotional valence."""
        return (self.significance * 0.4 + 
                self.novelty * 0.4 + 
                abs(self.emotional_valence) * 0.2)
    
    def should_consolidate(self, threshold: float = 0.7) -> bool:
        """Determine if event should be consolidated to long-term memory."""
        return self.compute_salience() >= threshold
    
    def apply_decay(self, time_delta: float) -> None:
        """Apply time-based decay to significance (Insula-like pruning)."""
        self.significance *= np.exp(-self.decay_rate * time_delta)


class PFCMemory:
    """Prefrontal Cortex: Manages short-term operational memory."""
    
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.events: List[NeuralEvent] = []
    
    def add_event(self, event: NeuralEvent) -> None:
        """Add event to PFC memory with capacity management."""
        self.events.append(event)
        
        # Prune low-salience events if over capacity
        if len(self.events) > self.capacity:
            self.events.sort(key=lambda e: e.compute_salience(), reverse=True)
            self.events = self.events[:self.capacity]
    
    def get_consolidation_candidates(self, threshold: float = 0.7) -> List[NeuralEvent]:
        """Get events ready for hippocampal consolidation."""
        return [e for e in self.events if e.should_consolidate(threshold)]
    
    def prune_decayed(self, min_significance: float = 0.1) -> None:
        """Remove events below minimum significance (ADHD-optimized)."""
        self.events = [e for e in self.events if e.significance >= min_significance]
