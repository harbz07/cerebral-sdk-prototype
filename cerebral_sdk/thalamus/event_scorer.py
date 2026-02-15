"""Thalamus Event Scorer with Vector-Based Novelty Detection.

Replaces regex keyword matching with semantic similarity scoring
using Hippocampus memory for truly subjective novelty detection.
"""
import re
from enum import Enum
from typing import Optional, Dict, Any
from cerebral_sdk.hippocampus.memory_store import MemoryStore


class EventType(Enum):
    """Event classification based on novelty and significance."""
    GLOW = "glow"  # High novelty: breakthrough moments
    FOUNDATION = "foundation"  # Medium novelty: building blocks
    CHAOS = "chaos"  # Low novelty: routine noise


class VectorEventScorer:
    """Vector-based event scorer using semantic similarity.
    
    Instead of checking for keywords like 'breakthrough' and 'eureka',
    this scorer queries the Hippocampus to find the nearest semantic
    neighbor and computes novelty as:
    
        Novelty = 1.0 - max(cosine_similarity)
    
    This makes the system purely subjective:
    - Seen this exact thought before (similarity 0.99) -> Novelty ~0.0 (CHAOS)
    - Never seen anything remotely like this (similarity 0.1) -> Novelty ~0.9 (GLOW)
    """
    
    def __init__(self, hippocampus: Optional[MemoryStore] = None,
                 adhd_mode: bool = True):
        """Initialize the vector-based event scorer.
        
        Args:
            hippocampus: MemoryStore instance for similarity queries
            adhd_mode: Enable ADHD-optimized scoring (spike glow, suppress chaos)
        """
        self.hippocampus = hippocampus or MemoryStore()
        self.adhd_mode = adhd_mode
        
        # Thresholds for event classification
        self.glow_threshold = 0.8  # High novelty
        self.chaos_threshold = 0.3  # Low novelty
        
        # ADHD multipliers
        self.glow_spike_multiplier = 1.2  # Amplify breakthrough moments
        self.chaos_suppression_multiplier = 0.8  # Dampen routine updates

    def score_event(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Score an event using vector similarity to determine novelty.
        
        Args:
            text: Event content to score
            metadata: Optional metadata for the event
            
        Returns:
            Dictionary containing:
                - novelty: float (0.0-1.0)
                - event_type: EventType (GLOW/FOUNDATION/CHAOS)
                - should_process: bool (pass gate check)
                - nearest_neighbor: Optional similarity info
        """
        # Query Hippocampus for nearest semantic neighbor
        nearest = self.hippocampus.query_similar(text, k=1)
        
        if nearest and len(nearest) > 0:
            # Compute novelty from semantic distance
            similarity = nearest[0].get('similarity', 0.0)
            novelty = 1.0 - similarity
        else:
            # No existing memories - this is completely novel
            novelty = 1.0
        
        # Classify event type
        event_type = self._classify_event(novelty)
        
        # ADHD-specific adjustments
        adjusted_novelty = self._apply_adhd_adjustments(novelty, event_type)
        
        # Gate check: Should this event be processed?
        should_process = self.should_pass_gate(adjusted_novelty, 0.5)
        
        return {
            'novelty': adjusted_novelty,
            'raw_novelty': novelty,
            'event_type': event_type.value,
            'should_process': should_process,
            'nearest_neighbor': nearest[0] if nearest else None,
            'similarity_to_past': 1.0 - novelty if nearest else 0.0
        }

    def _classify_event(self, novelty: float) -> EventType:
        """Classify event based on novelty score."""
        if novelty >= self.glow_threshold:
            return EventType.GLOW
        elif novelty <= self.chaos_threshold:
            return EventType.CHAOS
        else:
            return EventType.FOUNDATION

    def _apply_adhd_adjustments(self, novelty: float, event_type: EventType) -> float:
        """Apply ADHD-specific adjustments to novelty score."""
        if not self.adhd_mode:
            return novelty
        
        adjusted_novelty = novelty
        
        # Spike on GLOW events (amplify breakthrough moments)
        if event_type == EventType.GLOW:
            adjusted_novelty = min(1.0, novelty * self.glow_spike_multiplier)
        
        # Suppress CHAOS events (filter routine updates)
        elif event_type == EventType.CHAOS:
            adjusted_novelty = novelty * self.chaos_suppression_multiplier
        
        return adjusted_novelty

    def should_pass_gate(self, novelty: float, threshold: float = 0.5) -> bool:
        """Determine if event should be processed based on novelty gate."""
        return novelty >= threshold

    def get_statistics(self) -> Dict[str, Any]:
        """Get scorer statistics."""
        return {
            'memory_count': len(self.hippocampus.memories),
            'adhd_mode': self.adhd_mode,
            'glow_threshold': self.glow_threshold,
            'chaos_threshold': self.chaos_threshold
        }
