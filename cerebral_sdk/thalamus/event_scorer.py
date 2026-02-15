"""Thalamus: Event scoring and filtering with Chaos/Foundation/Glow heuristics."""
from typing import Dict, Optional
from enum import Enum
import re


class EventType(Enum):
    """Event classification based on Thalamic scoring."""
    CHAOS = "chaos"  # Low novelty, routine
    FOUNDATION = "foundation"  # Medium, structural
    GLOW = "glow"  # High novelty, breakthrough


class ThalamusScorer:
    """Thalamus-inspired event scoring and sensory gating."""
    
    def __init__(self, 
                 chaos_threshold: float = 0.3,
                 foundation_threshold: float = 0.6,
                 glow_threshold: float = 0.8):
        """Initialize with scoring thresholds."""
        self.chaos_threshold = chaos_threshold
        self.foundation_threshold = foundation_threshold
        self.glow_threshold = glow_threshold
        
        # Keyword patterns for heuristic scoring
        self.glow_keywords = [
            r'breakthrough', r'discovery', r'insight', r'revelation',
            r'eureka', r'novel', r'unprecedented', r'paradigm'
        ]
        self.foundation_keywords = [
            r'important', r'significant', r'key', r'essential',
            r'critical', r'core', r'fundamental'
        ]
    
    def score_novelty(self, content: str, context: Dict = None) -> float:
        """Score novelty using keyword heuristics and context."""
        content_lower = content.lower()
        score = 0.0
        
        # Check for glow keywords (high novelty)
        for pattern in self.glow_keywords:
            if re.search(pattern, content_lower):
                score += 0.3
        
        # Check for foundation keywords (medium novelty)
        for pattern in self.foundation_keywords:
            if re.search(pattern, content_lower):
                score += 0.15
        
        # Length heuristic (longer might be more novel)
        word_count = len(content.split())
        if word_count > 100:
            score += 0.1
        elif word_count < 10:
            score -= 0.1
        
        # Context-based adjustment
        if context:
            if context.get('user_initiated'):
                score += 0.2
            if context.get('emotional_trigger'):
                score += 0.15
        
        return min(1.0, max(0.0, score))
    
    def classify_event(self, novelty_score: float) -> EventType:
        """Classify event based on novelty score."""
        if novelty_score >= self.glow_threshold:
            return EventType.GLOW
        elif novelty_score >= self.foundation_threshold:
            return EventType.FOUNDATION
        else:
            return EventType.CHAOS
    
    def should_pass_gate(self, 
                        novelty_score: float, 
                        significance: float,
                        min_threshold: float = 0.5) -> bool:
        """Sensory gating: decide if event should pass to PFC."""
        # Combine novelty and significance for gating decision
        combined_score = (novelty_score * 0.6 + significance * 0.4)
        return combined_score >= min_threshold
    
    def score_event(self, 
                   content: str, 
                   context: Optional[Dict] = None) -> Dict[str, any]:
        """Full event scoring pipeline."""
        novelty = self.score_novelty(content, context)
        event_type = self.classify_event(novelty)
        
        # ADHD-specific: spike on glow events, suppress chaos
        if event_type == EventType.GLOW:
            adjusted_novelty = min(1.0, novelty * 1.2)
        elif event_type == EventType.CHAOS:
            adjusted_novelty = novelty * 0.8
        else:
            adjusted_novelty = novelty
        
        return {
            'novelty': adjusted_novelty,
            'event_type': event_type.value,
            'should_process': self.should_pass_gate(adjusted_novelty, 0.5)
        }
