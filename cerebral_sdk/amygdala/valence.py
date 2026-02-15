"""Amygdala Module: Emotional Valence & Arousal Detection.

This module provides emotional significance scoring for events,
replacing the hardcoded 0.0 valence with actual sentiment analysis.
Implements arousal detection for flashbulb memory formation.
"""
import re
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EmotionalState:
    """Represents the emotional dimensions of an event."""
    valence: float  # -1.0 (negative) to +1.0 (positive)
    arousal: float  # 0.0 (calm) to 1.0 (intense)
    is_flashbulb: bool  # High arousal + high novelty = flashbulb memory


class ValenceAnalyzer:
    """Emotional valence and arousal analyzer using lexicon-based approach.
    
    For production, consider upgrading to:
    - VADER (vaderSentiment) for better accuracy
    - distilbert-base-uncased-finetuned-sst-2-english for transformer-based
    
    Current implementation: Fast lexicon-based approach for prototyping.
    """
    
    def __init__(self):
        # Positive emotion keywords and their intensities
        self.positive_words = {
            # High intensity (0.8-1.0)
            r'\b(ecstatic|overjoyed|thrilled|elated|euphoric|exhilarated)\b': 0.9,
            r'\b(breakthrough|discovery|amazing|incredible|fantastic|wonderful)\b': 0.85,
            r'\b(love|adore|perfect|brilliant|excellent|outstanding)\b': 0.8,
            
            # Medium intensity (0.5-0.7)
            r'\b(happy|joy|pleased|glad|delighted|excited)\b': 0.7,
            r'\b(good|great|nice|helpful|positive|successful)\b': 0.6,
            r'\b(like|enjoy|appreciate|satisfied|content)\b': 0.5,
        }
        
        # Negative emotion keywords and their intensities
        self.negative_words = {
            # High intensity (-0.8 to -1.0)
            r'\b(devastated|horrible|terrible|catastrophic|disaster|nightmare)\b': -0.9,
            r'\b(hate|despise|furious|enraged|disgusted|appalled)\b': -0.85,
            r'\b(awful|dreadful|miserable|tragic|horrific)\b': -0.8,
            
            # Medium intensity (-0.5 to -0.7)
            r'\b(angry|sad|upset|disappointed|frustrated|annoyed)\b': -0.7,
            r'\b(bad|poor|wrong|problem|issue|error)\b': -0.6,
            r'\b(dislike|concerned|worried|anxious|stressed)\b': -0.5,
        }
        
        # Arousal indicators (intensity markers)
        self.arousal_markers = {
            # High arousal (0.8-1.0)
            r'\b(!{2,}|\?{2,})\b': 0.9,  # Multiple punctuation
            r'\b(URGENT|EMERGENCY|CRITICAL|IMPORTANT)\b': 0.9,
            r'\b(shocking|stunning|explosive|overwhelming|intense)\b': 0.85,
            
            # Medium arousal (0.5-0.7)
            r'\b(exciting|surprising|unexpected|dramatic|significant)\b': 0.7,
            r'\b(quickly|suddenly|immediately|urgent|now)\b': 0.6,
            r'[!]': 0.5,  # Single exclamation
        }
        
        # Modifiers that intensify or reduce emotion
        self.intensifiers = [
            r'\b(very|extremely|incredibly|absolutely|totally|completely)\b',
            r'\b(really|quite|pretty|fairly|rather|somewhat)\b',
        ]
        
        self.negations = [
            r'\b(not|no|never|neither|nobody|nothing|nowhere)\b',
            r"\b(don't|doesn't|didn't|won't|wouldn't|can't|couldn't)\b",
        ]

    def analyze(self, text: str, novelty: Optional[float] = None) -> EmotionalState:
        """Analyze emotional valence and arousal of text.
        
        Args:
            text: Input text to analyze
            novelty: Optional novelty score (0.0-1.0) from Thalamus
            
        Returns:
            EmotionalState with valence, arousal, and flashbulb memory flag
        """
        text_lower = text.lower()
        
        # Compute valence
        valence = self._compute_valence(text_lower)
        
        # Compute arousal
        arousal = self._compute_arousal(text, text_lower)
        
        # Determine if this is a flashbulb memory
        # High arousal + high novelty = instant consolidation
        is_flashbulb = False
        if novelty is not None:
            is_flashbulb = arousal > 0.7 and novelty > 0.7
        
        return EmotionalState(
            valence=valence,
            arousal=arousal,
            is_flashbulb=is_flashbulb
        )

    def _compute_valence(self, text: str) -> float:
        """Compute emotional valence (-1.0 to +1.0)."""
        positive_score = 0.0
        negative_score = 0.0
        positive_count = 0
        negative_count = 0
        
        # Check for negations
        has_negation = any(re.search(pattern, text) for pattern in self.negations)
        
        # Score positive words
        for pattern, score in self.positive_words.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                count = len(matches)
                positive_score += score * count
                positive_count += count
        
        # Score negative words
        for pattern, score in self.negative_words.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                count = len(matches)
                negative_score += abs(score) * count
                negative_count += count
        
        # Compute net valence
        if positive_count + negative_count == 0:
            return 0.0
        
        net_score = positive_score - negative_score
        total_count = positive_count + negative_count
        valence = net_score / total_count
        
        # Flip valence if negation detected
        if has_negation:
            valence *= -0.7  # Partial flip
        
        # Clamp to [-1.0, 1.0]
        return max(-1.0, min(1.0, valence))

    def _compute_arousal(self, text_original: str, text_lower: str) -> float:
        """Compute emotional arousal/intensity (0.0 to 1.0)."""
        arousal_score = 0.0
        arousal_count = 0
        
        # Check arousal markers
        for pattern, score in self.arousal_markers.items():
            # Use original text to detect case (CAPS)
            search_text = text_original if pattern.isupper() else text_lower
            matches = re.findall(pattern, search_text)
            if matches:
                count = len(matches)
                arousal_score += score * count
                arousal_count += count
        
        # Check for ALL CAPS words (high arousal)
        caps_words = re.findall(r'\b[A-Z]{3,}\b', text_original)
        if caps_words:
            arousal_score += 0.8 * len(caps_words)
            arousal_count += len(caps_words)
        
        # Check for intensifiers
        for pattern in self.intensifiers:
            if re.search(pattern, text_lower):
                arousal_score += 0.3
                arousal_count += 1
        
        # Baseline arousal from text length
        # Longer text can indicate more involved/arousing content
        word_count = len(text_lower.split())
        if word_count > 50:
            arousal_score += 0.2
            arousal_count += 1
        
        if arousal_count == 0:
            return 0.3  # Baseline arousal
        
        arousal = arousal_score / arousal_count
        
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, arousal))

    def get_emotion_label(self, valence: float, arousal: float) -> str:
        """Map valence/arousal to emotion labels (Russell's Circumplex Model)."""
        if arousal > 0.6:
            if valence > 0.3:
                return "excited" if arousal > 0.8 else "happy"
            elif valence < -0.3:
                return "angry" if arousal > 0.8 else "frustrated"
            else:
                return "surprised"
        else:
            if valence > 0.3:
                return "content"
            elif valence < -0.3:
                return "sad"
            else:
                return "neutral"
