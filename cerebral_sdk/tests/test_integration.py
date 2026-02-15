"""Integration tests for Cerebral SDK modules."""
import pytest
from datetime import datetime
from cerebral_sdk.pfc.neural_event import NeuralEvent, PFCMemory
from cerebral_sdk.thalamus.event_scorer import ThalamusScorer, EventType


class TestPFCModule:
    """Test Prefrontal Cortex module."""
    
    def test_neural_event_creation(self):
        """Test NeuralEvent instantiation."""
        event = NeuralEvent(
            event_id="test_001",
            content="This is a breakthrough discovery!",
            novelty=0.9,
            significance=0.8
        )
        assert event.event_id == "test_001"
        assert event.novelty == 0.9
        assert event.compute_salience() > 0.7
    
    def test_event_consolidation_threshold(self):
        """Test event consolidation decision."""
        high_salience = NeuralEvent(
            event_id="test_002",
            content="Critical insight",
            novelty=0.9,
            significance=0.9
        )
        assert high_salience.should_consolidate(threshold=0.7)
        
        low_salience = NeuralEvent(
            event_id="test_003",
            content="Routine update",
            novelty=0.2,
            significance=0.3
        )
        assert not low_salience.should_consolidate(threshold=0.7)
    
    def test_pfc_capacity_management(self):
        """Test PFC memory capacity pruning."""
        pfc = PFCMemory(capacity=5)
        
        # Add 10 events
        for i in range(10):
            event = NeuralEvent(
                event_id=f"event_{i}",
                content=f"Content {i}",
                novelty=0.5 + (i * 0.05),
                significance=0.5
            )
            pfc.add_event(event)
        
        # Should keep only top 5
        assert len(pfc.events) == 5
        # Should keep highest salience events
        assert all(e.compute_salience() >= 0.5 for e in pfc.events)


class TestThalamusModule:
    """Test Thalamus event scoring."""
    
    def test_novelty_scoring_glow_keywords(self):
        """Test novelty scoring with glow keywords."""
        scorer = ThalamusScorer()
        
        glow_content = "This is a breakthrough discovery with unprecedented insights!"
        score = scorer.score_novelty(glow_content)
        assert score > 0.5
    
    def test_event_classification(self):
        """Test event type classification."""
        scorer = ThalamusScorer(
            chaos_threshold=0.3,
            foundation_threshold=0.6,
            glow_threshold=0.8
        )
        
        assert scorer.classify_event(0.9) == EventType.GLOW
        assert scorer.classify_event(0.65) == EventType.FOUNDATION
        assert scorer.classify_event(0.2) == EventType.CHAOS
    
    def test_sensory_gating(self):
        """Test sensory gating threshold."""
        scorer = ThalamusScorer()
        
        # High novelty + high significance should pass
        assert scorer.should_pass_gate(novelty_score=0.8, significance=0.8)
        
        # Low novelty + low significance should not pass
        assert not scorer.should_pass_gate(novelty_score=0.2, significance=0.2)
    
    def test_adhd_spike_on_glow(self):
        """Test ADHD-optimized glow event spiking."""
        scorer = ThalamusScorer()
        
        result = scorer.score_event(
            "Breakthrough paradigm shift with revolutionary insights!",
            context={'user_initiated': True}
        )
        
        assert result['event_type'] == 'glow'
        assert result['novelty'] > 0.8  # Should be spiked
        assert result['should_process']


class TestIntegration:
    """Test integrated workflow."""
    
    def test_thalamus_to_pfc_pipeline(self):
        """Test event flow from Thalamus to PFC."""
        scorer = ThalamusScorer()
        pfc = PFCMemory(capacity=10)
        
        test_inputs = [
            ("Routine status update", {'user_initiated': False}),
            ("Critical breakthrough discovery!", {'user_initiated': True}),
            ("Important milestone achieved", {'user_initiated': True}),
        ]
        
        for content, context in test_inputs:
            score_result = scorer.score_event(content, context)
            
            if score_result['should_process']:
                event = NeuralEvent(
                    event_id=f"event_{len(pfc.events)}",
                    content=content,
                    novelty=score_result['novelty'],
                    significance=0.7
                )
                pfc.add_event(event)
        
        # Should have added at least the glow and foundation events
        assert len(pfc.events) >= 2
        # Glow event should have highest salience
        assert pfc.events[0].novelty > 0.7
