"""CerebralEngine: Main orchestrator for neuromorphic LLM pipeline.

This engine coordinates the full cognitive flow:
1. Input -> Amygdala (Feel it: emotional valence & arousal)
2. Thalamus (Score it: novelty vs. history)
3. PFC (Decide: Ignore, Process, or Obsess?)
4. Corpus Callosum (Route: which LLM for this complexity?)
5. Hippocampus (Remember: consolidate if significant)
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from cerebral_sdk.amygdala.valence import ValenceAnalyzer
from cerebral_sdk.thalamus.event_scorer import VectorEventScorer
from cerebral_sdk.pfc.working_memory import WorkingMemory
from cerebral_sdk.hippocampus.memory_store import MemoryStore
from cerebral_sdk.corpus_callosum.router import ModelRouter


@dataclass
class CognitiveState:
    """Represents the current cognitive state of the system."""
    input_text: str
    emotional_valence: float
    emotional_arousal: float
    is_flashbulb: bool
    novelty: float
    event_type: str
    should_process: bool
    selected_model: Optional[str]
    response: Optional[str]
    consolidated: bool


class CerebralEngine:
    """Main orchestrator for the Cerebral SDK cognitive pipeline.
    
    Integrates all brain modules into a cohesive system that processes
    inputs through emotion, novelty, attention, routing, and memory.
    """
    
    def __init__(self, adhd_mode: bool = True):
        """Initialize the Cerebral Engine.
        
        Args:
            adhd_mode: Enable ADHD-optimized processing
        """
        # Initialize all modules
        self.amygdala = ValenceAnalyzer()
        self.hippocampus = MemoryStore()
        self.thalamus = VectorEventScorer(
            hippocampus=self.hippocampus,
            adhd_mode=adhd_mode
        )
        self.pfc = WorkingMemory()
        self.corpus_callosum = ModelRouter()
        
        # Engine configuration
        self.adhd_mode = adhd_mode
        self.consolidation_threshold = 0.7  # Novelty threshold for memory consolidation
        
    def process(self, input_text: str, metadata: Optional[Dict[str, Any]] = None) -> CognitiveState:
        """Process input through the full cognitive pipeline.
        
        Args:
            input_text: Text to process
            metadata: Optional metadata
            
        Returns:
            CognitiveState containing all processing results
        """
        # Step 1: AMYGDALA - Emotional analysis
        emotional_state = self.amygdala.analyze(input_text)
        
        # Step 2: THALAMUS - Novelty scoring
        scoring_result = self.thalamus.score_event(input_text, metadata)
        
        # Update emotional state with novelty for flashbulb detection
        if not emotional_state.is_flashbulb:
            emotional_state = self.amygdala.analyze(
                input_text, 
                novelty=scoring_result['novelty']
            )
        
        # Step 3: PFC - Attention gating
        should_process = scoring_result['should_process']
        
        if should_process:
            # Add to working memory
            self.pfc.add_event({
                'content': input_text,
                'novelty': scoring_result['novelty'],
                'valence': emotional_state.valence,
                'arousal': emotional_state.arousal
            })
            
            # Step 4: CORPUS CALLOSUM - Model routing
            # Determine complexity based on event type and arousal
            if scoring_result['event_type'] == 'glow' or emotional_state.arousal > 0.7:
                task_type = 'reasoning'  # High complexity
            elif emotional_state.arousal > 0.4:
                task_type = 'creative'  # Medium complexity
            else:
                task_type = 'fast'  # Low complexity
            
            selected_model = self.corpus_callosum.route(task_type)
            
            # Generate response (placeholder - would call actual LLM here)
            response = f"[{selected_model}] Processing: {input_text[:50]}..."
        else:
            selected_model = None
            response = None
        
        # Step 5: HIPPOCAMPUS - Memory consolidation
        consolidated = False
        if should_process and self._should_consolidate(scoring_result, emotional_state):
            self.hippocampus.store_memory(
                content=input_text,
                metadata={
                    'novelty': scoring_result['novelty'],
                    'valence': emotional_state.valence,
                    'arousal': emotional_state.arousal,
                    'is_flashbulb': emotional_state.is_flashbulb,
                    'event_type': scoring_result['event_type']
                }
            )
            consolidated = True
        
        # Return complete cognitive state
        return CognitiveState(
            input_text=input_text,
            emotional_valence=emotional_state.valence,
            emotional_arousal=emotional_state.arousal,
            is_flashbulb=emotional_state.is_flashbulb,
            novelty=scoring_result['novelty'],
            event_type=scoring_result['event_type'],
            should_process=should_process,
            selected_model=selected_model,
            response=response,
            consolidated=consolidated
        )
    
    def _should_consolidate(self, scoring_result: Dict, emotional_state) -> bool:
        """Determine if memory should be consolidated.
        
        Consolidation criteria:
        1. Flashbulb memory (high arousal + high novelty)
        2. High novelty above threshold
        3. Glow events (breakthroughs)
        """
        if emotional_state.is_flashbulb:
            return True
        
        if scoring_result['novelty'] >= self.consolidation_threshold:
            return True
        
        if scoring_result['event_type'] == 'glow':
            return True
        
        return False
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state across all modules."""
        return {
            'working_memory_count': len(self.pfc.events),
            'long_term_memory_count': len(self.hippocampus.memories),
            'adhd_mode': self.adhd_mode,
            'thalamus_stats': self.thalamus.get_statistics()
        }
    
    def clear_working_memory(self):
        """Clear working memory (PFC)."""
        self.pfc.clear()
    
    def export_memories(self) -> list:
        """Export all consolidated memories."""
        return self.hippocampus.memories
