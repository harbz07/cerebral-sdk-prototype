"""Hippocampus: Semantic memory with embeddings and consolidation."""
from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from ..pfc.neural_event import NeuralEvent


class SemanticMemory:
    """Hippocampus-inspired semantic memory with vector embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.memory_store: List[NeuralEvent] = []
        self.embedding_dim = 384  # MiniLM embedding dimension
    
    def embed_event(self, event: NeuralEvent) -> NeuralEvent:
        """Generate embedding for event content."""
        embedding = self.model.encode(event.content)
        event.embedding = embedding.tolist()
        return event
    
    def consolidate(self, event: NeuralEvent, similarity_threshold: float = 0.7) -> None:
        """Consolidate event into long-term semantic memory."""
        # Ensure event has embedding
        if event.embedding is None:
            event = self.embed_event(event)
        
        # Check for similar existing memories (pattern completion)
        similar = self.find_similar(event, threshold=similarity_threshold)
        
        if similar:
            # Strengthen existing memory
            similar[0].significance = min(1.0, similar[0].significance * 1.2)
        else:
            # Store new memory
            self.memory_store.append(event)
    
    def find_similar(self, query_event: NeuralEvent, 
                    threshold: float = 0.7, 
                    top_k: int = 5) -> List[NeuralEvent]:
        """Find similar events using cosine similarity."""
        if not self.memory_store or query_event.embedding is None:
            return []
        
        query_vec = np.array(query_event.embedding)
        similarities = []
        
        for stored_event in self.memory_store:
            if stored_event.embedding:
                stored_vec = np.array(stored_event.embedding)
                sim = np.dot(query_vec, stored_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(stored_vec)
                )
                if sim >= threshold:
                    similarities.append((sim, stored_event))
        
        # Sort by similarity and return top_k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [event for _, event in similarities[:top_k]]
    
    def retrieve_by_content(self, query: str, top_k: int = 5) -> List[NeuralEvent]:
        """Retrieve events similar to query string."""
        query_embedding = self.model.encode(query)
        query_event = NeuralEvent(
            event_id="query",
            content=query,
            embedding=query_embedding.tolist()
        )
        return self.find_similar(query_event, threshold=0.5, top_k=top_k)
    
    def prune_low_significance(self, min_significance: float = 0.2) -> int:
        """Remove low-significance memories (ADHD cognitive load management)."""
        initial_count = len(self.memory_store)
        self.memory_store = [
            e for e in self.memory_store 
            if e.significance >= min_significance
        ]
        return initial_count - len(self.memory_store)
