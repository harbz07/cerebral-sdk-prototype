"""Wernicke's Area: Context compression and summarization for token efficiency."""
from typing import List, Optional
from openai import OpenAI


class WernickeSummarizer:
    """Wernicke-inspired summarization for context compression."""
    
    def __init__(self, client: Optional[OpenAI] = None, max_tokens: int = 16384):
        """Initialize with OpenAI client and token budget."""
        self.client = client or OpenAI()
        self.max_tokens = max_tokens
    
    def summarize_context(self, 
                         contexts: List[str], 
                         compression_ratio: float = 0.3) -> str:
        """Compress multiple context snippets into concise summary."""
        if not contexts:
            return ""
        
        combined = "\n\n".join(contexts)
        target_tokens = int(len(combined.split()) * compression_ratio)
        
        prompt = f"""Summarize the following context concisely, preserving key information.
        Target length: ~{target_tokens} words.
        
        Context:
        {combined}
        
        Summary:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=min(target_tokens * 2, 500),
            temperature=0.3
        )
        
        return response.choices[0].message.content or ""
    
    def fit_context(self, 
                   contexts: List[str], 
                   max_context_tokens: int = 12000) -> str:
        """Fit contexts within token budget, summarizing if needed."""
        combined = "\n\n".join(contexts)
        approx_tokens = len(combined.split()) * 1.3  # rough token estimate
        
        if approx_tokens <= max_context_tokens:
            return combined
        
        # Need compression
        compression_ratio = max_context_tokens / approx_tokens
        return self.summarize_context(contexts, compression_ratio)
    
    def extract_key_points(self, text: str, max_points: int = 5) -> List[str]:
        """Extract key bullet points from text (ADHD-friendly)."""
        prompt = f"""Extract {max_points} key points from this text as bullet points:
        
        {text}
        
        Key points:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2
        )
        
        content = response.choices[0].message.content or ""
        # Parse bullet points
        points = [line.strip().lstrip('-•*').strip() 
                 for line in content.split('\n') 
                 if line.strip().startswith(('-', '•', '*'))]
        return points[:max_points]
