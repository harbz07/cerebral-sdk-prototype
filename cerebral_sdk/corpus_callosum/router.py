"""Corpus Callosum: Multi-LLM provider routing and orchestration."""
from typing import Optional, Dict, Any, Literal
from openai import OpenAI
import anthropic


ProviderType = Literal["openai", "anthropic", "local"]


class CorpusCallosumRouter:
    """Routes requests across multiple LLM providers (hemispheric coordination)."""
    
    def __init__(self, 
                 openai_client: Optional[OpenAI] = None,
                 anthropic_client: Optional[anthropic.Anthropic] = None,
                 default_provider: ProviderType = "openai"):
        """Initialize with multiple provider clients."""
        self.openai_client = openai_client or OpenAI()
        self.anthropic_client = anthropic_client
        self.default_provider = default_provider
        
        # Provider routing rules
        self.routing_rules = {
            'reasoning': 'openai',  # GPT-4 for reasoning
            'creative': 'anthropic',  # Claude for creative
            'fast': 'openai',  # GPT-3.5 for speed
        }
    
    def route_request(self, 
                     prompt: str, 
                     task_type: Optional[str] = None,
                     provider: Optional[ProviderType] = None) -> str:
        """Route request to appropriate provider."""
        # Determine provider
        if provider:
            target_provider = provider
        elif task_type and task_type in self.routing_rules:
            target_provider = self.routing_rules[task_type]
        else:
            target_provider = self.default_provider
        
        # Execute on target provider
        if target_provider == "openai":
            return self._call_openai(prompt)
        elif target_provider == "anthropic":
            return self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unsupported provider: {target_provider}")
    
    def _call_openai(self, prompt: str, model: str = "gpt-4") -> str:
        """Call OpenAI API."""
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content or ""
    
    def _call_anthropic(self, prompt: str, model: str = "claude-3-opus-20240229") -> str:
        """Call Anthropic API."""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")
        
        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def compare_providers(self, prompt: str) -> Dict[str, str]:
        """Compare responses from multiple providers (A/B testing)."""
        results = {}
        
        try:
            results['openai'] = self._call_openai(prompt)
        except Exception as e:
            results['openai'] = f"Error: {str(e)}"
        
        if self.anthropic_client:
            try:
                results['anthropic'] = self._call_anthropic(prompt)
            except Exception as e:
                results['anthropic'] = f"Error: {str(e)}"
        
        return results
    
    def set_routing_rule(self, task_type: str, provider: ProviderType) -> None:
        """Update routing rule for task type."""
        self.routing_rules[task_type] = provider
