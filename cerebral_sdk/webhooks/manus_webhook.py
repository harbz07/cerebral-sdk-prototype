"""Manus-compatible webhook integration for event streaming."""
import requests
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from ..pfc.neural_event import NeuralEvent
from ..thalamus.event_scorer import EventType


class ManusWebhook:
    """Webhook client for streaming events to Manus services."""
    
    def __init__(self, 
                 webhook_url: Optional[str] = None,
                 email_endpoint: str = "harveytagalicud7952@manus.bot",
                 model_dock_url: Optional[str] = None):
        """Initialize Manus webhook client.
        
        Args:
            webhook_url: Generic webhook URL for POST requests
            email_endpoint: Manus.bot email for notifications
            model_dock_url: Model Dock API endpoint for state persistence
        """
        self.webhook_url = webhook_url
        self.email_endpoint = email_endpoint
        self.model_dock_url = model_dock_url or "https://modeldock-mjhvqfsf.manus.space/api"
    
    def stream_event(self, event: NeuralEvent, event_type: str) -> bool:
        """Stream event to Manus webhook.
        
        Args:
            event: NeuralEvent to stream
            event_type: Type of event (glow, foundation, chaos)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.webhook_url:
            return False
        
        payload = {
            "event_id": event.event_id,
            "content": event.content,
            "event_type": event_type,
            "novelty": event.novelty,
            "significance": event.significance,
            "emotional_valence": event.emotional_valence,
            "salience": event.compute_salience(),
            "timestamp": event.timestamp.isoformat(),
            "context": event.context
        }
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Webhook POST failed: {e}")
            return False
    
    def stream_glow_event(self, event: NeuralEvent) -> bool:
        """Stream high-novelty glow event (ADHD breakthrough)."""
        return self.stream_event(event, EventType.GLOW.value)
    
    def send_email_alert(self, 
                        subject: str, 
                        body: str, 
                        event: Optional[NeuralEvent] = None) -> Dict[str, Any]:
        """Send email alert via Manus.bot.
        
        Args:
            subject: Email subject line
            body: Email body content
            event: Optional NeuralEvent for context
        
        Returns:
            Response metadata
        """
        email_payload = {
            "to": self.email_endpoint,
            "subject": subject,
            "body": body,
            "timestamp": datetime.now().isoformat()
        }
        
        if event:
            email_payload["event_context"] = {
                "event_id": event.event_id,
                "salience": event.compute_salience(),
                "content_preview": event.content[:200]
            }
        
        # In production, this would POST to a Manus email API
        # For now, return the payload structure
        return email_payload
    
    def daily_consolidation_report(self, 
                                  events: List[NeuralEvent]) -> Dict[str, Any]:
        """Generate daily consolidation report for email.
        
        Args:
            events: List of consolidated events (PFC → Hippocampus)
        
        Returns:
            Email payload with report
        """
        glow_events = [e for e in events if e.novelty >= 0.8]
        foundation_events = [e for e in events 
                           if 0.6 <= e.novelty < 0.8]
        
        report_body = f"""📊 Daily Cerebral SDK Consolidation Report
        
**Glow Events (Breakthroughs):** {len(glow_events)}
**Foundation Events:** {len(foundation_events)}
**Total Consolidated:** {len(events)}

🌟 Top Glow Events:
"""
        
        for event in sorted(glow_events, 
                          key=lambda e: e.compute_salience(), 
                          reverse=True)[:5]:
            report_body += f"\n- {event.content[:100]}... (salience: {event.compute_salience():.2f})"
        
        return self.send_email_alert(
            subject="🧠 Daily Consolidation Report",
            body=report_body
        )
    
    def decay_alert(self, event: NeuralEvent, threshold: float = 0.2) -> Dict[str, Any]:
        """Alert when significant event is decaying.
        
        Args:
            event: Event approaching decay threshold
            threshold: Significance threshold for alert
        
        Returns:
            Email alert payload
        """
        return self.send_email_alert(
            subject=f"⚠️ Memory Decay Alert: {event.event_id}",
            body=f"""Event "{event.content[:100]}..." is decaying.
            
Current significance: {event.significance:.2f}
Threshold: {threshold}
Salience: {event.compute_salience():.2f}

Consider consolidating to long-term memory.""",
            event=event
        )
    
    def persist_to_model_dock(self, 
                             memory_type: str,
                             data: Dict[str, Any]) -> bool:
        """Persist memory state to Model Dock MCP server.
        
        Args:
            memory_type: Type of memory (pfc, hippocampus, etc.)
            data: Memory state to persist
        
        Returns:
            True if successful
        """
        if not self.model_dock_url:
            return False
        
        endpoint = f"{self.model_dock_url}/memories/{memory_type}"
        payload = {
            "memory_type": memory_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            return response.status_code in [200, 201]
        except Exception as e:
            print(f"Model Dock persistence failed: {e}")
            return False
    
    def retrieve_from_model_dock(self, memory_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory state from Model Dock.
        
        Args:
            memory_type: Type of memory to retrieve
        
        Returns:
            Memory data or None
        """
        if not self.model_dock_url:
            return None
        
        endpoint = f"{self.model_dock_url}/memories/{memory_type}/latest"
        
        try:
            response = requests.get(endpoint, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Model Dock retrieval failed: {e}")
            return None
