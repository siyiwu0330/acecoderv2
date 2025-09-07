"""
OpenAI Utilities for AceCoderV2

This module provides backward-compatible OpenAI client utilities
while integrating with the new centralized openai_client.py
"""

import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os
import logging
from typing import Optional, List, Dict, Any

# Setup logging
logger = logging.getLogger(__name__)

# Import the new centralized client
try:
    from openai_client import get_openai_client, OpenAIClientManager
    CENTRALIZED_CLIENT_AVAILABLE = True
except ImportError:
    CENTRALIZED_CLIENT_AVAILABLE = False
    logger.warning("Centralized OpenAI client not available, using legacy implementation")

class OpenAISyncClient:
    """
    Legacy OpenAI client for backward compatibility
    Now uses the centralized client manager internally
    """
    def __init__(self, api_key=None, base_url="https://api.openai.com/v1"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        
        if CENTRALIZED_CLIENT_AVAILABLE:
            # Use centralized client manager
            self.client_manager = get_openai_client(api_key=self.api_key, base_url=self.base_url)
            self.client = self.client_manager.sync_client
        else:
            # Fallback to direct client creation
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt, model="gpt-3.5-turbo", **kwargs):
        """Generate text using the OpenAI API with retry logic"""
        messages = [{"role": "user", "content": prompt}]
        return generate_with_retry_sync(self.client, messages, model=model, **kwargs)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_with_retry_sync(openai_client, messages, model="gpt-3.5-turbo", **kwargs):
    """
    Generate text with automatic retry logic
    
    Args:
        openai_client: OpenAI client instance
        messages: List of message dictionaries
        model: Model name to use
        **kwargs: Additional parameters for the API call
        
    Returns:
        List of choice dictionaries
    """
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return [choice.model_dump() for choice in response.choices]
    except Exception as e:
        logger.error(f"Error in generate_with_retry_sync: {e}")
        raise

# Convenience functions for easy migration
def get_legacy_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAISyncClient:
    """Get a legacy OpenAISyncClient instance"""
    return OpenAISyncClient(api_key=api_key, base_url=base_url)

def get_modern_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAIClientManager:
    """Get the modern centralized client manager"""
    if not CENTRALIZED_CLIENT_AVAILABLE:
        raise ImportError("Centralized client not available. Please ensure openai_client.py is present.")
    return get_openai_client(api_key=api_key, base_url=base_url) 
