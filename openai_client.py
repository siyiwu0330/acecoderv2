#!/usr/bin/env python3
"""
OpenAI Client Utilities for AceCoderV2

Centralized OpenAI client management to avoid code duplication
and provide consistent API access across the application.
"""

import os
import logging
from typing import Optional, Dict, Any
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

# Setup logging
logger = logging.getLogger(__name__)

class OpenAIClientManager:
    """Centralized OpenAI client manager for AceCoderV2"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize OpenAI client manager
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            base_url: Custom base URL for OpenAI API (if None, uses default)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL')
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize clients
        self._sync_client = None
        self._async_client = None
        
        logger.info(f"OpenAI client initialized with base_url: {self.base_url or 'default'}")
    
    @property
    def sync_client(self) -> OpenAI:
        """Get synchronous OpenAI client"""
        if self._sync_client is None:
            self._sync_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._sync_client
    
    @property
    def async_client(self) -> AsyncOpenAI:
        """Get asynchronous OpenAI client"""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._async_client
    
    def chat_completion(
        self,
        messages: list,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> ChatCompletion:
        """
        Create a chat completion using the sync client
        
        Args:
            messages: List of message dictionaries
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters for the API call
            
        Returns:
            ChatCompletion object
        """
        try:
            response = self.sync_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise
    
    async def async_chat_completion(
        self,
        messages: list,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> ChatCompletion:
        """
        Create a chat completion using the async client
        
        Args:
            messages: List of message dictionaries
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters for the API call
            
        Returns:
            ChatCompletion object
        """
        try:
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"Error in async chat completion: {e}")
            raise
    
    def validate_connection(self) -> bool:
        """
        Validate the OpenAI connection
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # Simple test call
            response = self.sync_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            logger.info("OpenAI connection validated successfully")
            return True
        except Exception as e:
            logger.error(f"OpenAI connection validation failed: {e}")
            return False

# Global client instance
_client_manager: Optional[OpenAIClientManager] = None

def get_openai_client(
    api_key: Optional[str] = None, 
    base_url: Optional[str] = None,
    force_reload: bool = False
) -> OpenAIClientManager:
    """
    Get or create the global OpenAI client manager
    
    Args:
        api_key: OpenAI API key (if None, will try to get from environment)
        base_url: Custom base URL for OpenAI API (if None, uses default)
        force_reload: Force reload the client even if one exists
        
    Returns:
        OpenAIClientManager instance
    """
    global _client_manager
    
    if _client_manager is None or force_reload:
        _client_manager = OpenAIClientManager(api_key=api_key, base_url=base_url)
    
    return _client_manager

def validate_api_key(api_key: Optional[str] = None) -> bool:
    """
    Validate OpenAI API key
    
    Args:
        api_key: API key to validate (if None, uses environment variable)
        
    Returns:
        True if valid, False otherwise
    """
    try:
        client = get_openai_client(api_key=api_key)
        return client.validate_connection()
    except Exception as e:
        logger.error(f"API key validation failed: {e}")
        return False

# Convenience functions for backward compatibility
def get_sync_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """Get synchronous OpenAI client (backward compatibility)"""
    return get_openai_client(api_key=api_key, base_url=base_url).sync_client

def get_async_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> AsyncOpenAI:
    """Get asynchronous OpenAI client (backward compatibility)"""
    return get_openai_client(api_key=api_key, base_url=base_url).async_client

# Example usage
if __name__ == "__main__":
    # Test the client
    try:
        client_manager = get_openai_client()
        print("✅ OpenAI client initialized successfully")
        
        # Test connection
        if client_manager.validate_connection():
            print("✅ OpenAI connection validated")
        else:
            print("❌ OpenAI connection failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please set OPENAI_API_KEY environment variable")
