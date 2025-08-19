"""Mistral API client for embeddings and chat completions."""

import requests
import json
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from app.core.config import config

logger = logging.getLogger(__name__)


class MistralAPIError(Exception):
    """Exception for Mistral API errors."""
    pass


class MistralClient:
    """Client for interacting with Mistral AI API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.MISTRAL_API_KEY
        self.base_url = config.MISTRAL_BASE_URL
        self.embed_model = config.MISTRAL_EMBED_MODEL
        self.chat_model = config.MISTRAL_CHAT_MODEL
        
        if not self.api_key:
            raise ValueError("Mistral API key is required")
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
    
    def get_embeddings(
        self, 
        texts: List[str],
        model: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            model: Model to use (defaults to configured embed model)
            
        Returns:
            List of embedding vectors as numpy arrays
            
        Raises:
            MistralAPIError: If the API request fails
        """
        if not texts:
            return []
        
        model = model or self.embed_model
        
        payload = {
            "model": model,
            "input": texts
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/embeddings",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "data" not in data:
                raise MistralAPIError(f"Unexpected response format: {data}")
            
            embeddings = []
            for item in data["data"]:
                if "embedding" not in item:
                    raise MistralAPIError(f"Missing embedding in response: {item}")
                embeddings.append(np.array(item["embedding"], dtype=np.float32))
            
            return embeddings
            
        except requests.RequestException as e:
            logger.error(f"Mistral API request failed: {e}")
            raise MistralAPIError(f"API request failed: {e}")
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing Mistral API response: {e}")
            raise MistralAPIError(f"Response parsing error: {e}")
    
    def get_single_embedding(self, text: str, model: Optional[str] = None) -> np.ndarray:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to embed
            model: Model to use (defaults to configured embed model)
            
        Returns:
            Embedding vector as numpy array
        """
        embeddings = self.get_embeddings([text], model)
        return embeddings[0] if embeddings else np.array([])
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Get chat completion from Mistral.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to configured chat model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
            
        Raises:
            MistralAPIError: If the API request fails
        """
        model = model or self.chat_model
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "choices" not in data or not data["choices"]:
                raise MistralAPIError(f"No choices in response: {data}")
            
            choice = data["choices"][0]
            if "message" not in choice or "content" not in choice["message"]:
                raise MistralAPIError(f"Invalid choice format: {choice}")
            
            return choice["message"]["content"].strip()
            
        except requests.RequestException as e:
            logger.error(f"Mistral chat API request failed: {e}")
            raise MistralAPIError(f"Chat API request failed: {e}")
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing Mistral chat API response: {e}")
            raise MistralAPIError(f"Chat response parsing error: {e}")


class EmbeddingManager:
    """Manages text embeddings using Mistral API."""
    
    def __init__(self, db_manager, mistral_client: Optional[MistralClient] = None):
        self.db_manager = db_manager
        self.mistral_client = mistral_client or MistralClient()
    
    def embed_and_store_chunks(self, chunk_data: List[Dict[str, Any]]):
        """
        Generate embeddings for chunks and store them in the database.
        
        Args:
            chunk_data: List of chunk dictionaries with 'chunk_id' and 'text'
        """
        if not chunk_data:
            return
        
        # Extract texts and chunk IDs
        texts = [chunk["text"] for chunk in chunk_data]
        chunk_ids = [chunk["chunk_id"] for chunk in chunk_data]
        
        try:
            # Get embeddings in batch
            embeddings = self.mistral_client.get_embeddings(texts)
            
            # Store embeddings in database
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                self.db_manager.store_embedding(chunk_id, embedding)
                
            logger.info(f"Successfully embedded and stored {len(chunk_data)} chunks")
            
        except MistralAPIError as e:
            logger.error(f"Failed to embed chunks: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Query embedding as numpy array
        """
        try:
            return self.mistral_client.get_single_embedding(query)
        except MistralAPIError as e:
            logger.error(f"Failed to embed query: {e}")
            raise


# Global instances
_mistral_client = None
_embedding_manager = None


def get_mistral_client() -> MistralClient:
    """Get a global Mistral client instance."""
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = MistralClient()
    return _mistral_client


def get_embedding_manager(db_manager) -> EmbeddingManager:
    """Get a global embedding manager instance."""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager(db_manager, get_mistral_client())
    return _embedding_manager