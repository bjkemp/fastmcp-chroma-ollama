"""
Recursive memory system with ChromaDB and Ollama embeddings.

Uses ChromaDB with SQLite backend for storage and Ollama for local embeddings.
No external API dependencies required.

Requirements:
    - uv pip install -r requirements.txt 
    - Ollama running locally with the nomic-embed-text model
"""

import asyncio
import datetime
import json
import math
import os
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import aiohttp
import chromadb
import numpy as np
from chromadb.config import Settings
# Define custom exception class (normally from fastmcp import UserError)
class UserError(Exception):
    """Error to be shown to the user."""
    pass

# Configuration constants
MAX_MEMORIES = 100
SIMILARITY_THRESHOLD = 0.75
DECAY_FACTOR = 0.98
REINFORCEMENT_FACTOR = 1.2

# Ollama settings
OLLAMA_HOST = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"  # A good lightweight embedding model
EMBEDDING_DIMENSIONS = 768  # nomic-embed-text uses 768 dimensions

# Set up persistent storage path
CHROMA_DIR = Path.home() / ".fastmcp" / os.environ.get("USER", "anon") / "memory_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Create tool registry (used in place of FastMCP for testing)
class ToolRegistry:
    def __init__(self, name, version, description=None):
        self.name = name
        self.version = version
        self.description = description
        self.tools = {}
        
    def tool(self, name):
        """Decorator to register a tool function."""
        def decorator(func):
            self.tools[name] = func
            return func
        return decorator
        
    def start(self, config=None):
        """Mock start method."""
        print(f"Starting {self.name} v{self.version}")
        print(f"Available tools: {', '.join(self.tools.keys())}")

# Create app registry
app = ToolRegistry(
    name="memory",
    version="0.1.0",
    description="Memory system with ChromaDB and Ollama"
)


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = OLLAMA_HOST):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def ensure_session(self) -> aiohttp.ClientSession:
        """Initialize or return existing session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
        
    async def get_embedding(self, text: str, model: str = EMBEDDING_MODEL) -> List[float]:
        """Get embeddings for a text using Ollama.
        
        Args:
            text: The text to embed
            model: The Ollama model to use for embeddings
            
        Returns:
            A list of floats representing the embedding vector
        """
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * EMBEDDING_DIMENSIONS
            
        session = await self.ensure_session()
        
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": model,
            "prompt": text
        }
        
        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Ollama API error ({response.status}): {error_text}")
                    
                data = await response.json()
                
                if "embedding" not in data:
                    raise ValueError(f"Unexpected response from Ollama: {json.dumps(data)[:100]}...")
                    
                return data["embedding"]
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return zero vector in case of error
            return [0.0] * EMBEDDING_DIMENSIONS
            
    async def close(self) -> None:
        """Close the client session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None


class Memory:
    """Structured representation of a memory."""
    
    def __init__(
        self,
        id: str,
        content: str,
        embedding: List[float],
        importance: float = 1.0,
        access_count: int = 0,
        timestamp: Optional[float] = None,
        last_accessed: Optional[float] = None,
        summary: Optional[str] = None,
        merged: bool = False,
        merged_from: Optional[List[str]] = None
    ):
        self.id = id
        self.content = content
        self.embedding = embedding
        self.importance = importance
        self.access_count = access_count
        self.timestamp = timestamp or datetime.datetime.now().timestamp()
        self.last_accessed = last_accessed
        self.summary = summary or self._generate_summary(content)
        self.merged = merged
        self.merged_from = merged_from or []
        
    @staticmethod
    def _generate_summary(content: str, max_length: int = 100) -> str:
        """Generate a summary of the content."""
        if len(content) <= max_length:
            return content
        
        # Try to find a sentence boundary
        for i in range(max_length - 10, max_length):
            if i < len(content) and content[i] in ['.', '!', '?']:
                return content[:i+1]
                
        return content[:max_length] + "..."
        
    @property
    def effective_importance(self) -> float:
        """Calculate effective importance based on importance and access count."""
        return self.importance * (1 + math.log(self.access_count + 1))
        
    @property
    def age_days(self) -> float:
        """Calculate the age of the memory in days."""
        now = datetime.datetime.now().timestamp()
        return (now - self.timestamp) / (60 * 60 * 24)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to a dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": {
                "importance": self.importance,
                "access_count": self.access_count,
                "timestamp": self.timestamp,
                "last_accessed": self.last_accessed,
                "summary": self.summary,
                "merged": self.merged,
                "merged_from": self.merged_from
            }
        }
        
    def to_response_dict(self, similarity: Optional[float] = None) -> Dict[str, Any]:
        """Convert memory to a dictionary for API response."""
        response = {
            "id": self.id,
            "content": self.content,
            "summary": self.summary,
            "importance": self.importance,
            "access_count": self.access_count,
            "timestamp": self.timestamp,
            "age_days": self.age_days,
        }
        
        if similarity is not None:
            response["similarity"] = similarity
            
        if self.last_accessed:
            response["last_accessed"] = self.last_accessed
            
        return response
    
    @classmethod
    def from_chroma_result(cls, result: Dict[str, Any], similarity: Optional[float] = None) -> "Memory":
        """Create a Memory object from a ChromaDB result."""
        metadata = result.get("metadata", {})
        
        memory = cls(
            id=result["id"],
            content=result["content"],
            embedding=result["embedding"],
            importance=metadata.get("importance", 1.0),
            access_count=metadata.get("access_count", 0),
            timestamp=metadata.get("timestamp"),
            last_accessed=metadata.get("last_accessed"),
            summary=metadata.get("summary"),
            merged=metadata.get("merged", False),
            merged_from=metadata.get("merged_from", [])
        )
        
        return memory


class ChromaMemoryStore:
    """Memory storage using ChromaDB with SQLite backend."""
    
    def __init__(self, persist_directory: Path):
        self.persist_directory = persist_directory
        
        # Configure ChromaDB client
        self.client = chromadb.Client(
            Settings(
                persist_directory=str(persist_directory),
                chroma_db_impl="duckdb+parquet",  # SQLite-compatible
                anonymized_telemetry=False
            )
        )
        
        # Create or get the collection
        try:
            self.collection = self.client.get_collection(name="memories")
            print(f"Using existing 'memories' collection")
        except ValueError:
            self.collection = self.client.create_collection(
                name="memories",
                metadata={"hnsw:space": "cosine"}  # Using cosine similarity
            )
            print(f"Created new 'memories' collection")
            
    def store_memory(self, memory: Memory) -> None:
        """Store a memory in ChromaDB.
        
        Args:
            memory: The Memory object to store
        """
        memory_dict = memory.to_dict()
        
        self.collection.add(
            ids=[memory.id],
            documents=[memory.content],
            embeddings=[memory.embedding],
            metadatas=[memory_dict["metadata"]]
        )
        
    def update_memory(self, memory: Memory) -> bool:
        """Update a memory in ChromaDB.
        
        Args:
            memory: The Memory object with updated values
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Check if memory exists
            result = self.collection.get(ids=[memory.id])
            
            if not result["ids"]:
                return False
            
            memory_dict = memory.to_dict()
            
            # Update in ChromaDB
            self.collection.update(
                ids=[memory.id],
                documents=[memory.content],
                embeddings=[memory.embedding],
                metadatas=[memory_dict["metadata"]]
            )
            return True
        except Exception as e:
            print(f"Error updating memory: {e}")
            return False
            
    def update_memory_metadata(self, memory_id: str, metadata_updates: Dict[str, Any]) -> bool:
        """Update only the metadata of a memory.
        
        Args:
            memory_id: ID of the memory to update
            metadata_updates: Dictionary of metadata fields to update
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Get current memory
            result = self.collection.get(ids=[memory_id])
            
            if not result["ids"]:
                return False
                
            # Update metadata fields
            current_metadata = result["metadatas"][0]
            current_metadata.update(metadata_updates)
            
            # Update in ChromaDB
            self.collection.update(
                ids=[memory_id],
                metadatas=[current_metadata]
            )
            return True
        except Exception as e:
            print(f"Error updating memory metadata: {e}")
            return False
            
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from ChromaDB.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            print(f"Error deleting memory: {e}")
            return False
            
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory object if found, None otherwise
        """
        try:
            result = self.collection.get(
                ids=[memory_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if not result["ids"]:
                return None
            
            # Convert to Memory object
            return Memory.from_chroma_result({
                "id": result["ids"][0],
                "content": result["documents"][0],
                "embedding": result["embeddings"][0],
                "metadata": result["metadatas"][0]
            })
        except Exception as e:
            print(f"Error getting memory: {e}")
            return None
            
    def find_similar_memories(
        self, 
        embedding: List[float], 
        limit: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[Memory, float]]:
        """Find memories similar to the given embedding.
        
        Args:
            embedding: The query embedding vector
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of tuples containing (Memory, similarity_score)
        """
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=limit,
                include=["documents", "metadatas", "embeddings", "distances"]
            )
            
            memories = []
            
            # No results
            if not results["ids"][0]:
                return []
                
            for i, (doc_id, doc, metadata, embedding_vec, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["embeddings"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity (1.0 is exact match)
                similarity = 1.0 - distance
                
                # Skip results below threshold
                if threshold is not None and similarity < threshold:
                    continue
                
                # Create Memory object
                memory = Memory(
                    id=doc_id,
                    content=doc,
                    embedding=embedding_vec,
                    importance=metadata.get("importance", 1.0),
                    access_count=metadata.get("access_count", 0),
                    timestamp=metadata.get("timestamp"),
                    last_accessed=metadata.get("last_accessed"),
                    summary=metadata.get("summary"),
                    merged=metadata.get("merged", False),
                    merged_from=metadata.get("merged_from", [])
                )
                
                memories.append((memory, similarity))
                    
            return memories
        except Exception as e:
            print(f"Error finding similar memories: {e}")
            return []
            
    def get_all_memories(self, limit: int = 100) -> List[Memory]:
        """Get all memories.
        
        Args:
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of Memory objects
        """
        try:
            results = self.collection.get(
                limit=limit,
                include=["documents", "metadatas", "embeddings"]
            )
            
            if not results["ids"]:
                return []
                
            memories = []
            
            for i, (doc_id, doc, metadata, embedding) in enumerate(zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
                results["embeddings"]
            )):
                memory = Memory(
                    id=doc_id,
                    content=doc,
                    embedding=embedding,
                    importance=metadata.get("importance", 1.0),
                    access_count=metadata.get("access_count", 0),
                    timestamp=metadata.get("timestamp"),
                    last_accessed=metadata.get("last_accessed"),
                    summary=metadata.get("summary"),
                    merged=metadata.get("merged", False),
                    merged_from=metadata.get("merged_from", [])
                )
                
                memories.append(memory)
                
            return memories
        except Exception as e:
            print(f"Error getting all memories: {e}")
            return []
            
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory collection.
        
        Returns:
            Dictionary with collection statistics
        """
        memories = self.get_all_memories()
        
        if not memories:
            return {
                "count": 0,
                "oldest": None,
                "newest": None,
                "avg_importance": None,
                "max_importance": None
            }
            
        importances = [memory.importance for memory in memories]
        timestamps = [memory.timestamp for memory in memories]
        
        return {
            "count": len(memories),
            "oldest": datetime.datetime.fromtimestamp(min(timestamps)).isoformat(),
            "newest": datetime.datetime.fromtimestamp(max(timestamps)).isoformat(),
            "avg_importance": sum(importances) / len(importances),
            "max_importance": max(importances)
        }


class MemoryManager:
    """Manager for memory operations."""
    
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.store = ChromaMemoryStore(CHROMA_DIR)
        
    async def add_memory(self, content: str) -> str:
        """Add a new memory.
        
        Args:
            content: The text content to store
            
        Returns:
            Status message
            
        Raises:
            ValueError: If content is empty
        """
        if not content.strip():
            raise ValueError("Cannot add empty memory")
            
        # Create a memory ID based on timestamp with some randomness
        memory_id = f"mem_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        
        # Get embedding from Ollama
        embedding = await self.ollama_client.get_embedding(content)
        
        # Create memory object
        memory = Memory(
            id=memory_id,
            content=content,
            embedding=embedding,
            timestamp=datetime.datetime.now().timestamp()
        )
        
        # Store in ChromaDB
        self.store.store_memory(memory)
        
        # Find similar memories for potential merging
        similar_memories = self.store.find_similar_memories(
            embedding=embedding,
            threshold=SIMILARITY_THRESHOLD
        )
        
        # Merge if very similar memory exists
        for similar_memory, similarity in similar_memories:
            if similar_memory.id != memory_id and similarity > 0.9:
                await self.merge_memories(memory_id, similar_memory.id)
                break
                
        # Update importance of related memories
        await self.update_importance(embedding)
        
        # Prune if too many memories
        await self.prune_memories()
        
        return f"Remembered: {memory.summary}"
        
    async def merge_memories(self, memory_id1: str, memory_id2: str) -> bool:
        """Merge two memories together.
        
        Args:
            memory_id1: ID of the first memory (will be kept)
            memory_id2: ID of the second memory (will be deleted)
            
        Returns:
            True if merge was successful, False otherwise
        """
        memory1 = self.store.get_memory(memory_id1)
        memory2 = self.store.get_memory(memory_id2)
        
        if not memory1 or not memory2:
            return False
            
        # Combine content
        combined_content = f"{memory1.content}\n\n{memory2.content}"
        
        # Normalize and average the embeddings
        embedding1 = np.array(memory1.embedding)
        embedding2 = np.array(memory2.embedding)
        
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 > 0:
            embedding1 = embedding1 / norm1
        if norm2 > 0:
            embedding2 = embedding2 / norm2
            
        # Average the normalized embeddings
        combined_embedding = ((embedding1 + embedding2) / 2).tolist()
        
        # Create merged memory
        merged_from = memory1.merged_from + memory2.merged_from
        if memory1.id not in merged_from:
            merged_from.append(memory1.id)
        if memory2.id not in merged_from:
            merged_from.append(memory2.id)
            
        merged_memory = Memory(
            id=memory1.id,
            content=combined_content,
            embedding=combined_embedding,
            importance=memory1.importance + memory2.importance,
            access_count=memory1.access_count + memory2.access_count,
            timestamp=datetime.datetime.now().timestamp(),
            merged=True,
            merged_from=merged_from
        )
        
        # Update the first memory with merged data
        success = self.store.update_memory(merged_memory)
        
        if success:
            # Delete the second memory
            self.store.delete_memory(memory2.id)
            return True
        
        return False
        
    async def update_importance(self, query_embedding: List[float]) -> None:
        """Update importance of memories based on similarity to query.
        
        Args:
            query_embedding: The query embedding to compare against
        """
        memories = self.store.get_all_memories()
        
        # Convert query embedding to numpy array
        query_vector = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vector)
        
        if query_norm > 0:
            query_vector = query_vector / query_norm
            
        for memory in memories:
            # Convert memory embedding to numpy array
            memory_vector = np.array(memory.embedding)
            memory_norm = np.linalg.norm(memory_vector)
            
            if memory_norm > 0:
                memory_vector = memory_vector / memory_norm
                
            # Calculate cosine similarity
            similarity = np.dot(query_vector, memory_vector)
            
            # Update importance based on similarity
            if similarity > SIMILARITY_THRESHOLD:
                # Reinforce similar memories
                new_importance = memory.importance * REINFORCEMENT_FACTOR
                new_access_count = memory.access_count + 1
            else:
                # Decay dissimilar memories
                new_importance = memory.importance * DECAY_FACTOR
                new_access_count = memory.access_count
                
            # Update metadata
            self.store.update_memory_metadata(
                memory_id=memory.id,
                metadata_updates={
                    "importance": new_importance,
                    "access_count": new_access_count
                }
            )
            
    async def prune_memories(self) -> None:
        """Remove least important memories if we exceed the maximum."""
        memories = self.store.get_all_memories()
        
        if len(memories) <= MAX_MEMORIES:
            return
            
        # Sort by effective importance (descending)
        memories.sort(key=lambda x: x.effective_importance, reverse=True)
        
        # Delete excess memories with lowest importance
        for memory in memories[MAX_MEMORIES:]:
            self.store.delete_memory(memory.id)
            
    async def retrieve_memories(
        self, 
        query: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve memories related to a query.
        
        Args:
            query: The search query text
            limit: Maximum number of results to return
            
        Returns:
            List of memory records with similarity scores
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        # Get embedding for query
        embedding = await self.ollama_client.get_embedding(query)
        
        # Find similar memories
        memory_results = self.store.find_similar_memories(
            embedding=embedding,
            limit=limit,
            threshold=SIMILARITY_THRESHOLD
        )
        
        # No results
        if not memory_results:
            return []
        
        # Update importance of found memories
        for memory, _ in memory_results:
            # Increase importance and access count
            self.store.update_memory_metadata(
                memory_id=memory.id,
                metadata_updates={
                    "importance": memory.importance * REINFORCEMENT_FACTOR,
                    "access_count": memory.access_count + 1,
                    "last_accessed": datetime.datetime.now().timestamp()
                }
            )
            
        # Prepare response
        results = []
        for memory, similarity in memory_results:
            # Add similarity to memory response
            memory_dict = memory.to_response_dict(similarity=similarity)
            results.append(memory_dict)
            
        return results
        
    async def get_memory_profile(self) -> str:
        """Build a summary of the current memory profile.
        
        Returns:
            Formatted string with memory statistics
        """
        memories = self.store.get_all_memories(limit=10)
        
        if not memories:
            return "No memories stored yet."
            
        # Sort by effective importance
        memories.sort(key=lambda x: x.effective_importance, reverse=True)
        
        # Get collection stats
        stats = self.store.get_collection_stats()
        
        # Build profile
        result = [
            f"Memory Profile ({stats['count']} total memories):\n",
            f"Oldest: {stats['oldest'] if stats['oldest'] else 'N/A'}\n",
            f"Newest: {stats['newest'] if stats['newest'] else 'N/A'}\n\n",
            "Top memories by importance:\n"
        ]
        
        for i, memory in enumerate(memories[:10]):
            date_str = datetime.datetime.fromtimestamp(memory.timestamp).strftime("%Y-%m-%d %H:%M")
            
            result.append(
                f"{i+1}. {memory.summary} "
                f"(Importance: {memory.effective_importance:.2f}, "
                f"Accessed: {memory.access_count}, "
                f"Date: {date_str})\n"
            )
            
        return "".join(result)
        
    async def search_by_text(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for memories by exact text matching.
        
        Args:
            text: Text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching memories
        """
        memories = self.store.get_all_memories()
        
        # Filter memories containing the text
        matches = []
        for memory in memories:
            if text.lower() in memory.content.lower():
                matches.append(memory)
                
        # Sort by importance
        matches.sort(key=lambda x: x.effective_importance, reverse=True)
        
        # Limit results
        matches = matches[:limit]
        
        # Convert to response format
        return [memory.to_response_dict() for memory in matches]
    
    async def delete_by_id(self, memory_id: str) -> bool:
        """Delete a memory by ID.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        return self.store.delete_memory(memory_id)
    
    async def clear_all_memories(self) -> int:
        """Remove all memories.
        
        Returns:
            Number of memories deleted
        """
        memories = self.store.get_all_memories()
        count = len(memories)
        
        for memory in memories:
            self.store.delete_memory(memory.id)
            
        return count
        
    async def close(self):
        """Clean up resources."""
        await self.ollama_client.close()


# FastMCP tools
@app.tool("remember")
async def remember(
    content: str
) -> str:
    """Store a new memory."""
    manager = MemoryManager()
    try:
        return await manager.add_memory(content)
    except ValueError as e:
        raise UserError(str(e))
    finally:
        await manager.close()


@app.tool("remember_multiple")
async def remember_multiple(
    contents: List[str]
) -> str:
    """Store multiple memories."""
    manager = MemoryManager()
    try:
        results = []
        for content in contents:
            try:
                result = await manager.add_memory(content)
                results.append(result)
            except ValueError as e:
                results.append(f"Error: {str(e)}")
                
        return "\n".join(results)
    finally:
        await manager.close()


@app.tool("recall")
async def recall(
    query: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Recall memories related to a query."""
    manager = MemoryManager()
    try:
        return await manager.retrieve_memories(query, limit)
    except ValueError as e:
        raise UserError(str(e))
    finally:
        await manager.close()


@app.tool("search")
async def search(
    text: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Search for memories by text matching."""
    manager = MemoryManager()
    try:
        return await manager.search_by_text(text, limit)
    finally:
        await manager.close()


@app.tool("delete")
async def delete(
    memory_id: str
) -> str:
    """Delete a memory by ID."""
    manager = MemoryManager()
    try:
        success = await manager.delete_by_id(memory_id)
        if success:
            return f"Memory {memory_id} deleted successfully"
        else:
            raise UserError(f"Failed to delete memory {memory_id}")
    finally:
        await manager.close()


@app.tool("clear_all")
async def clear_all() -> str:
    """Clear all memories."""
    manager = MemoryManager()
    try:
        count = await manager.clear_all_memories()
        return f"Cleared {count} memories"
    finally:
        await manager.close()


@app.tool("get_profile")
async def get_profile() -> str:
    """Get a summary of the current memory profile."""
    manager = MemoryManager()
    try:
        return await manager.get_memory_profile()
    finally:
        await manager.close()


async def check_ollama_availability():
    """Check if Ollama is available and has the required model."""
    client = OllamaClient()
    try:
        session = await client.ensure_session()
        async with session.get(f"{OLLAMA_HOST}/api/tags") as response:
            if response.status != 200:
                print(f"Warning: Ollama is not available at {OLLAMA_HOST}")
                print("Please install Ollama from https://ollama.ai/ and ensure it's running")
                return False
                
            data = await response.json()
            models = [model["name"] for model in data.get("models", [])]
            
            if EMBEDDING_MODEL not in models:
                print(f"Warning: Model '{EMBEDDING_MODEL}' not found in Ollama")
                print(f"Please run: ollama pull {EMBEDDING_MODEL}")
                return False
                
            print(f"✓ Ollama is available with model '{EMBEDDING_MODEL}'")
            return True
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print(f"Please ensure Ollama is running at {OLLAMA_HOST}")
        return False
    finally:
        await client.close()


if __name__ == "__main__":
    # Check Ollama availability
    loop = asyncio.get_event_loop()
    loop.run_until_complete(check_ollama_availability())
    
    # Start the FastMCP server
    app.start({
        "transportType": "stdio"  # Default transport type
    })
