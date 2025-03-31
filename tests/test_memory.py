import pytest
import asyncio
from fastmcp_memory import MemoryManager

@pytest.mark.asyncio
async def test_add_memory():
    """
    Test basic memory addition functionality.
    """
    manager = MemoryManager()
    
    # Add a simple memory
    memory_id = await manager.add_memory("Test memory content")
    
    # Verify memory ID is returned
    assert memory_id is not None
    assert isinstance(memory_id, str)

@pytest.mark.asyncio
async def test_retrieve_memories():
    """
    Test memory retrieval functionality.
    """
    manager = MemoryManager()
    
    # Add a test memory
    await manager.add_memory("The capital of France is Paris")
    
    # Retrieve memories
    memories = await manager.retrieve_memories("France")
    
    # Verify retrieval
    assert len(memories) > 0
    assert any("Paris" in memory['content'] for memory in memories)
