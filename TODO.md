# FastMCP ChromaDB Memory Server - TODO

## Action Plan

1. **Analyze the current memory.py file**
   - Review the existing code structure and functionality
   - Identify areas for improvement in organization, error handling, and efficiency

2. **Refactor memory.py**
   - Improve type hints and documentation
   - Enhance error handling
   - Optimize memory operations (merging, pruning, etc.)
   - Add better abstraction with class-based components

3. **Fix FastMCP integration**
   - Ensure proper tool definitions and signatures
   - Add appropriate error handling with UserError
   - Implement proper transport configuration for both stdio and SSE

4. **Set up execution environment**
   - Use `/Users/kempb/.local/bin/uv run` for executing Python scripts
   - Ensure shebang lines in scripts use `/usr/bin/env uv run` for portability
   - Properly source `/Users/kempb/Projects/Claude/.clauderc` for environment setup

5. **Set up server script**
   - Create a single entry point that handles both stdio and SSE transports
   - Add command-line arguments for configuration options
   - Implement proper error handling and logging

6. **Implement any missing features**
   - Additional search/filtering capabilities
   - Memory metadata and statistics
   - Better diagnostics for ChromaDB and Ollama

7. **Test the server**
   - Test with both stdio and SSE transport options
   - Verify ChromaDB persistence
   - Check Ollama embedding functionality
   - Create simple test client for verification

8. **Make scripts executable**
   - Set proper permissions for all scripts
   - Ensure proper shebang lines for direct execution

## Project Structure

```
fastmcp-chroma-ollama/
├── memory.py          # Main memory implementation
├── server.py          # Server entry point (supports stdio/SSE)
├── client.py          # Interactive CLI client
├── example_client.py  # Example client for easy testing
├── requirements.txt   # Dependencies
└── README.md          # Documentation
```

## Important Commands

- `source /Users/kempb/Projects/Claude/.clauderc` - Required for environment setup
- `/Users/kempb/.local/bin/uv pip install -r requirements.txt` - Install dependencies
- `/Users/kempb/.local/bin/uv run server.py` - Run server with stdio transport
- `/Users/kempb/.local/bin/uv run server.py --transport sse --port 8080` - Run with SSE transport

## External Dependencies

1. **FastMCP**: MCP server framework
   - Repo: https://github.com/punkpeye/fastmcp
   - Usage: Provides MCP protocol implementation

2. **ChromaDB**: Vector database
   - Version: >=0.4.18
   - Backend: SQLite (duckdb+parquet)
   - Storage path: `~/.fastmcp/{username}/memory_db`

3. **Ollama**: Local embedding model
   - Host: http://localhost:11434
   - Model: nomic-embed-text
   - Installation: https://ollama.ai/

## Configuration Options

- `MAX_MEMORIES`: Default=100, maximum number of memories to retain
- `SIMILARITY_THRESHOLD`: Default=0.75, threshold for similarity matching
- `DECAY_FACTOR`: Default=0.98, rate of importance decay for unused memories
- `REINFORCEMENT_FACTOR`: Default=1.2, rate of importance increase for used memories
- `OLLAMA_HOST`: Default="http://localhost:11434", Ollama API endpoint
- `EMBEDDING_MODEL`: Default="nomic-embed-text", model for embeddings

## Future Improvements

- [ ] Add authentication support
- [ ] Implement memory categorization/tagging
- [ ] Add bulk import/export capabilities
- [ ] Improve memory merging algorithm
- [ ] Add memory visualization tools
- [ ] Add support for alternative embedding models
- [ ] Implement more sophisticated importance scoring
- [ ] Add memory expiration based on time
- [ ] Implement memory compression for long-term storage

## Troubleshooting

1. **ChromaDB Connection Issues**
   - Check if the SQLite path exists and is writable
   - Verify ChromaDB version compatibility

2. **Ollama Issues**
   - Ensure Ollama is running (`ollama serve`)
   - Check if the model is available (`ollama list`)
   - Pull required model if missing (`ollama pull nomic-embed-text`)

3. **FastMCP Protocol Issues**
   - Check transport configuration
   - Verify client capabilities match server requirements

## Testing

To test the server functionality without a full MCP client:

```python
import asyncio
from memory import MemoryManager

async def test_memory():
    manager = MemoryManager()
    try:
        # Add a test memory
        result = await manager.add_memory("This is a test memory")
        print(f"Add result: {result}")
        
        # Retrieve related memories
        memories = await manager.retrieve_memories("test memory")
        print(f"Found {len(memories)} memories")
        for mem in memories:
            print(f"- {mem['content']} (similarity: {mem['similarity']:.2f})")
    finally:
        await manager.close()

if __name__ == "__main__":
    asyncio.run(test_memory())
```

Save this as `test_basic.py` and run with:
```
/Users/kempb/.local/bin/uv run test_basic.py
```