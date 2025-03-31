# FastMCP ChromaDB Memory Server with Ollama

## Project Overview

This project implements a memory system for LLM assistants using the Model Context Protocol (MCP). It provides a persistent memory storage solution using local tools without requiring external API dependencies.

### Key Components:

- **FastMCP**: Framework for building MCP servers that can handle client sessions
- **ChromaDB**: Vector database with SQLite backend for storing memory embeddings
- **Ollama**: Local embedding model service for generating vector embeddings

The memory system stores text as vector embeddings, allowing semantic search and retrieval based on meaning rather than exact text matching. It includes features like memory importance scoring, automatic memory merging, and memory pruning to maintain optimal collection size.

## Development Environment Setup

### Prerequisites

1. **Python Environment**
   - Python 3.7+ is required
   - UV package manager is used for dependency management

2. **Ollama**
   - Install from [ollama.ai](https://ollama.ai/)
   - Pull the required embedding model:
     ```
     ollama pull nomic-embed-text
     ```
   - Ensure Ollama is running with `ollama serve`

3. **Environment Configuration**
   - Source the environment configuration:
     ```
     source /Users/kempb/Projects/Claude/.clauderc
     ```

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd fastmcp-chroma-ollama
   ```

2. Install dependencies:
   ```
   /Users/kempb/.local/bin/uv pip install -r requirements.txt
   ```

3. Make scripts executable:
   ```
   chmod +x server.py client.py example_client.py
   ```

## Project Structure

### Files

- **memory.py**: Core implementation of the memory system
  - `OllamaClient`: Client for generating embeddings
  - `Memory`: Structured representation of a memory
  - `ChromaMemoryStore`: Interface to ChromaDB for storage
  - `MemoryManager`: Main class for memory operations
  - FastMCP tool definitions for the MCP interface

- **server.py**: Server entry point
  - Handles command-line arguments
  - Configures transport (stdio or SSE)
  - Initializes and starts the FastMCP server

- **client.py**: Interactive command-line client
  - Connects to the server
  - Provides a CLI for testing memory operations

- **example_client.py**: Example implementation
  - Demonstrates how to use the memory system in applications

- **requirements.txt**: Project dependencies

### Configuration

The memory system can be configured through several constants in `memory.py`:

- `MAX_MEMORIES`: Maximum number of memories to keep
- `SIMILARITY_THRESHOLD`: Threshold for memory similarity matching
- `DECAY_FACTOR`: Rate of importance decay for unused memories
- `REINFORCEMENT_FACTOR`: Rate of importance increase for accessed memories
- `OLLAMA_HOST`: Endpoint for Ollama API
- `EMBEDDING_MODEL`: Model used for generating embeddings
- `CHROMA_DIR`: Path for persistent storage

## Running the Server

### With stdio transport (for Claude Desktop):

```
/Users/kempb/.local/bin/uv run server.py --transport stdio
```

### With SSE transport (for web applications):

```
/Users/kempb/.local/bin/uv run server.py --transport sse --port 8080 --endpoint /sse
```

## Testing

To test the basic functionality:

```
/Users/kempb/.local/bin/uv run client.py
```

This opens an interactive client where you can enter commands to test the memory system.

## Integration with Claude

The memory system can be integrated with Claude through the MCP protocol. This allows Claude to store and retrieve memories across conversations, maintaining context and knowledge over time.

To use the memory system with Claude Desktop, follow the instructions in the README.md to add the server to your MCP configuration.

## Architecture

The memory system uses a vector database (ChromaDB) to store memories as embeddings. When a memory is added:

1. Text is converted to a vector embedding using Ollama
2. The embedding is stored in ChromaDB with metadata
3. Similar memories are found and potentially merged
4. Memory importance is updated based on similarity
5. Least important memories are pruned if needed

When retrieving memories:

1. The query is converted to an embedding
2. ChromaDB finds similar vectors based on cosine similarity
3. Matching memories above the similarity threshold are returned
4. Retrieved memories have their importance boosted

This approach allows finding semantically similar information even when the wording is different.