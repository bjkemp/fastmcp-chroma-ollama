# FastMCP ChromaDB Memory Server

## Overview

A local memory system for LLM assistants using ChromaDB, Ollama, and FastMCP. This project provides a persistent, semantically-aware memory storage and retrieval system with advanced features like memory importance scoring, automatic merging, and intelligent pruning.

## Features

- 🧠 Semantic Memory Storage
- 🔒 Local-first Architecture
- 🚀 High-Performance Vector Search
- 🤖 Ollama Embedding Support
- 📊 Memory Importance Scoring
- 🔍 Advanced Retrieval Mechanisms

## Requirements

- Python 3.8+
- Ollama
- ChromaDB
- FastMCP

## Installation

1. Install Ollama:
```bash
curl https://ollama.ai/install.sh | sh
ollama pull nomic-embed-text
```

2. Clone the repository:
```bash
git clone https://github.com/bjkemp/fastmcp-chroma-ollama.git
cd fastmcp-chroma-ollama
```

3. Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Starting the Server

#### STDIO Transport (for Claude Desktop):
```bash
python server.py --transport stdio
```

#### SSE Transport (for web applications):
```bash
python server.py --transport sse --port 8080
```

### Basic Client Example

```python
from memory import MemoryManager

async def example():
    manager = MemoryManager()
    
    # Store a memory
    memory_id = await manager.add_memory("The capital of France is Paris")
    
    # Retrieve memories
    memories = await manager.retrieve_memories("France's capital")
    print(memories)
```

## Contributing

Contributions are welcome! Please check the [TODO.md](TODO.md) for current development priorities.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Roadmap

Check out [TODO.md](TODO.md) for detailed development plans and future directions.

## Acknowledgements

- [ChromaDB](https://docs.trychroma.com/)
- [Ollama](https://ollama.ai/)
- [FastMCP](https://github.com/punkpeye/fastmcp)
