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

### From PyPI (Coming Soon)
```bash
pip install fastmcp-memory
```

### From GitHub
```bash
pip install git+https://github.com/bjkemp/fastmcp-chroma-ollama.git
```

### Manual Installation
1. Clone the repository:
```bash
git clone https://github.com/bjkemp/fastmcp-chroma-ollama.git
cd fastmcp-chroma-ollama
```

2. Install Ollama:
```bash
curl https://ollama.ai/install.sh | sh
ollama pull nomic-embed-text
```

3. Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

### Starting the Server

#### CLI
```bash
# Using the installed script
fastmcp-memory-server --transport stdio

# Or directly
python -m fastmcp_memory.server --transport stdio
```

### Basic Client Example

```python
from fastmcp_memory import MemoryManager

async def example():
    manager = MemoryManager()
    
    # Store a memory
    memory_id = await manager.add_memory("The capital of France is Paris")
    
    # Retrieve memories
    memories = await manager.retrieve_memories("France's capital")
    print(memories)
```

## Project Structure

```
fastmcp-chroma-ollama/
├── src/
│   └── fastmcp_memory/
│       ├── __init__.py
│       ├── memory.py
│       ├── server.py
│       ├── client.py
│       └── example_client.py
├── tests/
├── .github/
│   └── workflows/
├── pyproject.toml
├── README.md
├── TODO.md
└── LICENSE
```

## Development

### Running Tests
```bash
pip install .[dev]
pytest tests/
```

### Contributing

Contributions are welcome! Please check the [TODO.md](TODO.md) for current development priorities.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Roadmap

Check out [TODO.md](TODO.md) for detailed development plans and future directions.

## Acknowledgements

- [ChromaDB](https://docs.trychroma.com/)
- [Ollama](https://ollama.ai/)
- [FastMCP](https://github.com/punkpeye/fastmcp)
