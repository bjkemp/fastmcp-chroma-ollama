# FastMCP ChromaDB Memory Server

## Overview

A local memory system for LLM assistants using ChromaDB, Ollama, and FastMCP. This project provides a persistent, semantically-aware memory storage and retrieval system with advanced features like memory importance scoring, automatic merging, and intelligent pruning.

## Requirements

- Python 3.8+
- [UV Package Manager](https://github.com/astral-sh/uv)
- Ollama
- ChromaDB
- FastMCP

## Installation

### Install UV Package Manager

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Project Setup

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

3. Create and activate virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. Install dependencies:
```bash
uv pip install -e .
```

## Development Setup

### Install Development Dependencies
```bash
uv pip install .[dev]
```

### Running Tests
```bash
uv run pytest tests/
```

## Usage

### Starting the Server

#### CLI
```bash
# Using UV
uv run -m fastmcp_memory.server --transport stdio

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

## Development Workflow

### Running Checks
```bash
# Run tests
uv run pytest

# Format code
uv run black src tests

# Sort imports
uv run isort src tests
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
- [UV Package Manager](https://github.com/astral-sh/uv)
