# FastMCP ChromaDB Memory Server

## Overview

A sophisticated, local memory system for AI assistants using ChromaDB, Ollama, and FastMCP. This project provides a persistent, semantically-aware memory storage and retrieval system with advanced features like memory importance scoring, automatic merging, and intelligent pruning.

## 🌟 Features

- 🧠 Semantic Memory Storage
- 🔒 Local-first Architecture
- 🚀 High-Performance Vector Search
- 🤖 Ollama Embedding Support
- 📊 Memory Importance Scoring
- 🔍 Advanced Retrieval Mechanisms

## 🛠 Requirements

- Python 3.10+
- [UV Package Manager](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.ai/)
- ChromaDB
- FastMCP

## 📦 Installation

### 1. Install UV Package Manager

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Ollama

```bash
curl https://ollama.ai/install.sh | sh
ollama pull nomic-embed-text
```

### 3. Install Project

```bash
# Clone the repository
git clone https://github.com/bjkemp/fastmcp-chroma-ollama.git
cd fastmcp-chroma-ollama

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project
uv pip install -e .
```

## 🚀 Usage

### Starting the Server

```bash
# With stdio transport (for Claude Desktop)
uv run -m fastmcp_memory.server --transport stdio

# With SSE transport
uv run -m fastmcp_memory.server --transport sse --port 8080
```

### Basic Python Example

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

## 🧪 Development

### Setup Development Environment

```bash
# Install development dependencies
uv pip install .[dev]

# Run tests
uv run pytest

# Run code formatters
uv run black src tests
uv run isort src tests
```

## 📋 Project Structure

```
fastmcp-chroma-ollama/
├── src/
│   └── fastmcp_memory/
│       ├── __init__.py
│       ├── server.py
│       ├── server_core.py
│       └── ...
├── tests/
├── .github/
│   └── workflows/
├── pyproject.toml
├── README.md
├── TODO.md
└── LICENSE
```

## 🤝 Contributing

Contributions are welcome! Please check our [TODO.md](TODO.md) for current development priorities and see our contribution guidelines.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🗺 Roadmap

Check out [TODO.md](TODO.md) for detailed development plans and future directions.

## 🙏 Acknowledgements

- [ChromaDB](https://docs.trychroma.com/)
- [Ollama](https://ollama.ai/)
- [FastMCP](https://github.com/jlowin/fastmcp)
- [UV Package Manager](https://github.com/astral-sh/uv)
