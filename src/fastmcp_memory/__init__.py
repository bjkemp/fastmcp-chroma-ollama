"""
FastMCP ChromaDB Memory Server

A local memory system for LLM assistants using ChromaDB, Ollama, and FastMCP.
"""

__version__ = "0.1.0"

from .memory import MemoryManager, Memory
from .server import MemoryMCPServer
from .client import FastMCPMemoryClient
