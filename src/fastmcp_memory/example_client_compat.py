#!/usr/bin/env python3
"""
Compatibility client for FastMCP Memory Server designed for Claude integration.

This client provides simplified function calls that can be directly invoked from
Claude conversations through Python code execution.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Union

from modelcontextprotocol.client import Client
from modelcontextprotocol.client.stdio import StdioClientTransport
from modelcontextprotocol.client.sse import SSEClientTransport


class MemoryClient:
    """Memory client for integration with Claude."""
    
    def __init__(
        self,
        transport_type: str = "stdio",
        sse_url: str = "http://localhost:8080/sse",
        stdio_command: str = "./server.py"
    ):
        """Initialize the memory client.
        
        Args:
            transport_type: The transport type ("stdio" or "sse")
            sse_url: The URL for SSE transport
            stdio_command: The command for stdio transport
        """
        self.transport_type = transport_type
        self.sse_url = sse_url
        self.stdio_command = stdio_command
        self.client = None
        self.connected = False
    
    async def connect(self) -> bool:
        """Connect to the memory server.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.connected:
            return True
            
        # Create client
        self.client = Client(
            {
                "name": "claude-memory-client",
                "version": "0.1.0"
            },
            {
                "capabilities": {
                    "stream": True
                }
            }
        )
        
        # Connect with appropriate transport
        try:
            if self.transport_type == "stdio":
                transport = StdioClientTransport(self.stdio_command)
            else:
                transport = SSEClientTransport(self.sse_url)
                
            await self.client.connect(transport)
            self.connected = True
            return True
            
        except Exception as e:
            print(f"Connection error: {e}")
            self.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the memory server."""
        if self.connected and self.client:
            await self.client.disconnect()
            self.connected = False
    
    async def remember(self, content: str) -> str:
        """Store a new memory.
        
        Args:
            content: The text content to store
            
        Returns:
            Status message
        """
        if not await self.connect():
            return "Failed to connect to memory server"
            
        try:
            result = await self.client.execute_tool("remember", {"content": content})
            return result
        except Exception as e:
            return f"Error storing memory: {e}"
    
    async def recall(
        self, 
        query: str, 
        limit: int = 5,
        format_output: bool = True
    ) -> Union[str, List[Dict]]:
        """Recall memories related to a query.
        
        Args:
            query: The search query
            limit: Maximum number of results
            format_output: Whether to format the output as a string
            
        Returns:
            Formatted string or raw memory objects
        """
        if not await self.connect():
            return "Failed to connect to memory server"
            
        try:
            memories = await self.client.execute_tool(
                "recall", 
                {"query": query, "limit": limit}
            )
            
            if not memories:
                return "No relevant memories found."
                
            if not format_output:
                return memories
            
            # Format memories for display
            result = [f"Found {len(memories)} relevant memories:"]
            
            for i, memory in enumerate(memories, 1):
                # Format timestamp
                timestamp = datetime.fromtimestamp(
                    memory.get("timestamp", 0)
                ).strftime("%Y-%m-%d %H:%M")
                
                similarity = memory.get("similarity", 0) * 100  # Convert to percentage
                
                result.append(
                    f"{i}. {memory.get('content')} "
                    f"(Relevance: {similarity:.1f}%, Date: {timestamp})"
                )
            
            return "\n\n".join(result)
            
        except Exception as e:
            return f"Error recalling memories: {e}"
    
    async def search(
        self, 
        text: str, 
        limit: int = 5,
        format_output: bool = True
    ) -> Union[str, List[Dict]]:
        """Search for memories by text matching.
        
        Args:
            text: Text to search for
            limit: Maximum number of results
            format_output: Whether to format the output as a string
            
        Returns:
            Formatted string or raw memory objects
        """
        if not await self.connect():
            return "Failed to connect to memory server"
            
        try:
            memories = await self.client.execute_tool(
                "search", 
                {"text": text, "limit": limit}
            )
            
            if not memories:
                return "No matching memories found."
                
            if not format_output:
                return memories
            
            # Format memories for display
            result = [f"Found {len(memories)} matching memories:"]
            
            for i, memory in enumerate(memories, 1):
                # Format timestamp
                timestamp = datetime.fromtimestamp(
                    memory.get("timestamp", 0)
                ).strftime("%Y-%m-%d %H:%M")
                
                result.append(
                    f"{i}. {memory.get('content')} "
                    f"(Date: {timestamp})"
                )
            
            return "\n\n".join(result)
            
        except Exception as e:
            return f"Error searching memories: {e}"
    
    async def get_profile(self) -> str:
        """Get memory profile summary.
        
        Returns:
            Formatted string with memory statistics
        """
        if not await self.connect():
            return "Failed to connect to memory server"
            
        try:
            return await self.client.execute_tool("get_profile", {})
        except Exception as e:
            return f"Error getting memory profile: {e}"
    
    async def delete(self, memory_id: str) -> str:
        """Delete a memory by ID.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            Status message
        """
        if not await self.connect():
            return "Failed to connect to memory server"
            
        try:
            return await self.client.execute_tool("delete", {"memory_id": memory_id})
        except Exception as e:
            return f"Error deleting memory: {e}"


# Helper functions for direct use in Claude


async def _run_memory_client(func_name, **kwargs):
    """Run a memory client function and return the result."""
    client = MemoryClient()
    try:
        func = getattr(client, func_name)
        return await func(**kwargs)
    finally:
        await client.disconnect()


def remember(content):
    """Store a memory."""
    return asyncio.run(_run_memory_client("remember", content=content))


def recall(query, limit=5, format_output=True):
    """Recall memories related to a query."""
    return asyncio.run(_run_memory_client(
        "recall", 
        query=query, 
        limit=limit,
        format_output=format_output
    ))


def search(text, limit=5, format_output=True):
    """Search for memories by text matching."""
    return asyncio.run(_run_memory_client(
        "search", 
        text=text, 
        limit=limit,
        format_output=format_output
    ))


def get_profile():
    """Get memory profile summary."""
    return asyncio.run(_run_memory_client("get_profile"))


def delete(memory_id):
    """Delete a memory by ID."""
    return asyncio.run(_run_memory_client("delete", memory_id=memory_id))


# Example usage if run directly
if __name__ == "__main__":
    # Get command line arguments
    if len(sys.argv) < 2:
        print("Usage: python example_client_compat.py <function> [args...]")
        sys.exit(1)
        
    func_name = sys.argv[1]
    
    if func_name == "remember" and len(sys.argv) > 2:
        print(remember(sys.argv[2]))
    elif func_name == "recall" and len(sys.argv) > 2:
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        print(recall(sys.argv[2], limit))
    elif func_name == "search" and len(sys.argv) > 2:
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        print(search(sys.argv[2], limit))
    elif func_name == "profile":
        print(get_profile())
    elif func_name == "delete" and len(sys.argv) > 2:
        print(delete(sys.argv[2]))
    else:
        print(f"Unknown function or missing arguments: {func_name}")#!/usr/bin/env python3
"""
Compatible example client for ChromaDB memory system with Ollama.

This version is designed to work with both older and newer MCP library versions.
It provides fallbacks and version detection to help with compatibility issues.

To run this example:
1. Ensure Ollama is installed and running (https://ollama.ai/)
2. Pull the embedding model: ollama pull nomic-embed-text
3. Install dependencies: uv pip install -r requirements.txt
4. Run this script: python example_client_compat.py
"""

import asyncio
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    # Try to get version info
    mcp_version = importlib.metadata.version("mcp")
    print(f"Detected MCP version: {mcp_version}")
except Exception as e:
    print(f"Could not detect MCP version: {e}")
    mcp_version = "unknown"

# Import MCP libraries
from mcp import ClientSession

# Different ways to import based on version
try:
    # Try newer interface first
    from mcp.client.stdio import StdioTransport
    USING_NEW_API = True
    print("Using new MCP stdio API (StdioTransport)")
except ImportError:
    try:
        # Try older interface
        from mcp.client.stdio import stdio_client
        USING_NEW_API = False
        print("Using legacy MCP stdio API (stdio_client)")
    except ImportError:
        print("ERROR: Could not import required MCP libraries. Please check your installation.")
        sys.exit(1)


async def main() -> None:
    """Run the example memory client with version compatibility."""
    # Start the server process
    server_cmd = sys.executable  # Path to current Python executable
    server_process = subprocess.Popen(
        [server_cmd, str(Path(__file__).parent / "memory.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False
    )
    
    print("Started memory server process...")
    # Give the server a moment to initialize
    time.sleep(2)
    
    try:
        # Connect to the server using the appropriate API
        print("Connecting to server...")
        
        # Try multiple approaches to connect
        try:
            if USING_NEW_API:
                # New API approach
                transport = StdioTransport(server_process.stdout, server_process.stdin)
                async with transport:
                    read, write = transport.get_transport_functions()
                    async with ClientSession(read, write) as session:
                        await run_demo(session)
            else:
                # Try direct connection without stdio_client
                read = lambda: server_process.stdout.readline()
                write = lambda data: server_process.stdin.write(data + b"\n") or server_process.stdin.flush()
                
                async with ClientSession(read, write) as session:
                    await run_demo(session)
        except Exception as e:
            print(f"First connection method failed: {e}")
            print("Trying alternative connection method...")
            
            # Direct connection as fallback
            read = lambda: server_process.stdout.readline()
            write = lambda data: server_process.stdin.write(data + b"\n") or server_process.stdin.flush()
            
            async with ClientSession(read, write) as session:
                await run_demo(session)
                    
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up the server process
        if server_process:
            print("\nShutting down server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
                
        print("\nDemo completed.")


async def run_demo(session: ClientSession) -> None:
    """Run the actual demo with the connected session."""
    await session.initialize()
    
    # List available tools
    tools = await session.list_tools()
    tool_names = [tool.name for tool in tools[0][1]]
    print(f"Available tools: {tool_names}\n")
    
    # Store some example memories
    print("Storing example memories...")
    
    await session.call_tool("remember", {
        "content": "The capital of France is Paris, a beautiful city known for the Eiffel Tower and fine cuisine."
    })
    
    await session.call_tool("remember", {
        "content": "Rome is the capital of Italy and was once the center of the Roman Empire."
    })
    
    await session.call_tool("remember_multiple", {
        "contents": [
            "Python is a programming language created by Guido van Rossum.",
            "The Mediterranean Sea is connected to the Atlantic Ocean.",
            "The Great Wall of China is visible from space and spans thousands of kilometers."
        ]
    })
    
    # Check memory profile
    print("\nChecking memory profile...")
    profile = await session.call_tool("get_profile")
    print(profile)
    
    # Recall related memories
    print("\nRecalling memories about European capitals...")
    results = await session.call_tool("recall", {
        "query": "What are some famous European capital cities?",
        "limit": 3
    })
    
    print(f"Found {len(results)} related memories:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['content']} (similarity: {result['similarity']:.4f})")
    
    # Recall with different query
    print("\nRecalling memories about oceans...")
    results = await session.call_tool("recall", {
        "query": "Tell me about bodies of water and oceans",
        "limit": 2
    })
    
    print(f"Found {len(results)} related memories:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['content']} (similarity: {result['similarity']:.4f})")
    
    # Add a memory that might merge with existing one
    print("\nAdding memory that might merge with existing one...")
    await session.call_tool("remember", {
        "content": "Paris is the largest city in France and its capital."
    })
    
    # Check updated profile
    print("\nChecking updated memory profile...")
    profile = await session.call_tool("get_profile")
    print(profile)


if __name__ == "__main__":
    asyncio.run(main())
