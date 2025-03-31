#!/usr/bin/env python3
"""
Example client for FastMCP Memory Server optimized for use with Claude.

This shows how to use the memory server to store and retrieve memories within Claude conversations.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pprint import pprint

from modelcontextprotocol.client import Client
from modelcontextprotocol.client.stdio import StdioClientTransport
from modelcontextprotocol.client.sse import SSEClientTransport


async def store_memory(client, content):
    """Store a new memory."""
    return await client.execute_tool("remember", {"content": content})


async def retrieve_memories(client, query, limit=5):
    """Retrieve memories related to a query."""
    memories = await client.execute_tool("recall", {"query": query, "limit": limit})
    
    if not memories:
        return "No relevant memories found."
    
    # Format memories for display
    result = [f"Found {len(memories)} relevant memories:"]
    
    for i, memory in enumerate(memories, 1):
        # Format timestamp
        timestamp = datetime.fromtimestamp(memory.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M")
        similarity = memory.get("similarity", 0) * 100  # Convert to percentage
        
        result.append(
            f"{i}. {memory.get('content')} "
            f"(Relevance: {similarity:.1f}%, Date: {timestamp})"
        )
    
    return "\n\n".join(result)


async def get_profile(client):
    """Get memory profile summary."""
    return await client.execute_tool("get_profile", {})


async def main():
    """Main example function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FastMCP Memory Example Client")
    parser.add_argument(
        "action",
        choices=["store", "retrieve", "profile"],
        help="Action to perform"
    )
    parser.add_argument(
        "--content",
        help="Content to store (for store action)"
    )
    parser.add_argument(
        "--query",
        help="Query to search for (for retrieve action)"
    )
    parser.add_argument(
        "--transport", 
        choices=["stdio", "sse"], 
        default="stdio",
        help="Transport type (stdio or sse)"
    )
    parser.add_argument(
        "--server", 
        default="http://localhost:8080/sse",
        help="SSE server URL (for SSE transport)"
    )
    parser.add_argument(
        "--command", 
        default="./server.py",
        help="Server command (for stdio transport)"
    )
    args = parser.parse_args()
    
    # Create client
    client = Client(
        {
            "name": "memory-example-client",
            "version": "0.1.0"
        },
        {
            "capabilities": {
                "stream": True
            }
        }
    )
    
    # Connect with appropriate transport
    if args.transport == "stdio":
        transport = StdioClientTransport(args.command)
    else:
        transport = SSEClientTransport(args.server)
    
    try:
        await client.connect(transport)
        
        # Perform the requested action
        if args.action == "store":
            if not args.content:
                print("Error: --content is required for store action")
                return
            result = await store_memory(client, args.content)
            print(result)
            
        elif args.action == "retrieve":
            if not args.query:
                print("Error: --query is required for retrieve action")
                return
            result = await retrieve_memories(client, args.query)
            print(result)
            
        elif args.action == "profile":
            result = await get_profile(client)
            print(result)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    # Run the async main function
    if sys.version_info >= (3, 7):
        asyncio.run(main())
    else:
        # Fallback for Python 3.6
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())#!/usr/bin/env python3
"""Example client for ChromaDB memory system with Ollama.

To run this example:
1. Ensure Ollama is installed and running (https://ollama.ai/)
2. Pull the embedding model: ollama pull nomic-embed-text
3. Install dependencies: uv pip install -r requirements.txt
4. Run this script: uv pip run python example_client.py
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.stdio import stdio_client


async def main():
    """Run the example memory client."""
    # Start the memory server in a subprocess
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
        # Connect to the server
        print("Connecting to server...")
        async with stdio_client(server_process.stdout, server_process.stdin) as (read, write):
            async with ClientSession(read, write) as session:
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


if __name__ == "__main__":
    asyncio.run(main())
