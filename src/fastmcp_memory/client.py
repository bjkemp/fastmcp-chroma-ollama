#!/usr/bin/env python3
"""
Test client for the FastMCP Memory Server.

This client connects to the memory server and provides a simple CLI to interact with it.
"""

import argparse
import asyncio
import json
import sys
from pprint import pprint

from modelcontextprotocol.client import Client
from modelcontextprotocol.client.stdio import StdioClientTransport
from modelcontextprotocol.client.sse import SSEClientTransport


async def main():
    """Main entry point for the client."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FastMCP Memory Client")
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
            "name": "memory-client",
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
        print(f"Connecting to server subprocess: {args.command}")
        transport = StdioClientTransport(args.command)
    else:
        print(f"Connecting to SSE server: {args.server}")
        transport = SSEClientTransport(args.server)
    
    try:
        await client.connect(transport)
        print("Connected successfully!")
        
        # Print server info
        info = await client.info()
        print(f"\nServer: {info['name']} v{info['version']}")
        
        # List available tools
        tools = await client.list_tools()
        print(f"\nAvailable tools ({len(tools)}):")
        for tool in tools:
            print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
        
        # Simple CLI loop
        print("\nEnter commands in the format: <tool_name> <json_args>")
        print("Example: remember {\"content\": \"This is a memory\"}")
        print("Type 'exit' to quit")
        
        while True:
            try:
                command = input("\n> ").strip()
                if command.lower() == "exit":
                    break
                
                parts = command.split(" ", 1)
                if len(parts) < 2:
                    print("Invalid command format. Use: <tool_name> <json_args>")
                    continue
                
                tool_name, args_str = parts
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    print("Invalid JSON arguments")
                    continue
                
                result = await client.execute_tool(tool_name, args)
                print("\nResult:")
                pprint(result)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        await client.disconnect()
        print("Disconnected from server")


if __name__ == "__main__":
    # Run the async main function
    if sys.version_info >= (3, 7):
        asyncio.run(main())
    else:
        # Fallback for Python 3.6
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())#!/usr/bin/env python3
"""
Client for ChromaDB memory system with Ollama embeddings.

This client uses the FastMCP client library to communicate
with the memory server.

To run:
1. Install dependencies: uv pip install -r requirements.txt
2. Start server in another terminal: python memory.py
3. Run this client: python client.py
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List

from fastmcp.client import FastMCPClient


async def main():
    """Run the memory client demo."""
    # Connect to the server
    client = FastMCPClient()
    await client.connect()
    
    print("Connected to memory server")
    
    try:
        # Display available tools
        tools = await client.list_tools()
        print(f"Available tools: {[tool['name'] for tool in tools]}\n")
        
        # Store some example memories
        print("Storing example memories...")
        
        result = await client.call_tool("remember", {
            "content": "The capital of France is Paris, a beautiful city known for the Eiffel Tower and fine cuisine."
        })
        print(f"Result: {result}")
        
        result = await client.call_tool("remember", {
            "content": "Rome is the capital of Italy and was once the center of the Roman Empire."
        })
        print(f"Result: {result}")
        
        result = await client.call_tool("remember_multiple", {
            "contents": [
                "Python is a programming language created by Guido van Rossum.",
                "The Mediterranean Sea is connected to the Atlantic Ocean.",
                "The Great Wall of China is visible from space and spans thousands of kilometers."
            ]
        })
        print(f"Result: {result}")
        
        # Check memory profile
        print("\nChecking memory profile...")
        profile = await client.call_tool("get_profile")
        print(profile)
        
        # Recall related memories
        print("\nRecalling memories about European capitals...")
        results = await client.call_tool("recall", {
            "query": "What are some famous European capital cities?",
            "limit": 3
        })
        
        print(f"Found {len(results)} related memories:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['content']} (similarity: {result['similarity']:.4f})")
        
        # Recall with different query
        print("\nRecalling memories about oceans...")
        results = await client.call_tool("recall", {
            "query": "Tell me about bodies of water and oceans",
            "limit": 2
        })
        
        print(f"Found {len(results)} related memories:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['content']} (similarity: {result['similarity']:.4f})")
        
        # Add a memory that might merge with existing one
        print("\nAdding memory that might merge with existing one...")
        result = await client.call_tool("remember", {
            "content": "Paris is the largest city in France and its capital."
        })
        print(f"Result: {result}")
        
        # Check updated profile
        print("\nChecking updated memory profile...")
        profile = await client.call_tool("get_profile")
        print(profile)
        
    finally:
        # Close the client
        await client.close()
        print("\nClient closed")


if __name__ == "__main__":
    asyncio.run(main())
