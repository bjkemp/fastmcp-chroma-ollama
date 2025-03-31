#!/usr/bin/env python3
"""
FastMCP server for Chroma DB with Ollama embeddings.

Provides a memory service with persistence using ChromaDB and local embeddings via Ollama.
Supports both stdio and SSE transport options.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from fastmcp import FastMCP

# Import our memory module
from memory import app, check_ollama_availability


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FastMCP Memory Server")
    parser.add_argument(
        "--transport", 
        choices=["stdio", "sse"], 
        default="stdio",
        help="Transport type (stdio or sse)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8080,
        help="Port for SSE server"
    )
    parser.add_argument(
        "--endpoint", 
        default="/sse",
        help="Endpoint for SSE server"
    )
    parser.add_argument(
        "--check-ollama", 
        action="store_true",
        help="Check Ollama availability before starting"
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Check Ollama availability if requested
    if args.check_ollama:
        available = await check_ollama_availability()
        if not available:
            print("Ollama check failed. Starting anyway, but embeddings may not work.")
    
    # Configure transport options
    if args.transport == "stdio":
        transport_config = {
            "transportType": "stdio"
        }
        print("Starting FastMCP server with stdio transport")
    else:
        transport_config = {
            "transportType": "sse",
            "sse": {
                "endpoint": args.endpoint,
                "port": args.port
            }
        }
        print(f"Starting FastMCP server with SSE transport at http://localhost:{args.port}{args.endpoint}")
    
    # Start the server
    app.start(transport_config)


if __name__ == "__main__":
    # Run the async main function
    if sys.version_info >= (3, 7):
        asyncio.run(main())
    else:
        # Fallback for Python 3.6
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())