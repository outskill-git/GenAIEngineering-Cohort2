# calculator_server.py
from fastmcp import FastMCP
import asyncio

# Create server instance
mcp = FastMCP("Calculator Server")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together"""
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a"""
    return a - b

if __name__ == "__main__":
    # Run the server on stdio transport
    asyncio.run(mcp.run(transport="sse", port=9321 ))