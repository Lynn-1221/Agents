from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

# Create an MCP server
mcp = FastMCP(
    name="Calculator",
    host="127.0.0.1",  # only used for SSE transport (localhost)
    port=1221,  # only used for SSE transport (set this to any port)
    stateless_http=True,
)

# Add a simple calculator tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

# Run the server
if __name__ == "__main__":
    transport = "stdio"
    if transport == "stdio":
        print("Running server with stdio transport")
        mcp.run(transport="stdio")
    elif transport == "sse":
        print("Running server with SSE transport")
        mcp.run(transport="sse")
    elif transport == "streamable-http":
        print("Running server with Streamable HTTP transport")
        mcp.run(transport="streamable-http")
    else:
        raise ValueError(f"Unknown transport: {transport}")