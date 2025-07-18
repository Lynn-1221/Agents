import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_fastmcp_server():
    """直接测试 FastMCP 服务器的各种工具"""
    
    print("🚀 Testing FastMCP Server...")
    
    # 连接到服务器
    server_params = StdioServerParameters(
        command="python",
        args=["fastmcp_server.py"],
    )
    
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            # 获取可用工具
            tools_result = await session.list_tools()
            print(f"\n📋 Available tools ({len(tools_result.tools)}):")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")
            
            print("\n" + "="*80)
            
            # 测试各种工具
            test_cases = [
                ("get_knowledge_base", {}),
                ("search_knowledge_base", {"query": "vacation"}),
                ("get_system_info", {}),
                ("get_current_working_directory", {}),
                ("list_directory_contents", {"path": "."}),
                ("calculate_math", {"expression": "2 + 3 * 4"}),
                ("calculate_math", {"expression": "sqrt(16) + pi"}),
                ("get_file_info", {"file_path": "data/kb.json"}),
                ("create_file", {"file_path": "test_output.txt", "content": "Hello from FastMCP!\nThis is a test file created by the MCP server."}),
                ("read_file_content", {"file_path": "test_output.txt", "max_lines": 10}),
            ]
            
            for i, (tool_name, args) in enumerate(test_cases, 1):
                print(f"\n🧪 Test {i}: {tool_name}")
                print(f"📝 Args: {args}")
                print("-" * 60)
                
                try:
                    result = await session.call_tool(tool_name, args)
                    print(f"✅ Result: {result.content[0].text}")
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                
                print("-" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(test_fastmcp_server())
    except Exception as e:
        print(f"❌ Test failed: {str(e)}") 