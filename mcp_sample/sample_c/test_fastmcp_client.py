import asyncio
import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

class FastMCPTestClient:
    """
    测试 FastMCP 服务器的客户端
    """
    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )
        self.model = model
        self.session = None

    async def connect_to_server(self, server_script_path: str = "fastmcp_server.py"):
        """
        连接到 FastMCP 服务器
        """
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
        )
        self.stdio_client_ctx = stdio_client(server_params)
        self.stdio = await self.stdio_client_ctx.__aenter__()
        self.read_stream, self.write_stream = self.stdio
        self.session_ctx = ClientSession(self.read_stream, self.write_stream)
        self.session = await self.session_ctx.__aenter__()
        await self.session.initialize()
        
        tools_result = await self.session.list_tools()
        print("\n🔗 Connected to FastMCP server with tools:")
        for tool in tools_result.tools:
            print(f"  - {tool.name}: {tool.description}")

    async def get_mcp_tools(self):
        """获取 MCP 工具列表"""
        tools_result = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema
                }
            }
            for tool in tools_result.tools
        ]

    async def process_query(self, query: str) -> str:
        """
        处理用户查询，让 LLM 选择合适的工具
        """
        tools = await self.get_mcp_tools()
        messages = [
            SystemMessage(content="你是一个智能助手，可以使用各种工具来帮助用户。根据用户的需求选择合适的工具。"),
            HumanMessage(content=query)
        ]
        
        resp = self.llm.invoke(messages, tools=tools, tool_choice="auto")
        print(f"\n🤖 LLM Response: {resp}")
        print("-" * 80)
        
        tool_results = []
        if hasattr(resp, "tool_calls") and resp.tool_calls:
            for tool_call in resp.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]
                print(f"🔧 Calling tool: {tool_name} with args: {args}")
                
                tool_result = await self.session.call_tool(tool_name, args)
                result_text = tool_result.content[0].text
                print(f"📋 Tool {tool_name} result: {result_text}")
                tool_results.append((tool_name, result_text))
            
            # 让 LLM 总结工具结果
            tool_results_str = "\n".join(
                [f"工具 {name} 的结果：\n{result}" for name, result in tool_results]
            )
            final_messages = [
                SystemMessage(content="你是一个善于总结和解释的AI助手。请根据工具返回的结果，为用户提供清晰、有用的回答。"),
                HumanMessage(content=f"用户问题：{query}\n\n工具执行结果：\n{tool_results_str}")
            ]
            final = self.llm.invoke(final_messages)
            print(f"\n💡 Final Answer: {final.content}")
            return final.content
        else:
            print(f"💬 Direct Answer: {resp.content}")
            return resp.content

    async def cleanup(self):
        """清理资源"""
        await self.session_ctx.__aexit__(None, None, None)
        await self.stdio_client_ctx.__aexit__(None, None, None)

async def main():
    """主函数 - 演示各种功能"""
    client = FastMCPTestClient()
    
    try:
        await client.connect_to_server("fastmcp_server.py")
        
        # 测试用例
        test_queries = [
            "What is our company's vacation policy?",
            "Show me the current directory contents",
            "What's the system information?",
            "Calculate 2 + 3 * 4",
            "Create a test file with some content",
            "What's the current working directory?",
            "Show me information about the data/kb.json file"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"🧪 Test {i}: {query}")
            print(f"{'='*80}")
            
            try:
                response = await client.process_query(query)
                print(f"\n✅ Test {i} completed successfully!")
            except Exception as e:
                print(f"\n❌ Test {i} failed: {str(e)}")
            
            print("\n" + "-"*80)
    
    finally:
        await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "Attempted to exit cancel scope" in str(e):
            print("Warning: Cancel scope exit order issue (can be ignored if all results above are correct).")
        else:
            raise 