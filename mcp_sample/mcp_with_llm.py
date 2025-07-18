import asyncio  # 异步编程库
import os  # 操作系统相关
from dotenv import load_dotenv  # 加载 .env 环境变量
from mcp import ClientSession, StdioServerParameters  # MCP 客户端相关类
from mcp.client.stdio import stdio_client  # MCP stdio 客户端
from langchain_openai import ChatOpenAI  # LangChain OpenAI LLM 封装
from langchain_core.messages import HumanMessage, SystemMessage  # 消息格式

load_dotenv()

class MCPOpenAIClient:
    """
    MCP 客户端，结合 LLM 实现自动工具调用和结果总结
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

    async def connect_to_server(self, server_script_path: str = "server.py"):
        """
        连接 MCP 工具服务器，初始化会话
        """
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
        )
        self.stdio_client_ctx = stdio_client(server_params) # 创建 MCP stdio 客户端的异步上下文管理器
        self.stdio = await self.stdio_client_ctx.__aenter__() # 异步进入上下文，建立与 MCP 服务器的连接，获取读写流
        # async with some_async_context_manager() as resource: 等同于
        # resource = await some_async_context_manager.__aenter__()
        self.read_stream, self.write_stream = self.stdio # 拆包得到读流和写流
        self.session_ctx = ClientSession(self.read_stream, self.write_stream) # 创建 MCP 会话的异步上下文管理器
        self.session = await self.session_ctx.__aenter__() # 异步进入上下文
        await self.session.initialize()
        tools_result = await self.session.list_tools()
        print("\nConnected to server with tools:")
        for tool in tools_result.tools:
            print(f"  - {tool.name}: {tool.description}")

    async def get_mcp_tools(self):
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
        主流程：
        1. 让 LLM 选择工具
        2. 客户端实际调用工具
        3. 汇总工具结果让 LLM 总结
        """
        tools = await self.get_mcp_tools()  # 获取工具列表
        messages = [
            SystemMessage(content="你可以调用工具来帮助用户完成任务。"),
            HumanMessage(content=query)
        ]
        resp = self.llm.invoke(messages, tools=tools, tool_choice="auto")
        print(f"[client] LLM 原始返回: {resp}")
        print("-"*100)
        tool_results = []
        if hasattr(resp, "tool_calls") and resp.tool_calls:
            for tool_call in resp.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]
                print(f"[client] LLM 选择工具: {tool_name}, 参数: {args}")
                tool_result = await self.session.call_tool(tool_name, args)
                print(f"[client] 工具 {tool_name} 返回: {tool_result.content[0].text}")
                tool_results.append((tool_name, tool_result.content[0].text))
            tool_results_str = "\n".join(
                [f"工具 {name} 的结果：\n{result}" for name, result in tool_results]
            )
            final_messages = [
                SystemMessage(content="你是一个善于总结结构化数据的AI助手。"),
                HumanMessage(content=f"用户问题：{query}\n\n{tool_results_str}")
            ]
            final = self.llm.invoke(final_messages)
            print("[client] LLM 最终回答：", final.content)
            return final.content
        else:
            print("[client] LLM 直接回答：", resp.content)
            return resp.content

    async def cleanup(self):
        # 关闭资源
        await self.session_ctx.__aexit__(None, None, None) # 退出 async with 上下文
        await self.stdio_client_ctx.__aexit__(None, None, None)

async def main():
    client = MCPOpenAIClient()
    try:
        await client.connect_to_server("server.py")
        query = "What is our company's vacation policy?"
        print(f"\nQuery: {query}")
        response = await client.process_query(query)
        print(f"\nResponse: {response}")
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