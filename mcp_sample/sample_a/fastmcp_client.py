# 客户端示例，通过 FastMCP 客户端连接到 mcp_server.py，获取工具列表，
# 并结合 LLM 实现自动工具调用和结果总结
from fastmcp import Client
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

def mcp_tool_to_openai_tool(tool):
    """
    将 MCP 工具转换为 OpenAI 工具

    Args:
        tool (_type_): MCP 工具

    Returns:
        _type_: 工具描述字典
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema
        }
    }

# 因为 main 函数中涉及到多个网络请求，并且 fastmcp 框架本身就是异步的：
# 1. 通过 Client 连接和调用远程工具服务器
# 2. 获取工具列表
# 3. 调用工具
async def main():
    # 重要：将 mcp 工具服务器地址传给 Client 类，Client 会自动用 stdio 启动 server
    async with Client("mcp_server.py") as client:
        print("[client] Connected to server")
        tools = await client.list_tools()
        print(f"[client] Tools: {tools}")
        openai_tools = [mcp_tool_to_openai_tool(t) for t in tools]
        print(f"[client] OpenAI 工具: {openai_tools}")
        print("-"*100)

        user_question = "请告诉我未来3天长沙的天气，并推荐一本关于 python 编程的书籍"
        messages = [
            SystemMessage(content="你可以调用工具来帮助用户完成任务。"),
            HumanMessage(content=user_question)
        ]
        resp = llm.invoke(messages, tools=openai_tools, tool_choice="auto")
        print(f"[client] LLM 原始返回: {resp}")
        print("-"*100)

        # 处理 tool_calls
        tool_results = []
        if hasattr(resp, "tool_calls") and resp.tool_calls:
            for tool_call in resp.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]
                print(f"[client] LLM 选择工具: {tool_name}, 参数: {args}")
                # 重要：客户端实际调用工具！！！
                tool_result = await client.call_tool(tool_name, args)
                print(f"[client] 工具 {tool_name} 返回: {tool_result[0].text}")
                tool_results.append((tool_name, tool_result[0].text))
            # 汇总所有工具结果，让 LLM 总结
            tool_results_str = "\n".join(
                [f"工具 {name} 的结果：\n{result}" for name, result in tool_results]
            )
            final_messages = [
                SystemMessage(content="你是一个善于总结结构化数据的AI助手。"),
                HumanMessage(content=f"用户问题：{user_question}\n\n{tool_results_str}")
            ]
            final = llm.invoke(final_messages)
            print("[client] LLM 最终回答：", final.content)
        else:
            print("[client] LLM 直接回答：", resp.content)

if __name__ == "__main__":
    asyncio.run(main())