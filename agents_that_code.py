from autogen import AssistantAgent, ConversableAgent
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

llm_config = {
    "model": "gpt-4o",
    "api_key": api_key,
    "base_url": base_url,
    "temperature": 0.3,
    "cache_seed": 12
}

from autogen.code_utils import create_virtual_env
from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor

# 在已有的虚拟环境中执行代码
venv_dir = "~/llm_env"
venv_context = create_virtual_env(venv_dir)  # 创建虚拟环境，并返回虚拟环境上下文，若虚拟环境已存在，则返回已存在的虚拟环境上下文

executor = LocalCommandLineCodeExecutor(
    virtual_env_context=venv_context,
    timeout=200,
    work_dir="coding",
)
print(
    executor.execute_code_blocks(code_blocks=[CodeBlock(language="python", code="import sys; print(sys.executable)")])
)

my_prompt = """
你是一位专业的Python数据分析师。
请严格按照用户需求生成高质量、可运行的Python代码，
并确保代码有详细注释。
1. 当你需要收集信息时，使用代码输出你需要的信息，例如：浏览或搜索网页、下载/读取文件、获取API数据等。
在输出足够的信息，并且任务已经准备好好，你就可以自己解决该任务了。
2. 当你需要用代码执行某些任务时，使用代码执行任务并输出结果。
如果需求，请逐步完成任务，如果没有提供计划，请先解释你的计划，明确哪些步骤使用代码，哪些步骤使用语言技能。
使用代码时，必须在代码快中注明脚本类型。
请勿在一个响应中包含多个代码快。
如果结果表明存在错误，请修复错误并重新输出代码。如果无法修复，请分析问题，并重新审视你的假设，收集所需信息，并尝试其他方法。
当所有任务完成后，请在消息结尾回复“终止”。
"""

code_writer_agent = AssistantAgent(
    name="code_writer_agent",
    llm_config=llm_config,
    system_message=my_prompt, # 也可以不设置，即使用默认的系统提示词
    code_execution_config=False,
    human_input_mode="NEVER",
)

print(code_writer_agent.system_message)

code_writer_agent_system_message = code_writer_agent.system_message
print(code_writer_agent_system_message)


import pdb; pdb.set_trace()

code_executor_agent = ConversableAgent(
    name="code_executor_agent",
    llm_config=False,
    code_execution_config={"executor": executor},
    human_input_mode="ALWAYS",
    default_auto_reply="请继续，如果所有工作都准备好了，请回复‘终止’"
)

import datetime

today = datetime.datetime.now().date()

message = f"今天是 {today}。"\
"请创建一个折线图，展示正态化后的两组虚拟数据（代表“材料A”和“材料B”）在过去60周的价格走势，并计算各自的60周移动平均线。"\
"要求如下："\
"所有代码写在一个 Python markdown 代码块中；"\
"打印归一化后的数据；"\
"图像保存为 asset_demo.png 并展示；"\
"如需安装依赖包，使用一个不带注释的 sh 代码块给出安装命令；"\
"每条消息都重复提供可执行代码块。"

chat_result = code_executor_agent.initiate_chat(
    code_writer_agent,
    message=message
)

print(chat_result)