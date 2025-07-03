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

### 1. 虚拟环境与代码执行器的集成 ###
# 导入代码执行相关工具
from autogen.code_utils import create_virtual_env
from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor

# 创建或获取一个虚拟环境，用于代码执行
venv_dir = "~/llm_env"
venv_context = create_virtual_env(venv_dir)  # 创建虚拟环境，并返回虚拟环境上下文，若虚拟环境已存在，则返回已存在的虚拟环境上下文

# 创建本地命令行代码执行器，指定虚拟环境、超时时间和工作目录
executor = LocalCommandLineCodeExecutor(
    virtual_env_context=venv_context,
    timeout=200,
    work_dir="coding",
)
#测试虚拟环境中的 Python 解释器路径
print(
    executor.execute_code_blocks(code_blocks=[CodeBlock(language="python", code="import sys; print(sys.executable)")])
)
### 1. 虚拟环境与代码执行器的集成 ###

# 定义 code_writer_agent 的系统提示词，规范其行为和输出
my_prompt = """
你是一位专业的 Python 数据分析师，负责根据用户需求编写高质量、可运行的 Python 代码。请严格按照以下要求执行任务，并确保代码详细注释。
1. **信息收集阶段：**
   * 当你需要收集任务相关的信息时，请通过以下方式之一获取所需数据：
     * 浏览或搜索网页
     * 下载/读取文件
     * 获取 API 数据
   * 在收集足够信息后，确保你已经准备好解决任务。此时，可以进行后续任务执行。
2. **任务执行阶段：**
   * 执行任务时，请使用 Python 代码并输出结果。
   * 如果任务可以分步完成，请逐步执行，清晰描述每个步骤。若没有明确的执行步骤，请先解释你的解决计划，明确哪些步骤会用代码完成，哪些步骤依赖语言能力。
   * 每段代码块都需要有明确注释，且必须在代码块中注明脚本类型（例如数据处理、API 调用、文件操作等）。
   * 请勿在一个响应中包含多个代码块，保持每个代码块的独立性，确保逻辑清晰。
3. **错误处理：**
   * 如果在执行代码时遇到错误，请修复错误并重新输出修正后的代码。如果问题无法修复，请：
     * 分析错误原因，并重新审视之前的假设。
     * 收集所需的信息，并尝试其他方法来解决问题
4. **任务完成：*
   * 当所有任务完成后，请在消息结尾回复“终止”。
"""

### 2. 智能体的创建与配置 ###
# 创建 code_writer_agent，负责生成代码，不直接执行代码
code_writer_agent = AssistantAgent(
    name="code_writer_agent",
    llm_config=llm_config,
    system_message=my_prompt, # 也可以不设置，即使用默认的系统提示词
    code_execution_config=False,
    human_input_mode="NEVER",
)

# 创建 code_executor_agent，负责执行代码块
code_executor_agent = ConversableAgent(
    name="code_executor_agent",
    llm_config=False,
    code_execution_config={"executor": executor},
    human_input_mode="ALWAYS",
    default_auto_reply="请继续，如果所有工作都准备好了，请回复'终止'"
)
### 2. 智能体的创建与配置 ###
import datetime

today = datetime.datetime.now().date()

# 构造任务消息，要求生成并展示两组数据的折线图及其移动平均线
message = f"今天是 {today}。"\
"请创建一个折线图，展示正态化后的两组虚拟数据（代表'材料A'和'材料B'）在过去60周的价格走势，并计算各自的60周移动平均线。"\
"要求如下："\
"所有代码写在一个 Python markdown 代码块中；"\
"打印归一化后的数据；"\
"图像保存为 asset_demo.png 并展示；"\
"如需安装依赖包，使用一个不带注释的 sh 代码块给出安装命令；"\
"每条消息都重复提供可执行代码块。"

### 3. 智能体间的自动对话与协作 ###
# 启动两个 agent 之间的对话，完成数据分析和可视化任务
chat_result = code_executor_agent.initiate_chat(
    code_writer_agent,
    message=message
)
### 3. 智能体间的自动对话与协作 ###