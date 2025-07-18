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

# 创建不同类型的函数
# 1. 生成虚拟数据
def generate_virtual_data(n_points=60, mean=0, std=1, seed=None):
    """
    生成指定数量的正态分布虚拟数据。
    参数：
        n_points (int): 数据点数量，默认60。
        mean (float): 均值，默认0。
        std (float): 标准差，默认1。
        seed (int or None): 随机种子，保证结果可复现。
    返回：
        np.ndarray: 生成的数据数组。
    """
    import numpy as np
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(loc=mean, scale=std, size=n_points)


# 2. 数据归一化

def normalize_data(data):
    """
    对输入数据进行归一化处理，将数据缩放到[0, 1]区间。
    参数：
        data (array-like): 输入数据。
    返回：
        np.ndarray: 归一化后的数据。
    """
    import numpy as np
    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)


# 3. 计算移动平均线

def moving_average(data, window=5):
    """
    计算输入数据的移动平均线。
    参数：
        data (array-like): 输入数据。
        window (int): 移动平均窗口大小，默认5。
    返回：
        np.ndarray: 移动平均后的数据。
    """
    import numpy as np
    data = np.array(data)
    if window < 1:
        raise ValueError("window must be >= 1")
    if window > len(data):
        raise ValueError("window is larger than data length")
    return np.convolve(data, np.ones(window)/window, mode='valid')


# 4. 绘制折线图

def plot_lines_with_ma(data_dict, window=5, filename="asset_demo.png"):
    """
    绘制多组数据及其移动平均线的折线图，并保存为图片。
    参数：
        data_dict (dict): 键为标签，值为数据数组。
        window (int): 移动平均窗口大小。
        filename (str): 图片保存路径。
    返回：
        None
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for label, data in data_dict.items():
        plt.plot(data, label=f"{label} (normalized)")
        ma = moving_average(data, window)
        plt.plot(range(window-1, len(data)), ma, label=f"{label} {window}w MA")
    plt.legend()
    plt.title("Normalized Data with Moving Average")
    plt.xlabel("Time (weeks)")
    plt.ylabel("Normalized Price")
    plt.savefig(filename)
    plt.show()


# 5. 自动安装依赖

def install_package(package):
    """
    自动安装指定的Python依赖包。
    参数：
        package (str): 需要安装的包名。
    返回：
        None
    """
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
from autogen.code_utils import create_virtual_env
from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor

venv_dir = "~/llm_env"
venv_context = create_virtual_env(venv_dir)

### 1. 自定义函数的注册与智能体调用 ###
executor = LocalCommandLineCodeExecutor(
    virtual_env_context=venv_context,
    timeout=200,
    work_dir="coding",
    functions=[generate_virtual_data, normalize_data, moving_average, plot_lines_with_ma, install_package],
)
print(executor.execute_code_blocks(code_blocks=[CodeBlock(language="python", code="import sys; print(sys.executable)")]))
### 1. 自定义函数的注册与智能体调用 ###

code_writer_agent_system_message = """
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

### 2. 将函数签名自动注入到 LLM 的系统提示词中 ###
code_writer_agent_system_message += executor.format_functions_for_prompt()
### 2. 将函数签名自动注入到 LLM 的系统提示词中 ###

print(code_writer_agent_system_message)

code_writer_agent = AssistantAgent(
    name="code_writer_agent",
    llm_config=llm_config,
    system_message=code_writer_agent_system_message,
    code_execution_config=False,
    human_input_mode="NEVER",
)

code_executor_agent = ConversableAgent(
    name="code_executor_agent",
    llm_config=False,
    code_execution_config={"executor": executor},
    human_input_mode="ALWAYS",
    default_auto_reply="请继续，如果所有工作都准备好了，请回复'终止'"
)

import datetime

today = datetime.datetime.now().date()

message = (
    f"今天是 {today}。"
    "请创建一个折线图，展示正态化后的两组虚拟数据（代表'材料A'和'材料B'）在过去60周的价格走势，并计算各自的60周移动平均线。"
    "要求如下："
    "所有代码写在一个 Python markdown 代码块中；"
    "打印归一化后的数据；"
    "图像保存为 asset_demo.png 并展示；"
    "如需安装依赖包，使用一个不带注释的 sh 代码块给出安装命令；"
    "每条消息都重复提供可执行代码块。"
)

chat_result = code_executor_agent.initiate_chat(
    code_writer_agent,
    message=message
)
