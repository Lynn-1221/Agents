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

# 关键代码：创建一个代码执行器，并指定可调用的函数
executor = LocalCommandLineCodeExecutor(
    virtual_env_context=venv_context,
    timeout=200,
    work_dir="coding",
    functions=[generate_virtual_data, normalize_data, moving_average, plot_lines_with_ma, install_package],
)
print(executor.execute_code_blocks(code_blocks=[CodeBlock(language="python", code="import sys; print(sys.executable)")]))

code_writer_agent_system_message = """
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
当所有任务完成后，请在消息结尾回复"终止"。
"""

code_writer_agent_system_message += executor.format_functions_for_prompt()

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
