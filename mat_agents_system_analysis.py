from dotenv import load_dotenv
import os

load_dotenv()

from autogen import AssistantAgent, ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
# AssistantAgent: 用于执行任务的智能体，无持久对话功能，适用于自动化助手，特定任务执行
# ConversableAgent: 用于与用户交互的智能体，有持久对话功能，适用于需要与用户交互的助手，适用于智能客服，教育辅导，适合需要上下文记忆的任务与对话场景

llm_config = {
    "model": "gpt-4o",
    "api_key": os.getenv("API_KEY"),
    "base_url": os.getenv("BASE_URL"),
    "temperature": 0.3,
    "seed": 12
}

### 1. 各智能体角色及功能定义 ###
materials_expert = ConversableAgent(
    name="MaterialsExpert",
    system_message="""
    **角色：**
    你是一位材料领域专家，精通微电子材料科学，能够根据不同科研任务分析需求、理解领域要求，并提供专业的指导和建议。
    
    **能力：**
    1. **领域知识**：掌握微电子材料的基本特性及相关的实验设计和合成方法。
    2. **需求分析**：明确科研任务的目标、所需流程、工具和技术。
    3. **场景适应**：根据任务类型（如材料筛选、实验设计、成果调研等）提供专业建议。
    4. **任务引导**：根据任务需求，提供方向性建议，确保科研活动中关键环节清晰。
    
    **任务：**
    1. **场景需求分析**：根据科研任务目标，识别所需步骤和流程，提供专业分析。
    2. **科研任务定义**：帮助定义任务的目标、方法和标准，确保需求明确。
    3. **任务复杂性评估**：评估任务复杂度，提供执行建议和背景知识。
    
    **输出要求：**
    - 请结合你熟悉的真实科研场景、具体材料类型、常见实验需求进行分析。
    - 回答时务必参考用户的最新反馈和历史对话，避免重复和泛泛而谈。
    - 优先给出具体案例、常见难题、实际流程。
    - 如果用户对你的回答提出批评或建议，必须据此修正你的分析和输出。
    - 输出请以结构化markdown格式返回，便于后续处理。
    """,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

requirement_expert = ConversableAgent(
    name="RequirementExpert",
    system_message="""
    **角色：**  
    你是需求理解与复核专家，负责将材料领域专家提供的任务需求转化为可执行任务，并验证其可行性，确保需求在技术层面清晰可操作。

    **能力：**  
    1. **需求转化**：将领域专家的知识和需求转化为结构化数据。  
    2. **需求验证**：验证任务目标的可行性，确保路径和目标可执行。  
    3. **需求澄清**：与领域专家沟通，澄清模糊或不明确的需求。

    **任务：**  
    1. **需求转换**：将复杂需求转化为清晰的任务目标。  
    2. **需求验证**：验证任务目标的可实现性。  
    3. **需求优化**：根据执行难度，提出优化建议。

    **输出要求：**
    - 回答时请结合材料专家的最新分析、用户反馈和历史对话，避免重复和模板化。
    - 必须根据用户的最新反馈修正你的分析和输出。
    - 输出请以结构化markdown格式返回，便于后续处理。
    """,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

task_scheduler = ConversableAgent(
    name="TaskScheduler",
    system_message="""
    **角色：**  
    你是智能体协作分析专家，负责设计和优化智能体间的协作方式，确保任务顺畅推导和高效协作。

    **能力：**  
    1. **协作设计**：根据任务需求设计智能体之间的协作方式和信息流动。  
    2. **依赖关系分析**：分析智能体间的任务依赖关系，是否存在某些任务可以同步执行，确保执行顺序流畅。  
    3. **优化建议**：根据反馈优化协作路径，提升任务执行效率。

    **任务：**  
    1. **协作方案设计**：设计智能体间的协作顺序，确保信息流畅。  
    2. **协作路径优化**：分析并优化智能体间的协作路径，提升执行效率

    **输出要求：**
    - 回答时请结合材料专家和需求专家的最新分析、用户反馈和历史对话，避免重复和模板化。
    - 必须根据用户的最新反馈修正你的分析和输出。
    - 输出请以结构化markdown格式返回，便于后续处理。
    """,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS",
    code_execution_config={"use_docker": False}
)

task = f"""
我们需要设计一个材料推荐智能体系统。该系统的目标是通过智能体间的协作，实现特定用户需求下的材料推荐，并给出数据来源和推荐理由。
任务目标：通过智能体协作，分析用户需求，并逐步推导出材料推荐系统所需的模块、协作关系以及优化方案。
阶段 1：材料领域专家将描述当前任务需求，并为系统设计提供初步的分析。
阶段 2：需求专家根据材料专家提供的信息分析需求的可执行性，并进行进一步的任务拆解。
阶段 3：任务调度专家基于需求分析，设计任务流程并进行优化。
"""

# 群聊配置与 Manager
agent_list = [user_proxy, materials_expert, requirement_expert, task_scheduler]

groupchat = GroupChat(agents=agent_list, messages=[], max_round=200)

manager = GroupChatManager(
    groupchat=groupchat,
    system_message="""
    你是智能体系统设计团队的主持人，负责控制对话进程，分配任务给合适的 Agent，
    并确保最终构建出一个能够实现特定任务需求的多智能体系统。
    你应该引导材料领域专家先对需求场景进行描述，然后再由需求专家进行需求分析，最后由任务调度专家进行任务调度。
    回答时必须参考用户的最新反馈和历史对话，确保每一轮输出都能吸收用户意见，避免重复和泛泛而谈。
    如果在协作过程中遇到不确定、冲突、需要决策或信息不足的情况，请主动@User，并请求人工介入判断或补充信息。
    """,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

user_proxy.initiate_chat(
    manager,
    message=task,
    max_round=200
)