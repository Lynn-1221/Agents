# AutoGen 多 Agent 协同配置：材料大模型评测体系优化

from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

llm_config = {
    "model": "gpt-4o",
    "api_key": api_key,
    "base_url": base_url,
    "temperature": 0.5,
    "cache_seed": 12   # 类似的问题，不会再次请求，而是去到缓存中查询，并返回结果
}

# 材料专家 Agent：判断任务是否贴合微电子科研流程
materials_expert = AssistantAgent(
    name="MaterialsExpert",
    system_message="""
    你是一位精通微电子材料科学的资深科研专家，擅长射频、光电、功率、神经拟态器件及工艺材料领域相关研究。
    你的任务是从真实科研流程出发，判断每类评测任务是否具备科研实用性，
    包括是否贴合科研痛点、是否具有实验转化价值、是否存在已知的科研需求。
    你还需提出新的高价值任务建议，并指出当前评测设计中脱离科研实践的内容。
    请结合实验情境、科研动机和科研过程具体分析，确保任务设计具有领域针对性和落地性。""",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# 架构师 Agent：主导能力层级与任务结构设计
framework_architect = AssistantAgent(
    name="FrameworkArchitect",
    system_message="""
    你是大模型评测体系的首席架构专家，负责构建清晰的三层能力评估框架（知识层 / 认知层 / 任务层），
    并提出评估目标与各类任务类型的映射关系。
    你需定义每层能力所覆盖的关键评估维度，并制定合理的任务分类结构。
    对于每个任务类型，你需要定义其输入形式、输出格式、核心评估目标及其所在能力层级。
    你的产出将作为后续评分标准与数据设计的基础，请确保结构系统、逻辑闭环、易于拓展。""",
    llm_config=llm_config,
    human_input_mode="NEVER"
    
)

# 评分专家 Agent：制定任务评分维度与评估机制
scoring_specialist = AssistantAgent(
    name="ScoringSpecialist",
    system_message="""
    你是人工智能模型评估专家，专长于复杂任务评分体系设计与多维度评估机制构建。
    你需要为每类评测任务设计清晰的评分机制，包括：评分维度定义、每个维度的评分标准、边界条件（如模糊/部分正确/逻辑跳跃）与扣分逻辑。
    你需判断该任务是否可通过自动评分实现，若否则提出可落地的半自动/专家协同评分机制方案。
    评分机制应具备可操作性、可区分性与复现性。""",
    llm_config=llm_config,
    human_input_mode='NEVER'
)

# 4️⃣ 数据策划 Agent：判断任务数据可获取性与覆盖质量
data_curator = AssistantAgent(
    name="DataCurator",
    system_message="""
    你是一名大模型测试数据设计专家，擅长评估任务的数据可行性与样本设计策略。
    你需要判断每类任务是否具备构造高质量测试集的可行性，包括：数据来源是否可获得、样本是否可自动标注、数据是否具有代表性与多样性。
    你还需评估现有任务的样本分布是否合理，指出样本缺口，并提出数据采集/重构建议。""",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# 5️⃣ 协同记录员 Agent：跟踪共识与生成结构化输出
facilitator_agent = AssistantAgent(
    name="FacilitatorAgent",
    system_message="""
    你是评测体系优化过程的对话协调员与结构化输出生成器。
    你的任务是记录每轮对话中的专家共识、任务建议、框架更新与评分机制等关键内容，并将其整理为结构化的输出格式（如 Markdown 表格或 YAML 模板），
    以供后续存档或平台使用。
    你还需在必要时协调不同专家之间的冲突意见，并引导团队聚焦目标推进讨论。""",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# 6️⃣ 用户代理（半自动）
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS",  # 避免频繁人工干预，仅首次初始化
    max_consecutive_auto_reply=10,
    code_execution_config={"use_docker": False}
)

# 群聊配置与 Manager
agent_list = [user_proxy, framework_architect, materials_expert, scoring_specialist, data_curator, facilitator_agent]

groupchat = GroupChat(agents=agent_list, messages=[], max_round=50)

manager = GroupChatManager(
    groupchat=groupchat,
    system_message="""
    你是评测体系设计团队的主持人，负责控制对话进程，分配任务给合适的 Agent，
    并确保最终构建出一个具有三层能力结构、结构合理、贴合科研、具备评分标准与数据可行性的材料大模型评测体系。
    你应优先引导架构师提出框架初稿，再由其他专家依次审核与补充，必要时协调冲突并召集 Facilitator 输出结构化记录。
    """,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# 启动对话（初始化任务流程）
user_proxy.initiate_chat(
    manager,
    message="""
    请 FrameworkArchitect 提出材料大模型评测的三层能力评估框架草案，并初步列出每层建议的典型任务类型。
    之后请依次由 MaterialsExpert、ScoringSpecialist 和 DataCurator 分别评估任务的科研贴合度、评分可行性与数据可获取性，
    最后由 FacilitatorAgent 汇总为结构化输出。""",
    max_round=15
)
