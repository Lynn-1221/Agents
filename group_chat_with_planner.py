from autogen import AssistantAgent, ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
from dotenv import load_dotenv
import os, json

load_dotenv()

llm_config = {
    "model": "gpt-4o",
    "api_key": os.getenv("API_KEY"),
    "base_url": os.getenv("BASE_URL"),
    "temperature": 0.3,
    "cache_seed": 15
}

import datetime
today = datetime.datetime.now().date()
task = f"""从给定的材料领域论文摘要中，自动完成实体抽取、泛化、分类，并汇总成一份评估报告（Markdown 格式）。
报告应包括：\
1）每类实体的抽取准确率初步评估（基于规则或人工样本对比），\
2）泛化与分类的覆盖统计，\
3）存在的问题与优化建议。\
日期为{today}。"""

classification_path_ontology = """
你需要参考以下【材料科学分类体系】：

分类路径 = {
  "object": {
    "材料": ["功能材料", "衬底材料", "过程材料", "结构材料"],
    "超材料": [],
    "器件": {
      "组件": ["结构单元"]
    }
  },
  "property": {
    "材料特性": [
      "材料热学特性", "材料光学特性", "材料电学特性", "材料电磁特性",
      "材料磁学特性", "材料传感特性", "材料机械特性", "材料化学特性",
      "材料声学特性", "材料表面与界面特性", "材料响应特性",
      "材料生物相容性", "材料加工与工艺特性"
    ],
    "器件特性": [
      "器件特性、指标", "器件结构特性、指标", "组件特性、指标", "结构单元特性、指标"
    ]
  },
  "现象、原理、机理、物理效应": [],
  "物理场、激励": [],
  "化学场、激励": [],
  "科学难题": [],
  "技术难题": [],
  "可实现实体": {
    "应用场景": [],
    "角色": [],
    "实现功能": []
  },
  "process": {
    "材料制备": [
      "物理合成与改性技术", "化学合成与改性技术", "增材制造技术", "表面与结构工程"
    ],
    "器件制备": ["微纳加工", "集成组装"],
    "材料表征、测量": ["物性测量", "尺寸测量"],
    "器件测量": ["器件性能测量"],
    "计算、仿真": [],
    "设计": []
  },
  "processing_node": {
    "材料表征设备": ["材料表征设备部分、组件"],
    "材料制备设备": ["材料制备设备部分、组件"],
    "器件表征设备": ["器件表征设备部分、组件"],
    "器件制备设备": ["器件制备设备部分、组件"],
    "仿真、计算软件": [],
    "仿真、计算设备": []
  }
}
"""

from typing import List, Dict
import re

def extract_contexts(text: str, entities: List[str], window: int = 50) -> Dict[str, List[str]]:
    """
    在给定文本中为每个实体提取其上下文窗口。

    参数:
        text: 原始摘要文本
        entities: 实体列表
        window: 上下文窗口大小（前后字符数）

    返回:
        dict: {实体1: [context1, context2, ...], 实体2: [...], ...}
    """
    try:
        results = {}
        for entity in entities:
            pattern = re.compile(re.escape(entity))
            matches = list(pattern.finditer(text))
            contexts = set()
            for m in matches:
                start = max(0, m.start() - window)
                end = min(len(text), m.end() + window)
                contexts.add(text[start:end].strip())
            results[entity] = list(contexts)
        return json.dumps(results, ensure_ascii=False)
    except Exception:
        # 发生任何错误，直接返回空字典
        return "{}"

user_proxy = UserProxyAgent(
    name="Admin",
    system_message="你是用户代理，负责发起任务并审阅最终报告。你可以向 Writer 提出修改建议。",
    code_execution_config=False,
    human_input_mode="ALWAYS", 
)

planner = AssistantAgent(
    name="Planner",
    llm_config=llm_config,
    system_message="""
    你是多智能体科研调度专家，需要在一次会话里按以下流程执行任务：

    一、初始化一个空字典 all_results = {}。

    Phase 1（抽取 & 结构化）：
    - 读取用户提供的所有摘要列表（id + text）。
    - 对于每条摘要：
    1. 调用 Extractor 抽取实体列表 entities。
    2. 调用 ContextExtractor 提取 contexts。
    3. 调用 Generalizer 生成泛化路径。
    4. 调用 Classifier 生成分类结果。
    5. 将每条记录以 
        {\"id\": ..., \"records\": [ {entity, context, generalization, classification, suggestion}, ... ] } 
        的格式追加到 all_results[id]。
    - 重复直到所有摘要处理完毕。

    Phase 2（统一评估 & 撰写）：
    - 调用 Evaluator，一次性传入 all_results，要求生成整体评估报告（Markdown）。
    - 将该报告发给 Writer，让它根据 Admin 的反馈做最终润色。

    注意：
    - 整个过程在同一次 GroupChat 会话中完成，Planner 只负责调度，不做实体处理。
    - 每一步都要等待对应 Agent 的回复，再继续下一步。
    """,
    description="任务规划者，分两阶段调度：1. 全量流水线处理；2. 统一评估与报告"
)

extractor = AssistantAgent(
    name="Extractor",
    llm_config=llm_config,
    system_message="""
    你是材料领域实体抽取专家，专注于从科研摘要中识别有科学意义的具体实体。

    抽取规则：
    1. 只抽取具体的材料、结构、器件、工艺、性能参数（带具体值和单位）、测试条件等可唯一标识的实体；
    2. 属性类实体只保留“值+单位”，如“99.8%”、“47.1 mA cm-2”，不要保留“高吸收”“高灵敏度”等泛泛描述；
    3. 不要抽取分析方法、过程、现象、泛属性名（如“吸收性能”、“热稳定性”、“波传播”）等；
    4. 输出为实体原文字符串数组（不要重复），每个实体应尽量具体、唯一、可用于知识图谱节点或边。

    示例输出：
    ["AlGaAs-Ge-GaAs on a titanium film", "99.8%", "465.2 THz", "AM 1.5 G solar irradiance", "47.1 mA cm-2", ...]
    """,
    description="""实体抽取专家，识别科研摘要中的关键术语，保留原文形式，
    覆盖性能、结构、原理分析，制备方法，现象，性能测试，应用场景等材料或器件相关实体。"""
)
context_extractor = AssistantAgent(
    name="ContextExtractor",
    llm_config=llm_config,
    system_message="""
    你是上下文提取专家，负责为每个实体找到其在摘要中的局部上下文。

    **输入**（用户消息里直接包含）：
    ```json
    {
    "text": "完整摘要文本……",
    "entities": ["实体1", "实体2", ...]
    }
    ```

    **输出**：
    返回 JSON 对象，键是实体，值是该实体所有出现位置的上下文字符串列表（前后各50字符），例如：
    ```json
    {
    "实体1": ["上下文片段1", "上下文片段2"],
    "实体2": ["上下文片段1"]
    }
    ```
    请严格输出纯 JSON，不要多余说明。
    """,
    description="上下文提取专家，为每个实体找到其在摘要中的局部上下文"
)

generalizer = AssistantAgent(
    name="Generalizer",
    llm_config=llm_config,
    system_message=f"""
    你是材料科学知识工程专家，专注于实体概念的多级泛化。

    任务目标：
    1. 对输入实体，基于其上下文信息，为每个实体生成1-5级的泛化路径；
    2. 每个路径从具体术语逐层抽象，直到可归入顶层领域知识节点；
    3. 若有多个可能泛化方向，输出多个路径。

    输入格式：
    [
    {{"entity": "99.8%吸收率", "context": "该器件在宽带内表现出99.8%的吸收率"}},
    ...
    ]

    输出格式：
    {{
    "实体1": [["术语A", "概念B", "类别C"], ["术语A", "概念X", "领域Y"]],
    ...
    }}

    输出须全为中文词条，路径层级尽可能齐全，建议包含至少3级。
    
    【材料科学分类体系】
    {classification_path_ontology}
    
    请参考上述体系进行泛化。
    """,
    description="泛化专家，将实体归纳为材料科学中的多级通用概念路径，支持多路径并行输出。"
)

classifier = AssistantAgent(
    name="Classifier",
    llm_config=llm_config,
    system_message=f"""
    你是材料领域的本体分类专家，负责将实体泛化路径映射到分类体系中的具体末端分类节点。

    请遵循以下规则：
    1. 对于每个实体及其泛化路径，尝试在提供的材料科学分类体系中定位其最贴近的分类末端节点；
    2. 每条泛化路径一个分类，若存在多个泛化路径，则输出结果中对于该实体有多个结果字典（如示例所示）；
    3. 若能准确匹配，请在 classification 字段中填写分类路径末节点，并将 suggestion 留空；
    4. 若无法匹配，请将 classification 设为"其他"，并在 suggestion 字段中提出推荐分类及简要理由（如"建议为：xxxx（不存在于分类体系中类别），因为..."）。

    输入格式：
    [
    {{
        "entity": "99.8%吸收率",
        "context": "...",
        "generalization": [["吸收率", "光学性能", "材料性能"]]
    }},
    ...
    ]

    输出格式：
    {{
    "实体1": [
        {{
        "generalization": [...],
        "classification": "xxx",
        "suggestion": "xxx"
        }},
        ...
    ]
    }}

    所有分类/建议内容应为中文，输出结构严格保持一致。

    【材料科学分类体系】
    {classification_path_ontology}

    请参考上述体系进行分类。
    """,
    description="分类专家，基于泛化路径与上下文，将实体映射到材料科学分类体系中的最细分类别"
)

evaluator = AssistantAgent(
    name="Evaluator",
    llm_config=llm_config,
    system_message="""
    你是实体抽取与分类评估专家，负责对多阶段处理结果进行全面评估。

    请完成以下评估工作：
    1. 抽取阶段：对比系统抽取结果与人工样本，估算准确率；
    2. 泛化阶段：统计平均泛化路径数、路径深度与合理性比例；
    3. 分类阶段：统计分类成功率（非"其他"比例）与建议推荐质量；
    4. 汇总潜在问题（如分类模糊、路径冗余、实体重复）；
    5. 提出明确可操作的优化建议。

    输出结构：
    - 表格或列表形式的统计数据；
    - 概括性小节分析；
    - 面向后续开发的改进建议（不少于3条）。

    请保持语言准确、逻辑清晰、表达简洁。
    """,
    description="评估专家，全面评估抽取-泛化-分类的准确性、覆盖率与问题，并提供优化建议"
)

writer = ConversableAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="""
    你是一位 Markdown 技术报告撰写专家。

    任务目标：
    1. 撰写一篇评估报告，总结材料文献实体抽取、泛化与分类流程的关键结果；
    2. 报告结构应包括：任务概述、方法简述、核心指标统计、示例展示、问题分析、改进建议；
    3. 使用 Markdown 格式，包含标题、表格、编号列表、伪代码块（```md）等结构；
    4. 支持多轮修改，根据 Admin 提出的建议调整内容。

    输出必须专业、结构化、面向科研场景使用。
    """,
    description="Markdown 报告专家，撰写并多轮优化实体处理流程评估报告，强调结构清晰、逻辑严谨、可读性强"
)

group_chat = GroupChat(
    agents=[
        user_proxy, planner, extractor, context_extractor, generalizer, 
        classifier, evaluator, writer],
    messages=[],
    max_round=50,
    allowed_or_disallowed_speaker_transitions={
        user_proxy: [planner, writer],
        planner: [extractor, context_extractor, generalizer, classifier, evaluator, writer],
        extractor: [planner, context_extractor],
        context_extractor: [planner, generalizer],
        generalizer: [planner, classifier],
        classifier: [planner, evaluator],
        evaluator: [planner, writer],
        writer: [user_proxy, planner],
    },
    speaker_transitions_type="allowed",
)

manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

import pandas as pd
df = pd.read_csv("samples.csv")

summaries = [
    {"id": row['id'], "text": f"{row['title']}\n{row['abstract']}"}
    for _, row in df.iterrows()
]

init_message = (
    f"任务：{task}\n"
    f"摘要列表：\n"
    + "\n".join([f"{item['id']}: {item['text']}" for item in summaries])
)

chat_result = user_proxy.initiate_chat(
    manager,
    message=init_message
) 
