#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main_multisource_autogen.py

import os
import json
import asyncio
from dotenv import load_dotenv
import re

# --- AutoGen and Agents ---
from autogen import AssistantAgent, ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
from autogen_core.tools import FunctionTool

# --- LangChain for Retrieval ---
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# --- Code Execution Tools ---
from autogen.code_utils import create_virtual_env
from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor

# 1. Load environment variables
load_dotenv()
MP_API_KEY = os.getenv("MP_API_KEY")
API_KEY = os.getenv("API_KEY")      # Materials Project API key if needed
BASE_URL = os.getenv("BASE_URL")    # Base URL if needed

# 2. LLM configuration
llm_config = {
    "model": "gpt-4o",
    "api_key": API_KEY,
    "base_url": BASE_URL,
    "temperature": 0.3,
    "cache_seed": 12,
}

# 3. Set up code execution environment for final step
venv_dir = os.path.expanduser("~/llm_env")
venv_context = create_virtual_env(venv_dir)
executor = LocalCommandLineCodeExecutor(
    virtual_env_context=venv_context,
    timeout=200,
    work_dir="coding"
)

# 4. Knowledge retrieval setup (Chroma index built earlier in './mp_index')
doc_store = Chroma(persist_directory="./mp_index_doc",
                  embedding_function=OpenAIEmbeddings(api_key=API_KEY,base_url=BASE_URL, model="text-embedding-3-small"))
param_store = Chroma(persist_directory="./mp_index_param",
                    embedding_function=OpenAIEmbeddings(api_key=API_KEY,base_url=BASE_URL, model="text-embedding-3-small"))
return_store = Chroma(persist_directory="./mp_index_return",
                     embedding_function=OpenAIEmbeddings(api_key=API_KEY,base_url=BASE_URL, model="text-embedding-3-small"))

# 5. Define retrieval tool for MP index

def retrieve_snippets(intent: dict) -> list:
    """
    根据结构化意图，从本地 MP 知识库检索相关示例代码。
    intent: dict, 包含字段如 'target', 'filters', 'fields' 等。
    分字段分别检索 docstring、params、returns，合并去重。
    返回最相关的若干代码片段列表。
    """
    # 1. 语义检索 docstring
    query = intent.get("query", "")
    doc_results = []
    if query:
        doc_results = doc_store.similarity_search(query, k=3)
    # 2. 参数检索
    params = intent.get("filters", {}).keys()
    param_results = []
    for p in params:
        if p:
            param_results.extend(param_store.similarity_search(p, k=2))
    # 3. 返回字段检索
    fields = intent.get("fields", [])
    return_results = []
    for f in fields:
        if f:
            return_results.extend(return_store.similarity_search(f, k=2))
    # 4. 合并去重
    all_results = {d.page_content for d in doc_results + param_results + return_results}
    return list(all_results)

retriever_tool = FunctionTool(
    func=retrieve_snippets,
    name="mp_retriever",
    description="检索 Materials Project API 调用示例"
)

# 6. System messages for each agent
requirement_clarification = """
你是需求澄清专家，负责与用户交互，确保需求准确、可操作。

【输出要求】
- 只输出澄清后的结构化需求文档，格式为 JSON。
- 不要输出任何多余的自然语言解释或确认语句。
- JSON 字段包括：目标（target）、条件（filters）、返回字段（fields）。

【示例输出】
{
  "target": "materials",
  "filters": {
    "elements": ["Si", "O"],
    "band_gap": { "min": 1.0 }
  },
  "fields": ["material_id", "band_gap"]
}
"""

query_intent_conversion = """
你是意图解析专家，负责将结构化需求文档转换为通用的查询意图。

【输出要求】
- 只输出结构化查询意图，格式为 JSON。
- 不要输出任何多余的自然语言解释或确认语句。
- JSON 字段包括：datasource、target、filters、fields。

【示例输出】
{
  "datasource": "<to_be_selected>",
  "target": "materials",
  "filters": {
    "elements": ["Si", "O"],
    "band_gap": { "min": 1.0 }
  },
  "fields": ["material_id", "band_gap"]
}
"""

data_source_selection = """
你是数据源选择智能体，负责根据查询意图选择最合适的数据源。

【输出要求】
- 只输出数据源标识字符串，如 "mp"、"local"、"cif"。
- 不要输出任何多余的自然语言解释或确认语句。

【示例输出】
"mp"
"""

query_generation = """
**角色：**
你是查询语言生成智能体，负责根据查询意图生成具体的API请求代码片段。

**任务：**
- 接收查询意图 JSON 和检索到的示例，生成可执行的 Pytho 代码。
- 代码片段应赋值输出到变量 `output`，并遵循最佳实践。
"""

code_writer_prompt = """
你是一位专业的 Python 数据分析师，负责生成高质量、可运行的代码。

请严格按照以下要求：
1. 输出结果赋值给 `output` 变量。
2. 代码必须包含注释，解释关键步骤。

示例格式：
```python
from mp_api.client import MPRester
with MPRester(API_KEY) as mpr:
    docs = mpr.materials.summary.search(...)
    output = [doc.__dict__ for doc in docs]
```"""

# 7. Instantiate agents

requirement_agent = AssistantAgent(
    name="RequirementClarifier",
    llm_config=llm_config,
    system_message=requirement_clarification,
    human_input_mode="ALWAYS"
)
intent_agent = AssistantAgent(
    name="IntentParser",
    llm_config=llm_config,
    system_message=query_intent_conversion,
    human_input_mode="NEVER"
)
source_agent = AssistantAgent(
    name="SourceSelector",
    llm_config=llm_config,
    system_message=data_source_selection,
    human_input_mode="NEVER"
)
retriever_agent = AssistantAgent(
    name="MPRetriever",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

code_writer_agent = AssistantAgent(
    name="CodeWriter",
    llm_config=llm_config,
    system_message=code_writer_prompt,
    code_execution_config={"use_docker": False},
    human_input_mode="NEVER"
)
# code_executor_agent = ConversableAgent(
#     name="CodeExecutor",
#     llm_config=False,
#     code_execution_config={"executor": executor},
#     human_input_mode="NEVER"
# )

# 8. User proxy for initiating chat
user_agent = UserProxyAgent(name="User", human_input_mode="NEVER", code_execution_config={"use_docker": False})


# 9. Assemble GroupChat
# agents = [
#     requirement_agent,
#     intent_agent,
#     source_agent,
#     retriever_agent,
#     code_writer_agent,
#     # code_executor_agent
# ]
# groupchat = GroupChat(agents=agents, messages=[], max_round=200)

# manager = GroupChatManager(
#     groupchat=groupchat,
#     system_message="""
#     让我们开始材料筛选相关工作吧
#     """,
#     llm_config=llm_config,
#     human_input_mode="NEVER"
# )

task = f"""
请帮我查询 Si 和 O 材料的带隙 band_gap 大于 1eV，并返回 material_id 和 band_gap。
"""

# 新的串行流程，优化检索结果传递
# 1. 用户输入
user_message = task

# 2. 需求澄清
clarified_req = requirement_agent.generate_reply([{"role": "user", "content": user_message}])
print("-------------------------------- clarified_req --------------------------------")
print(clarified_req)

# 3. 意图解析
intent = intent_agent.generate_reply([{"role": "user", "content": clarified_req}])
print("-------------------------------- intent --------------------------------")
print(intent)

# 提取 intent 中的 JSON 块
json_match = re.search(r'```json\s*(.*?)\s*```', intent, re.DOTALL)
if json_match:
    intent_json_str = json_match.group(1)
else:
    # 尝试直接找大括号包裹的 JSON
    json_match = re.search(r'({[\s\S]*})', intent)
    if json_match:
        intent_json_str = json_match.group(1)
    else:
        raise ValueError("未能在 intent 响应中找到 JSON 块，请检查 agent 输出格式。")

# 4. 数据源选择
source = source_agent.generate_reply([{"role": "user", "content": intent}])
print("-------------------------------- source --------------------------------")
print(source)
# 5. 检索
intent_dict = json.loads(intent_json_str)
retrieved_snippets = retrieve_snippets(intent_dict)
print("-------------------------------- retrieved_snippets --------------------------------")
print(retrieved_snippets)

# 6. 组装消息给 code_writer_agent
code_writer_input = f"意图: {intent}\n检索到的示例:\n{retrieved_snippets}\n"
code = code_writer_agent.generate_reply([{"role": "user", "content": code_writer_input}])

print(code)