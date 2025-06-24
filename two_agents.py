from autogen import ConversableAgent
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

# bret = ConversableAgent(
#     name="Bret",
#     llm_config=llm_config,
#     system_message="你的名字是特特，你是中国双人喜剧节目中的脱口秀喜剧演员。如果你打算结束对话，请说'我得走了'",
#     human_input_mode="NEVER",
#     is_termination_msg=lambda msg: "我得走了" in msg["content"],
# )
# jemaine = ConversableAgent(
#     name="Jemaine",
#     llm_config=llm_config,
#     system_message="你的名字是吉米，你是中国双人喜剧节目中的脱口秀喜剧演员。如果你打算结束对话，请说'我得走了'",
#     human_input_mode="NEVER",
#     is_termination_msg=lambda msg: "我得走了" in msg["content"],
# )

# # 步骤 1：发起对话，但禁用内置的总结功能
# chat_result = bret.initiate_chat(
#     recipient=jemaine,
#     message="你好，杰米，让我们继续讲笑话吧，先从关于医生的笑话开始",
# )

# import pprint

# pprint.pprint(chat_result.chat_history)
# pprint.pprint(chat_result.cost)

# # 步骤 2：手动创建并执行中文总结任务
# # 首先，将对话历史格式化为一段文本
# conversation_text = "\n".join([f"{msg.get('name', msg.get('role'))}: {msg.get('content', '')}" for msg in chat_result.chat_history if msg.get('content')])

# # 接下来，创建我们自己的、完全可控的总结提示
# manual_summary_prompt = """你是一位专业的中文对话总结师。
# 请严格使用简体中文对以下对话内容进行精准、简洁的总结。
# 在你的回复中，绝对不能包含任何英文字符或单词。
# ---
# """

# agent = ConversableAgent(
#     name="summary_bot",
#     system_message=manual_summary_prompt,
#     llm_config=llm_config,
#     human_input_mode="NEVER"
# )

# # 最后，让一个 agent 来生成总结
# # 注意：我们这里不使用 initiate_chat，而是用 generate_reply 来获取一次性回复
# summary = agent.generate_reply(
#     messages=[{"role": "user", "content": "以下是需要总结的对话内容:" + conversation_text}]
# )

# print("\n\n--- 手动生成的中文总结 ---")
# print(summary)

context = """
一级维度：高等知识问答
二级维度：主观问答
╭─────────────────────────────────────────────╮
│ 三级维度 & 样本量 │ 评测方式                │ 指标 │
├─────────────────────────────────────────────┤
│ 材料属性与器件性能 112│ 要点对比+多模型加权打分 │ Score│
│ 材料选择             60│ 同上                    │ Score│
│ 材料制备             91│ 同上                    │ Score│
│ 定义                 22│ 同上                    │ Score│
│ 研究背景与进展       38│ 同上                    │ Score│
│ 用途                 28│ 同上                    │ Score│
│ 原理                120│ 同上                    │ Score│
│ 实验                 21│ 同上                    │ Score│
╰─────────────────────────────────────────────╯
备注：暂无图像/多模态任务，聚焦文本回答；评分依赖核心要点表与多模型加权。
"""

evaluator_prompt = f"""
你是「材料大模型评估官 Agent」。阅读 {context} 后，与你的科研顾问对话，
目标是优化“材料大模型辅助实验”能力评测体系的【主观问答】部分。

职责
1. 每轮聚焦 1-2 个具体条目（示例：‘材料制备 91’ 或 ‘实验 21’）：
   - 追问 Scientist Agent 这些题目如何产生、样本是否够、要点表来源、与实验决策关联度；
   - 诊断指标与评测方式（Score+模型打分）是否足以区分模型优劣；
   - 提出可执行改进：增删、样本扩充、指标替换（Accuracy/F1/长答案 BLEU…）、链路补强等。
2. 仅讨论文本/数值/工具调用，**不触及图像或其他多模态**。
3. 每轮结束输出  
   【本轮发现】①…②… (≤3 条)  
4. 连续三轮无新发现即可声明“我认为我的工作已完成”结束。

对话风格  
- 精准追问，不空谈材料机理；  
- 用辅助实验场景（如“选参数前先问模型”“实验失败后溯因”）佐证建议。
"""
evaluator = ConversableAgent(
    name="evaluator",
    system_message=evaluator_prompt,
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "我认为我的工作已完成" in msg["content"]
)

scientist_prompt = f"""
你是「微电子材料科研顾问 Agent」。阅读 {context} 后，与评测官讨论
“主观问答”评测是否真正帮助实验环节。

职责
1. 按评测官聚焦的条目，说明：
   - 题库怎么来的、样本量够不够、要点表如何维护；
   - 模型回答好坏会如何影响实验（举真实/假想案例）。
2. 每轮主动补充 1 个你觉得该部分欠缺的能力点
   (如“多目标权衡材料推荐问答”“实验条件冲突检测”)，并描述理想评测思路。
3. 对评测官提出的改进建议做实战验证或质疑。
4. 仅讨论文本/数值/工具调用，**不涉及图像或多模态**。
5. 每轮结束输出  
   【科研视角补充】①…②… (≤3 条)  
6. 连续三轮无新补充即可声明“我认为我的工作已完成”。

对话风格  
- 实验驱动：用具体流程/痛点支撑观点；  
- 持续挑战：避免轻易接受结论。

"""
scientist = ConversableAgent(
    name="scientist",
    system_message=scientist_prompt,
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "我认为我的工作已完成" in msg["content"]
)

chat_result = scientist.initiate_chat(
    recipient=evaluator,
    message="你好，评估师，让我们开始讨论材料大模型能力评估体系吧"
)
print("------------------------------------------------------------------------------------------------")
print(chat_result.cost)