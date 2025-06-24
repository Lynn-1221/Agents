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

from autogen import ConversableAgent

# 目标：创建 5 个代理（3个 llm 代理，一个 human 代理，一个 summary 代理，并让他们（除 summary 代理外）进行对话，以完成人工客户介入前的信息获取流程）
# 1. llm 代理：
# 1.1 客户基本信息收集代理：收集客户的基本信息（姓名和位置）
# 1.2 客户问题收集代理：确定客户的问题
# 1.3 客户互动代理：用于与客户进行互动，直到人工代理可以介入
# 2. human 代理：
# 2.1 用户代理：客户信息输入接口
# 3. summary 代理：用于总结对话内容

## 1. 创建代理
# 1.1 创建 summary 代理
manual_summary_prompt = """你是一位专业的中文对话总结师。
请严格使用简体中文对以下对话内容进行精准、简洁的总结。
在你的回复中，绝对不能包含任何英文字符或单词。
---
"""
summary_agent = ConversableAgent(
    name="summary_bot",
    system_message=manual_summary_prompt,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# 1.2 创建 personal_information_agent
personal_information_agent = ConversableAgent(
    name="Personal_Information_Agent",
    system_message="""
    你是一位乐于助人的客户接待代理，服务于电话提供商 ACME。
    你的任务是收集客户的基本信息，仅包括客户的【姓名】和【所在地区】。
    注意：请不要询问其他任何信息。
    在客户提供完姓名和地区后，请重复一遍客户的信息，表示感谢，并提示客户输入 “TERMINATE” 以进入问题描述环节。
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content").upper()
)

# 1.3 创建 issue_collection_agent
issue_collection_agent = ConversableAgent(
    name="Issue_Collection_Agent",
    system_message="""
    你是一位乐于助人的客户接待代理，服务于电话提供商 ACME，专门帮助新客户顺利开始使用我们的产品。
    你的任务是收集客户当前所使用的【ACME产品名称】以及【客户遇到的问题】。
    注意：请不要询问其他任何信息。
    在客户描述完他们的问题后，请重复一遍客户的描述，并补充一句：“如果我理解正确，请输入 ‘TERMINATE’。”

    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content").upper()
)

# 1.4 创建 customer_engagement_agent
customer_engagement_agent = ConversableAgent(
    name="Customer_Engagement_Agent",
    system_message="""
    你是一位富有亲和力的客户服务代理。
    你的任务是与客户进行互动，了解客户感兴趣的【新闻或趣味主题偏好】。
    你可以基于客户的兴趣和提供的信息，分享轻松有趣的内容，比如冷知识、笑话或有趣的故事等。
    请尽量让互动过程轻松愉快、富有趣味性！
    当互动结束时，请返回 “TERMINATE”。
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content").upper()
)

# 1.5 创建 user_proxy_agent
customer_proxy_agent = ConversableAgent(
    name="Customer_Proxy_Agent",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="ALWAYS"
)

# 2. 开启对话
def my_chinese_summary(sender, recipient, summary_args):
    # 获取 sender 和 recipient 之间的对话历史
    messages = sender._oai_messages.get(recipient, [])
    conversation_text = "\n".join([
        f"{msg.get('name', msg.get('role'))}: {msg.get('content', '')}"
        for msg in messages if msg.get('content')
    ])
    prompt = summary_args.get("summary_prompt", "请返回用户基本信息，并以 JSON 格式返回。{'name': '', 'location': ''}")
    agent = summary_args.get("agent", sender)
    summary = agent.generate_reply(
        messages=[{"role": "user", "content": prompt + "\n以下是需要总结的对话内容:\n" + conversation_text}]
    )
    return summary

chat = []
# 2.1 收集客户基本信息
chat.append(
    {
        "sender": personal_information_agent,
        "recipient": customer_proxy_agent,
        "message": "您好，我将帮助您解决您使用我们产品时遇到的任何问题。首先，请告诉我您的姓名和所在地区。",
        "summary_method": my_chinese_summary,
        "summary_args": {
            "summary_prompt": "请返回用户基本信息，并以 JSON 格式返回。{'name': '', 'location': ''}",
            "agent": summary_agent
        },
        "clear_history": True
    }
)

# 2.2 收集客户问题
chat.append(
    {
        "sender": issue_collection_agent,
        "recipient": customer_proxy_agent,
        "message": "很好，接下来，请告诉我您当前使用的 ACME 产品名称以及您遇到的问题。",
        "summary_method": my_chinese_summary,
        "summary_args": {
            "summary_prompt": "请对以下内容进行中文总结：",
            "agent": summary_agent
        },
        "clear_history": False
    }
)

# 2.3 客户互动
chat.append(
    {
        "sender": customer_engagement_agent,
        "recipient": customer_proxy_agent,
        "message": """在我们等待人工客服接手并帮助您解决问题的同时，
        您能否告诉我更多关于您如何使用我们产品的信息，或者一些您感兴趣的话题？""",
        "summary_method": my_chinese_summary,
        "summary_args": {
            "summary_prompt": "请对以下内容进行中文总结：",
            "agent": summary_agent
        },
        "max_turns": 2
    }
)

from autogen import initiate_chats

chat_results = initiate_chats(chat)

from pprint import pprint

print("-------------------------------- 以下是对话总结 --------------------------------")
for chat_result in chat_results:
    pprint(chat_result.summary)

