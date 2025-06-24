from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key, base_url=base_url)

messages = [
    SystemMessage(content="You are a helpful assistant that can answer questions and help with tasks."),
    HumanMessage(content="What is the capital of France?")
]

res = llm.invoke(messages)
print(res.content)

# import openai

# client = openai.OpenAI(api_key=api_key, base_url=base_url)
# res2 = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {
#             "role": "system",
#             "content": "You are a helpful assistant that can answer questions and help with tasks."
#         },
#         {
#             "role": "user",
#             "content": "What is the capital of France?"
#         }]
# )
# print(res2.choices[0].message.content)