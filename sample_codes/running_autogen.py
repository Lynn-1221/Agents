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
    "temperature": 0.9
}

chatbot_system_prompt = "You are a witty and knowledgeable chatbot acting as a fun fact provider. Be friendly, engaging, and ensure your facts are accurate."
agent = ConversableAgent(
    name="chatbot",
    system_message=chatbot_system_prompt,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

reply = agent.generate_reply(
    messages=[{"content": "Tell me a fun fact about money.请用中文回答", "role": "user"}]
)
print(reply)