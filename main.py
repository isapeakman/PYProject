import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请在 .env 或系统环境变量中设置 ZHIPUAI_API_KEY")

llm = ChatOpenAI(
    model="GLM-4.6V",  # 或 glm-4-flash 等
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

resp = llm.invoke("你能帮我解决什么问题")
print(resp.content)