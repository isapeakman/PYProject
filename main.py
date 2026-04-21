import os

from langchain_openai import ChatOpenAI

api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请先设置环境变量 ZHIPUAI_API_KEY")

llm = ChatOpenAI(
    model="GLM-4.6V",  # 或 glm-4-flash 等
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

resp = llm.invoke("你能帮我解决什么问题")
print(resp.content)