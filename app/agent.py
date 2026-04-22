import os
import time

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import APITimeoutError

from app.prompts import RAG_PROMPT_TEMPLATE
from app.vector_store import search_products

load_dotenv()

BASE_URL = os.getenv("ZHIPUAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "GLM-4.6V")


def _get_api_key() -> str:
    """读取智谱 API Key，避免在代码中硬编码。"""
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 或系统环境变量中设置 ZHIPUAI_API_KEY")
    return api_key


llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=_get_api_key(),
    base_url=BASE_URL,
    request_timeout=60,
    max_retries=3,
)

prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


def ask(query: str, top_k: int = 3) -> str:
    """先检索商品上下文，再调用大模型生成答案。"""
    contexts = search_products(query, k=top_k)
    if not contexts:
        return "抱歉，没找到相关信息"

    prompt_text = prompt.format(context="\n".join(contexts), query=query)
    for attempt in range(3):
        try:
            response = llm.invoke(prompt_text)
            return response.content
        except APITimeoutError:
            # 网络抖动时做简单退避重试，避免一次超时就直接失败。
            if attempt == 2:
                return "请求超时，请稍后重试"
            time.sleep(2**attempt)

    return "请求失败，请稍后重试"
