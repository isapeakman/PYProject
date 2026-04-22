import os
import time
import logging

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import APITimeoutError

from app.prompts import RAG_PROMPT_TEMPLATE
from app.vector_store import search_products

load_dotenv()

BASE_URL = os.getenv("ZHIPUAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "GLM-4.6V")
logger = logging.getLogger("app.agent")


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


def _short_text(text: str, max_len: int = 300) -> str:
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}...(truncated)"


def ask(query: str, top_k: int = 3, request_id: str = "local") -> str:
    """先检索商品上下文，再调用大模型生成答案。"""
    retrieve_start = time.perf_counter()
    contexts = search_products(query, k=top_k)
    retrieve_ms = (time.perf_counter() - retrieve_start) * 1000
    logger.info(
        "[%s] rag.retrieve done top_k=%s hit_count=%s elapsed_ms=%.2f",
        request_id,
        top_k,
        len(contexts),
        retrieve_ms,
    )

    if not contexts:
        logger.info("[%s] rag.no_context query=%r", request_id, query)
        return "抱歉，没找到相关信息"

    prompt_text = prompt.format(context="\n".join(contexts), query=query)
    logger.info(
        "[%s] llm.input query=%r prompt_preview=%r",
        request_id,
        _short_text(query, 120),
        _short_text(prompt_text),
    )
    for attempt in range(3):
        try:
            llm_start = time.perf_counter()
            response = llm.invoke(prompt_text)
            llm_ms = (time.perf_counter() - llm_start) * 1000
            answer = response.content
            logger.info(
                "[%s] llm.output attempt=%s elapsed_ms=%.2f answer_preview=%r",
                request_id,
                attempt + 1,
                llm_ms,
                _short_text(answer),
            )
            return answer
        except APITimeoutError:
            # 网络抖动时做简单退避重试，避免一次超时就直接失败。
            logger.warning(
                "[%s] llm.timeout attempt=%s",
                request_id,
                attempt + 1,
            )
            if attempt == 2:
                return "请求超时，请稍后重试"
            time.sleep(2**attempt)

    return "请求失败，请稍后重试"
