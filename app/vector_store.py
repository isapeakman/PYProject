import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

BASE_URL = os.getenv("ZHIPUAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-3")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "products")


def _get_api_key() -> str:
    """读取智谱 API Key，缺失时给出明确报错。"""
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 或系统环境变量中设置 ZHIPUAI_API_KEY")
    return api_key


def get_embeddings() -> OpenAIEmbeddings:
    """构造统一的 embedding 客户端配置。"""
    return OpenAIEmbeddings(
        api_key=_get_api_key(),
        model=EMBEDDING_MODEL,
        base_url=BASE_URL,
        request_timeout=60,
        max_retries=3,
    )


def get_vector_store() -> Chroma:
    """返回持久化的 Chroma 向量库实例。"""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=VECTOR_DB_DIR,
    )


def search_products(query: str, k: int = 3) -> list[str]:
    """按用户问题召回最相关的商品文本。"""
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]
