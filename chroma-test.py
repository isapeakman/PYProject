import os
import time

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from openai import APITimeoutError

load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请在 .env 或系统环境变量中设置 ZHIPUAI_API_KEY")
base_url = "https://open.bigmodel.cn/api/paas/v4/"
# 1. 初始化
embeddings = OpenAIEmbeddings(
    api_key=api_key,
    model="embedding-3",
    base_url=base_url,
    request_timeout=60,
    max_retries=3,
)
vector_store = Chroma(
    collection_name="products",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# 2. 准备电商知识库
product_docs = [
    "索尼WH-1000XM5耳机，价格2299元，降噪效果极佳，续航30小时",
    "苹果AirPods Pro 2，价格1899元，支持主动降噪，续航24小时",
    "小米手环8，价格249元，支持心率监测，续航14天",
]

# 3. 批量添加（只添加一次，后续可以注释掉）
vector_store.add_texts(product_docs)

# 4. 检索函数
def search_products(query: str, k: int = 3) -> list:
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

# 5. 基于检索结果的问答
llm = ChatOpenAI(
    model="GLM-4.6V",  # 或 glm-4-flash 等
    api_key=api_key,
    base_url=base_url,
    request_timeout=60,
    max_retries=3,
)

def ask(query: str) -> str:
    # 检索相关商品
    contexts = search_products(query)
    
    if not contexts:
        return "抱歉，没找到相关信息"
    
    # 构建提示词
    prompt = ChatPromptTemplate.from_template("""
你是一个电商客服，基于以下商品信息回答问题。

商品信息：
{context}

用户问题：{query}

要求：
- 只基于上面的信息回答
- 如果信息不够，说"我查不到"
- 价格用¥符号
""")
    
    prompt_text = prompt.format(context="\n".join(contexts), query=query)
    for attempt in range(3):
        try:
            response = llm.invoke(prompt_text)
            return response.content
        except APITimeoutError:
            if attempt == 2:
                return "请求超时，请稍后重试"
            time.sleep(2 ** attempt)
    
    return "请求失败，请稍后重试"

# 6. 测试
print(ask("索尼耳机多少钱？"))
# 输出：索尼WH-1000XM5耳机的价格是¥2299元。

print(ask("有什么降噪耳机？"))
# 输出：索尼WH-1000XM5耳机和苹果AirPods Pro 2都支持降噪功能...