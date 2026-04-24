import os
import time
import logging
import re

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import APITimeoutError
from app.IntentEnum import Intent, IntentResult, INTENT_CLASSIFY_PROMPT
from app.prompts import RAG_PROMPT_TEMPLATE, GREETING_PROMPT_TEMPLATE
from app.vector_store import search_products
from app.logging_config import configure_logging
import re
import json
load_dotenv()

BASE_URL = os.getenv("ZHIPUAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "GLM-4.6V")
configure_logging()
logger = logging.getLogger("app.agent")

# 读取智谱 API Key，避免在代码中硬编码。
def _get_api_key() -> str:
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 或系统环境变量中设置 ZHIPUAI_API_KEY")
    return api_key


llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=_get_api_key,
    base_url=BASE_URL,
    timeout=60,
    max_retries=3,
)

# 初始化提示模板
prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# 截图文本，防止输出过长
def _short_text(text: str, max_len: int = 300) -> str:
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}...(truncated)"

# 主函数，用于处理用户查询
def ask(query: str, top_k: int = 3, request_id: str = "local") -> str:
    # 先意图分类
    intent_result = classify_intent(query)
    intent = intent_result.intent
    confidence = intent_result.confidence
    reasoning = intent_result.reasoning
    logger.info(
        "[%s] intent.classify intent=%s confidence=%.2f reasoning=%s",
        request_id,
        intent,
        confidence,
        reasoning,
    )
    # 根据意图分类，调用不同的处理函数
    if intent == Intent.PRODUCT_QUERY:
        return handle_product_query(query, top_k=top_k, request_id=request_id)
    elif intent == Intent.PRICE_CHECK:
        return handle_price_check(query, top_k=top_k, request_id=request_id)
    elif intent == Intent.COMPARISON:
        return handle_comparison(query, top_k=top_k, request_id=request_id)
    elif intent == Intent.RETURN_REFUND:
        return handle_return_refund(query, top_k=top_k, request_id=request_id)
    elif intent == Intent.SHIPPING:
        return handle_shipping(query, top_k=top_k, request_id=request_id)
    elif intent == Intent.COMPLAINT:
        return handle_complaint(query, request_id=request_id)
    elif intent == Intent.GREETING:
        return handle_greeting(query, request_id=request_id)
    else:
        return "抱歉，我不明白你的意思"
    
def handle_product_query(query: str, top_k: int = 3, request_id: str = "default") -> str:
    """处理商品查询意图"""
    contexts = search_products_context(query, top_k=top_k, request_id=request_id)
    return invokeLLMByPrompt(RAG_PROMPT_TEMPLATE, contexts, query, request_id=request_id)

def handle_price_check(query: str, top_k: int = 3, request_id: str = "default") -> str:
    """处理价格咨询意图"""
    contexts = search_products_context(query, top_k=top_k, request_id=request_id)
    return invokeLLMByPrompt(RAG_PROMPT_TEMPLATE, contexts, query, request_id=request_id)

def handle_greeting(query: str, request_id: str = "default") -> str:
    """处理问候闲聊意图"""
    return invokeLLMByPrompt(GREETING_PROMPT_TEMPLATE, [], query, request_id=request_id)

def handle_comparison(query: str, top_k: int = 3, request_id: str = "default") -> str:
    """处理产品比较意图"""
    contexts = search_products_context(query, top_k=top_k, request_id=request_id)
    return invokeLLMByPrompt(RAG_PROMPT_TEMPLATE, contexts, query, request_id=request_id)

def handle_return_refund(query: str, top_k: int = 3, request_id: str = "default") -> str:
    """处理退换货意图"""
    contexts = search_products_context(query, top_k=top_k, request_id=request_id)
    return invokeLLMByPrompt(RAG_PROMPT_TEMPLATE, contexts, query, request_id=request_id)

def handle_shipping(query: str, top_k: int = 3, request_id: str = "default") -> str:
    """处理物流配送意图"""
    contexts = search_products_context(query, top_k=top_k, request_id=request_id)
    return invokeLLMByPrompt(RAG_PROMPT_TEMPLATE, contexts, query, request_id=request_id)

def handle_complaint(query: str, request_id: str = "default") -> str:
    """处理投诉建议意图"""
    return invokeLLMByPrompt(RAG_PROMPT_TEMPLATE, [], query, request_id=request_id)

# 检索商品上下文
def search_products_context(query: str, top_k: int = 3, request_id: str = "default") -> list:
    """检索商品上下文"""
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
        return []
    return contexts

# 调用大模型生成答案
def invokeLLMByPrompt(prompt: str, contexts: list, query: str, request_id: str = "default") -> str:
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
            response =  llm.invoke(prompt_text)
            llm_ms = (time.perf_counter() - llm_start) * 1000
            answer = response.content
            logger.info(
                "[%s] llm.output attempt=%s elapsed_ms=%.2f answer_preview=%r",
                request_id,
                attempt + 1,
                llm_ms,
                _short_text(str(answer)),
            )
            return str(answer)
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

# 意图规则分类
def classify_intent(query: str) -> IntentResult:
    """意图分类器"""

    # 先尝试快速意图分类
    fast_intent = classify_intent_fast(query)
    if fast_intent:
        return IntentResult(
            intent=fast_intent,
            confidence=1.0,
            reasoning="快速匹配意图",
        )

    # 轻量级LLM兜底，用于处理复杂意图
    prompt = INTENT_CLASSIFY_PROMPT.format(query=query)
    response =  llm.invoke(prompt)
    logger.info(
        "llm invoke intention answer_preview=%r",
        _short_text(str(response.content)) 
    )

    try:
        result = extract_json_from_response(str(response.content))
        return IntentResult(
            intent=result.get("intent", "other"),
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", "LLM分类"),
        )
    except Exception as e:
        logger.warning(f"解析意图结果失败: {e}")
        return IntentResult(
            intent=Intent.OTHER,
            confidence=0.3,
            reasoning="解析失败，默认为其他意图",
        )



FAST_INTENT_PATTERNS = {
    Intent.PRODUCT_QUERY: [r"什么", r"怎么样", r"功能", r"规格", r"参数", r"型号", r"样式", r"颜色", r"尺寸", r"材质", r"质量"],
    Intent.PRICE_CHECK: [r"多少[钱价]", r"price", r"价钱", r"价格", r"多少钱", r"价位"],
    Intent.COMPARISON: [r"对比", r"比较", r"哪个好", r"和.*哪个", r"vs", r"PK"],
    Intent.RETURN_REFUND: [r"退货", r"退款", r"换货", r"售后", r"退钱", r"退货政策"],
    Intent.SHIPPING: [r"物流", r"快递", r"发货", r"配送", r"几天到", r"运费", r"包邮"],
    Intent.COMPLAINT: [r"投诉", r"建议", r"问题", r"质量", r"不好", r"差", r"坏"],
    Intent.GREETING: [r"你好", r"您好", r"hi", r"hello", r"早上好", r"下午好", r"晚上好", r"在吗"],
}
# 快速意图分类
def classify_intent_fast(query: str) -> Intent | None:
    """规则快速匹配"""
    for intent, patterns in FAST_INTENT_PATTERNS.items():
        if any(re.search(p, query) for p in patterns):
            return intent
    return None




def extract_json_from_response(content: str) -> dict:
    """从 LLM 响应中提取 JSON"""
    content = str(content)
    
    # 尝试直接解析
    try:
        return json.loads(content)
    except:
        pass
    
    # 提取 ```json ... ``` 中的内容
    match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    # 提取 { ... } 中的内容
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    
    raise ValueError(f"无法从响应中解析JSON: {content[:200]}")