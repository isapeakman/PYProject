from enum import Enum
from pydantic import BaseModel

class Intent(str, Enum):
    PRODUCT_QUERY = "product_query"      # 商品查询
    PRICE_CHECK = "price_check"           # 价格咨询
    COMPARISON = "comparison"             # 产品比较
    RETURN_REFUND = "return_refund"      # 退换货
    SHIPPING = "shipping"                # 物流配送
    COMPLAINT = "complaint"              # 投诉建议
    GREETING = "greeting"                # 问候闲聊
    OTHER = "other"                      # 其他

class IntentResult(BaseModel):
    intent: Intent
    confidence: float
    reasoning: str

INTENT_CLASSIFY_PROMPT = """
你是一个电商客服意图分类器。根据用户问题，判断用户意图。

可选意图类型：
- product_query: 商品信息查询（功能、规格、库存等）
- price_check: 价格咨询
- comparison: 产品对比
- return_refund: 退换货咨询
- shipping: 物流配送相关
- complaint: 投诉建议
- greeting: 问候闲聊
- other: 不属于以上类型

用户问题：{query}

请以JSON格式输出：
{{"intent": "意图类型", "confidence": 0.0-1.0置信度, "reasoning": "判断理由"}}
"""