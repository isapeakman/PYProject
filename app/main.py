import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request

from app.agent import ask
from app.logging_config import configure_logging
from app.models import AskRequest, AskResponse, HealthResponse

configure_logging()
logger = logging.getLogger("app.main")

app = FastAPI(title="ecommerce-agent")


@app.middleware("http")
async def request_log_middleware(request: Request, call_next):
    """记录 HTTP 请求全链路耗时与状态码。"""
    request_id = str(uuid4())[:8]
    start = time.perf_counter()
    logger.info(
        "[%s] request.start method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "[%s] request.end method=%s path=%s status=%s elapsed_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查接口，便于确认服务已启动。"""
    return HealthResponse(status="ok")


@app.post("/ask", response_model=AskResponse)
def ask_api(payload: AskRequest) -> AskResponse:
    """接收前端问题并返回 RAG 问答结果。"""
    request_id = str(uuid4())[:8]
    logger.info(
        "[%s] ask.input query=%r top_k=%s",
        request_id,
        payload.query,
        payload.top_k,
    )
    answer = ask(payload.query, top_k=payload.top_k, request_id=request_id)
    logger.info("[%s] ask.output answer=%r", request_id, answer)
    return AskResponse(answer=answer)
