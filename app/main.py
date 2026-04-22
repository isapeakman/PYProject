from fastapi import FastAPI

from app.agent import ask
from app.models import AskRequest, AskResponse, HealthResponse

app = FastAPI(title="ecommerce-agent")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查接口，便于确认服务已启动。"""
    return HealthResponse(status="ok")


@app.post("/ask", response_model=AskResponse)
def ask_api(payload: AskRequest) -> AskResponse:
    """接收前端问题并返回 RAG 问答结果。"""
    answer = ask(payload.query, top_k=payload.top_k)
    return AskResponse(answer=answer)
