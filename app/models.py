from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(default=3, ge=1, le=10, description="召回条数")


class AskResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str
