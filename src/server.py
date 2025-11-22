from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, constr

from . import agent

QuestionStr = constr(min_length=2, strip_whitespace=True)


class ChatRequest(BaseModel):
    question: QuestionStr = Field(..., description="Natural language procurement question")


class ChatResponse(BaseModel):
    answer: str


app = FastAPI(title="Procurement AI Assistant", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    try:
        answer = agent.chat(payload.question)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(answer=answer)

