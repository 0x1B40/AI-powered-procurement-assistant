from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, constr
from typing import List

from ..core import agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

QuestionStr = constr(min_length=2, strip_whitespace=True)


class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender ('user' or 'assistant')")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    question: QuestionStr = Field(..., description="Natural language procurement question")
    conversation_history: List[Message] = Field(default_factory=list, description="Previous conversation messages")


class ChatResponse(BaseModel):
    answer: str


app = FastAPI(title="Procurement AI Assistant", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    try:
        # Convert the conversation history to BaseMessage objects
        conversation_history = []
        for msg in payload.conversation_history:
            if msg.role == "user":
                conversation_history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                conversation_history.append(AIMessage(content=msg.content))

        answer, _ = agent.chat(payload.question, conversation_history)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(answer=answer)

