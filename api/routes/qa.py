"""
Optional Q&A route — only loaded when ENABLE_QA=true in .env
"""
from fastapi import APIRouter
from core.generation.content_generator import content_generator

router = APIRouter(prefix="/qa", tags=["qa"])


@router.post("/ask")
async def ask(client_id: str, client_name: str, question: str):
    return content_generator.answer_question(client_id, client_name, question)
