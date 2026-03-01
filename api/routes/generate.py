from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os

from core.generation.content_generator import content_generator
from core.vector_store import vector_store_manager

router = APIRouter(prefix="/generate", tags=["generate"])
templates = Jinja2Templates(directory="templates")


from pydantic import BaseModel, Field
from typing import Optional

class GenerateRequest(BaseModel):
    client_id: str
    client_name: str
    # User's frontend sends input_content, but requested internal name is generic_content
    generic_content: str = Field(..., alias="input_content")
    content_type: str = "blog_post"
    mode: str = "contextualize"   # "contextualize" or "scratch"
    topic: Optional[str] = None   # Only used in scratch mode
    source_types: Optional[list[str]] = None  # Optional filter, None = all

    class Config:
        populate_by_name = True


from qdrant_client.models import Filter, FieldCondition, MatchValue
from core.generation.prompts import (
    CONTENT_GEN_SYSTEM, CONTEXTUALIZE_PROMPT, GENERATE_FROM_SCRATCH_PROMPT,
    CONTENT_TYPE_CONFIGS
)
from core.llm_router import llm_router

def retrieve_context_for_generation(
    qdrant_client,
    collection_name: str,
    query_text: str,
    client_id: str,
) -> dict:
    """
    Four-pass retrieval that separates brand voice from facts.
    Returns a dict with:
      - brand_voice_context (str)
      - factual_context (str)
      - brand_voice_count (int)
      - factual_count (int)
    """
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    
    embed_model = FastEmbedEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_dir="./.fastembed_cache"
    )
    query_vector = embed_model.get_text_embedding(query_text)

    # ── PASS 1: Brand voice chunks (always retrieve, not semantic) ──────────
    brand_voice_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="source_folder",
                    match=MatchValue(value="brand_voice")
                )
            ]
        ),
        limit=4,
        with_payload=True,
    ).points

    # ── PASS 2: Factual semantic search (exclude brand_voice folder) ─────────
    factual_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=Filter(
            must_not=[
                FieldCondition(
                    key="source_folder",
                    match=MatchValue(value="brand_voice")
                ),
                FieldCondition(
                    key="source_folder",
                    match=MatchValue(value="pricing")
                ),
            ]
        ),
        limit=8,
        with_payload=True,
    ).points

    # ── PASS 3: Pricing sweep (always pull pricing chunks explicitly) ────────
    pricing_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="source_folder",
                    match=MatchValue(value="pricing")
                )
            ]
        ),
        limit=3,
        with_payload=True,
    ).points

    # ── PASS 4: Timeline/Location/Budget sweep (specific semantic pull) ───────
    timeline_query = "timeline duration weeks location city neighborhood budget cost total"
    timeline_vector = embed_model.get_text_embedding(timeline_query)
    timeline_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=timeline_vector,
        query_filter=Filter(
            must_not=[
                FieldCondition(
                    key="source_folder",
                    match=MatchValue(value="brand_voice")
                )
            ]
        ),
        limit=4,
        with_payload=True,
    ).points

    # ── BUILD CONTEXT STRINGS ─────────────────────────────────────────────────
    brand_voice_chunks = [
        r.payload.get("text", r.payload.get("_node_content", ""))
        for r in brand_voice_results
        if r.payload
    ]
    brand_voice_context = "\n---\n".join(brand_voice_chunks) if brand_voice_chunks else (
        "No brand voice document found. Match a professional, clear, and specific tone."
    )

    seen_ids = set()
    merged_factual = []
    all_results = list(factual_results) + list(pricing_results) + list(timeline_results)
    
    for r in all_results:
        point_id = r.id
        if point_id not in seen_ids:
            seen_ids.add(point_id)
            text = r.payload.get("text", r.payload.get("_node_content", ""))
            if text:
                source = r.payload.get("source_folder", "general")
                merged_factual.append(f"[{source}]\n{text}")
    
    factual_context = "\n---\n".join(merged_factual) if merged_factual else (
        "No factual context found for this client."
    )

    return {
        "brand_voice_context": brand_voice_context,
        "factual_context": factual_context,
        "brand_voice_count": len(brand_voice_chunks),
        "factual_count": len(merged_factual),
    }


@router.get("/", response_class=HTMLResponse)
async def generate_ui(request: Request):
    clients = vector_store_manager.list_clients()
    return templates.TemplateResponse("generate.html", {
        "request": request,
        "clients": clients,
        "content_types": CONTENT_TYPE_CONFIGS,
        "qa_enabled": os.getenv("ENABLE_QA", "false").lower() == "true",
    })


@router.post("/content")
async def generate_content(request: Request, body: GenerateRequest):
    """
    Generate or contextualize content using four-pass RAG retrieval.
    Separates brand voice from factual context for higher quality output.
    """
    try:
        collection_name = f"client_{body.client_id}"
        
        # Four-pass retrieval using established client
        retrieval = retrieve_context_for_generation(
            qdrant_client=vector_store_manager.client,
            collection_name=collection_name,
            query_text=body.generic_content,
            client_id=body.client_id,
        )
        brand_voice_context = retrieval["brand_voice_context"]
        factual_context     = retrieval["factual_context"]

        # Build content type config
        content_cfg = CONTENT_TYPE_CONFIGS.get(
            body.content_type, 
            {"label": body.content_type, "length": "400-800 words"}
        )

        # Build prompt based on mode
        if body.mode == "scratch":
            user_prompt = GENERATE_FROM_SCRATCH_PROMPT.format(
                client_name=body.client_name,
                brand_voice_context=brand_voice_context,
                factual_context=factual_context,
                content_type=content_cfg["label"],
                topic=body.topic or "general",
                target_length=content_cfg["length"],
            )
        else:
            # Default: contextualize mode
            user_prompt = CONTEXTUALIZE_PROMPT.format(
                client_name=body.client_name,
                brand_voice_context=brand_voice_context,
                factual_context=factual_context,
                content_type=content_cfg["label"],
                input_content=body.generic_content,
            )

        # Call LLM via your existing llm_router (Groq primary, Gemini fallback)
        llm_result = llm_router.complete(
            system=CONTENT_GEN_SYSTEM,
            prompt=user_prompt,
        )

        return {
            "status": "success",
            "content": llm_result["text"],
            "provider": llm_result["provider"],
            "client_id": body.client_id,
            "mode": body.mode,
            "brand_voice_chunks_used": retrieval["brand_voice_count"],
            "factual_chunks_used": retrieval["factual_count"],
            # Maintain field for frontend compatibility
            "chunks_retrieved": retrieval["brand_voice_count"] + retrieval["factual_count"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/qa")
async def answer_question(client_id: str, client_name: str, question: str):
    if os.getenv("ENABLE_QA", "false").lower() != "true":
        raise HTTPException(status_code=403, detail="Q&A feature is disabled")
    # Note: QA could also benefit from multi-pass retrieval, but request only mentioned generation
    return content_generator.answer_question(client_id, client_name, question)
