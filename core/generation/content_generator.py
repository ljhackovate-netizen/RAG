"""
RAG content generation: retrieve relevant chunks → LLM generates client-specific content.
Dynamic retrieval — no domain keywords in queries. The embedding model finds semantic matches.
"""
import logging

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from core.embeddings import embed_model
from core.vector_store import vector_store_manager
from core.llm_router import llm_router
from core.generation.prompts import (
    CONTENT_GEN_SYSTEM, QA_SYSTEM,
    CONTEXTUALIZE_PROMPT, GENERATE_FROM_SCRATCH_PROMPT, QA_PROMPT,
    CONTENT_TYPE_CONFIGS,
)

logger = logging.getLogger(__name__)


def _retrieve_context(client_id: str, query: str, top_k: int = 8) -> str:
    """
    Retrieve top-K relevant chunks for client. Strictly filtered by client_id.
    Query is the actual user input — no synthetic keyword injection.
    """
    collection_name = vector_store_manager.collection_name(client_id)
    vector_store = QdrantVectorStore(
        client=vector_store_manager.client,
        collection_name=collection_name,
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    if not nodes:
        return "No relevant content found in this client's knowledge base."

    parts = []
    for i, node in enumerate(nodes, 1):
        fname = node.metadata.get("file_name", "unknown")
        folder = node.metadata.get("source_folder", "")
        parts.append(f"[{i}] Source: {folder}/{fname}\n{node.get_content()}")
    return "\n\n---\n\n".join(parts)


class ContentGenerator:

    def generate(
        self,
        client_id: str,
        client_name: str,
        input_content: str,
        content_type: str = "blog_post",
        mode: str = "contextualize",   # "contextualize" | "scratch"
    ) -> dict:
        ct = CONTENT_TYPE_CONFIGS.get(content_type, CONTENT_TYPE_CONFIGS["blog_post"])

        # Retrieval query = the actual user input. No keyword injection.
        retrieval_query = input_content[:600]
        context = _retrieve_context(client_id, retrieval_query)

        if mode == "scratch":
            prompt = GENERATE_FROM_SCRATCH_PROMPT.format(
                client_name=client_name,
                context=context,
                content_type=ct["label"],
                topic=input_content or "general",
                target_length=ct["length"],
            )
        else:
            prompt = CONTEXTUALIZE_PROMPT.format(
                client_name=client_name,
                context=context,
                content_type=ct["label"],
                input_content=input_content,
            )

        result = llm_router.complete(
            prompt=prompt,
            system=CONTENT_GEN_SYSTEM,
            max_tokens=4096,
            temperature=0.4,
        )

        return {
            "content": result["text"],
            "provider": result["provider"],
            "mode": mode,
            "content_type": ct["label"],
            "client_id": client_id,
            "client_name": client_name,
            "chunks_retrieved": context.count("["),
        }

    def answer_question(self, client_id: str, client_name: str, question: str) -> dict:
        context = _retrieve_context(client_id, question, top_k=6)
        result = llm_router.complete(
            prompt=QA_PROMPT.format(client_name=client_name, context=context, question=question),
            system=QA_SYSTEM,
            max_tokens=1024,
            temperature=0.1,
        )
        return {"answer": result["text"], "provider": result["provider"]}


content_generator = ContentGenerator()
