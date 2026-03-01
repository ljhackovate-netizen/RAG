"""
Ingestion pipeline: file → extract → (summarize if long) → chunk → embed → Qdrant
"""
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional
import io

from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore

from core.embeddings import embed_model
from core.vector_store import vector_store_manager
from core.llm_router import llm_router
from core.ingestion.chunker import get_chunk_config
from core.ingestion.extractors import extract_text

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv"}


class IngestionPipeline:

    def ingest_file(
        self,
        file_path_or_bytes: str | io.BytesIO,
        client_id: str,
        client_name: str,
        source_folder: str = "",
        ext: str = None,
        file_name: str = None,
        last_modified: float = None,
    ) -> dict:
        if isinstance(file_path_or_bytes, str):
            fp = Path(file_path_or_bytes)
            if not fp.exists():
                return {"status": "error", "file": fp.name, "reason": "not found"}
            ext = fp.suffix.lower()
            file_name = file_name or fp.name
            last_modified = last_modified or fp.stat().st_mtime
            if source_folder == "":
                source_folder = fp.parent.name
        else:
            if not ext or not file_name:
                raise ValueError("Must provide ext and file_name when using BytesIO")
            ext = ext.lower()
            last_modified = last_modified or datetime.utcnow().timestamp()

        if ext not in SUPPORTED_EXTENSIONS:
            return {"status": "skipped", "file": file_name, "reason": "unsupported type"}

        # 1. Extract raw text directly from memory buffer or file
        raw_text = extract_text(file_path_or_bytes, ext=ext)
        if not raw_text or len(raw_text.strip()) < 30:
            return {"status": "skipped", "file": file_name, "reason": "empty or too short"}

        # 2. Summarize long documents to avoid rate limits during embedding
        processed_text = llm_router.summarize_for_rag(raw_text, file_name)

        # 3. Get chunk config (adaptive, folder-based, no domain keywords)
        # Using a dummy path for config logic if memory stream
        dummy_path = f"dummy/{source_folder}/{file_name}"
        chunk_cfg = get_chunk_config(dummy_path, source_folder)

        # 4. Build LlamaIndex Document with metadata
        doc_id = hashlib.md5(f"{client_id}:{file_name}:{last_modified}".encode()).hexdigest()
        document = Document(
            text=processed_text,
            metadata={
                "client_id": client_id,
                "client_name": client_name,
                "file_name": file_name,
                "source_folder": source_folder,
                "date_ingested": datetime.utcnow().isoformat(),
            },
            id_=doc_id,
            excluded_embed_metadata_keys=["date_ingested"],
        )

        # 5. Setup Qdrant collection
        collection_name = vector_store_manager.ensure_collection(client_id)
        vector_store = QdrantVectorStore(
            client=vector_store_manager.client,
            collection_name=collection_name,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        node_parser = SentenceSplitter(
            chunk_size=chunk_cfg.chunk_size,
            chunk_overlap=chunk_cfg.chunk_overlap,
        )

        # 6. Index (embed + store)
        VectorStoreIndex.from_documents(
            [document],
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=[node_parser],
            show_progress=False,
        )

        logger.info(f"✅ Ingested: {file_name} | client={client_id} | folder={source_folder}")
        return {"status": "success", "file": file_name, "source_folder": source_folder}

    def ingest_folder(self, folder_path: str, client_id: str, client_name: str) -> list[dict]:
        """Recursively ingest all supported files in folder. Skips auto_generated/."""
        results = []
        for fp in sorted(Path(folder_path).rglob("*")):
            if not fp.is_file():
                continue
            if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if "auto_generated" in fp.parts:
                continue  # Skip our own outputs
            source_folder = fp.parent.name
            result = self.ingest_file(str(fp), client_id, client_name, source_folder)
            results.append(result)
        return results


ingestion_pipeline = IngestionPipeline()
