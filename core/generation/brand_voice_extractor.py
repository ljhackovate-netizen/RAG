"""
Brand Voice Auto-Generator.

Flow:
1. Read ingested text files from local client_data folder
2. LLM generates a comprehensive brand voice guide
3. Save file locally: client_data/{client_id}/auto_generated/brand_voice_guide_{timestamp}.txt
4. UPLOAD the file back to Google Drive:
   - Into the SAME client Drive folder the user originally selected
   - Under a subfolder called "auto_generated/" (created if not exists)
   - File name: brand_voice_guide_{timestamp}.txt

This means Drive is the permanent record. Local is just the working cache.
"""
import os
import io
import logging
import warnings
from pathlib import Path
from datetime import datetime

from core.llm_router import llm_router
from core.generation.prompts import BRAND_VOICE_EXTRACTION_PROMPT, BRAND_VOICE_SYSTEM

logger = logging.getLogger(__name__)

MAX_CHARS_FOR_EXTRACTION = 14000   # Cap total input to stay within LLM context


def _read_client_texts(client_id: str, client_data_path: str) -> str:
    """
    Read text content from all ingested files for this client.
    Skips auto_generated folder (our own outputs).
    Caps total chars to avoid LLM context overflow.
    """
    base = Path(client_data_path) / client_id
    if not base.exists():
        return ""

    text_parts = []
    total_chars = 0

    for fp in sorted(base.rglob("*")):
        if not fp.is_file():
            continue
        if "auto_generated" in fp.parts:
            continue
        if fp.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            content = fp.read_text(encoding="utf-8", errors="ignore")
            segment = f"=== {fp.parent.name}/{fp.name} ===\n{content[:3000]}"
            text_parts.append(segment)
            total_chars += len(segment)
            if total_chars >= MAX_CHARS_FOR_EXTRACTION:
                logger.info("Brand voice: reached input cap, using collected text")
                break
        except Exception as e:
            logger.warning(f"Could not read {fp}: {e}")

    return "\n\n".join(text_parts)


def _get_or_create_auto_generated_folder(service, client_drive_folder_id: str) -> str:
    """
    Find or create 'auto_generated' subfolder inside the client's Drive folder.
    Returns the folder ID.
    Requires EDITOR permission on the client folder.
    """
    query = (
        f"'{client_drive_folder_id}' in parents "
        f"and name='auto_generated' "
        f"and mimeType='application/vnd.google-apps.folder' "
        f"and trashed=false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    if files:
        folder_id = files[0]["id"]
        logger.info(f"Found existing auto_generated folder: {folder_id}")
        return folder_id

    # Create it
    folder_meta = {
        "name": "auto_generated",
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [client_drive_folder_id],
    }
    folder = service.files().create(body=folder_meta, fields="id").execute()
    folder_id = folder["id"]
    logger.info(f"Created auto_generated folder in Drive: {folder_id}")
    return folder_id


def _upload_to_drive(service, folder_id: str, file_name: str, content: str) -> str:
    """Upload a text file to a Drive folder. Returns the new file's Drive URL."""
    from googleapiclient.http import MediaIoBaseUpload
    file_meta = {"name": file_name, "parents": [folder_id]}
    media = MediaIoBaseUpload(
        io.BytesIO(content.encode("utf-8")),
        mimetype="text/plain",
        resumable=False,
    )
    uploaded = service.files().create(
        body=file_meta,
        media_body=media,
        fields="id, webViewLink",
    ).execute()
    return uploaded.get("webViewLink", "")


def generate_brand_voice(
    client_id: str,
    client_name: str,
    client_data_path: str,
    client_drive_folder_id: str,
    direct_texts: list[str] = None,
) -> dict:
    # 1. Collect source texts
    if direct_texts:
        documents_text = "\n\n".join(direct_texts)[:MAX_CHARS_FOR_EXTRACTION]
    else:
        documents_text = _read_client_texts(client_id, client_data_path)
        
    if not documents_text:
        return {
            "status": "error",
            "message": "No text documents found. Please ingest documents first.",
        }

    # 2. Generate brand voice via LLM
    logger.info(f"Generating brand voice for {client_name}...")
    result = llm_router.complete(
        prompt=BRAND_VOICE_EXTRACTION_PROMPT.format(
            client_name=client_name,
            documents_text=documents_text,
        ),
        system=BRAND_VOICE_SYSTEM,
        max_tokens=4000,
        temperature=0.3,
    )
    guide_text = result["text"]

    # 3. Save to brand_voice/ subfolder
    file_name = "brand_voice_guide.txt"
    brand_voice_dir = Path(client_data_path) / client_id / "brand_voice"
    brand_voice_dir.mkdir(parents=True, exist_ok=True)
    brand_voice_path = brand_voice_dir / file_name
    brand_voice_path.write_text(guide_text, encoding="utf-8")
    logger.info(f"Saved brand voice locally: {brand_voice_path}")

    # 4. Upload to Google Drive
    drive_url = ""
    drive_upload_success = False
    if client_drive_folder_id:
        try:
            from core.drive.sync import get_drive_service
            service = get_drive_service()
            auto_gen_folder_id = _get_or_create_auto_generated_folder(service, client_drive_folder_id)
            drive_url = _upload_to_drive(service, auto_gen_folder_id, file_name, guide_text)
            logger.info(f"✅ Uploaded brand voice to Drive: {drive_url}")
            drive_upload_success = True
        except Exception as e:
            logger.error(f"Drive upload failed: {e}")

    # 5. Clear old brand voice chunks from Qdrant
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Api key is used with an insecure connection")
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY") or None,
        )
    collection_name = f"client_{client_id}"
    
    try:
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source_folder",
                        match=MatchValue(value="brand_voice")
                    )
                ]
            )
        )
        logger.info(f"DEBUG: Cleared old brand voice chunks for {client_id}")
    except Exception as e:
        logger.info(f"DEBUG: No existing brand voice chunks to clear: {e}")

    # 6. Explicit Ingestion into Qdrant using LlamaIndex
    chunks_stored = 0
    try:
        from llama_index.core import VectorStoreIndex, StorageContext, Document
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from llama_index.embeddings.fastembed import FastEmbedEmbedding

        doc = Document(
            text=guide_text,
            metadata={
                "client_id": client_id,
                "client_name": client_name,
                "file_name": file_name,
                "source_folder": "brand_voice",
                "date_ingested": datetime.utcnow().isoformat(),
            },
            excluded_embed_metadata_keys=["date_ingested"],
        )

        node_parser = SentenceSplitter(chunk_size=384, chunk_overlap=80)
        embed_model = FastEmbedEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir="./.fastembed_cache"
        )
        vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            [doc],
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=[node_parser],
            show_progress=False,
        )
        
        chunks_stored = max(1, len(guide_text) // (384 * 4))
        logger.info(f"✅ Ingested Brand Voice Guide: ~{chunks_stored} chunks stored")
    except Exception as e:
        logger.error(f"Failed to ingest brand voice into Qdrant: {e}")

    # 7. Local backup JSON
    try:
        import json
        local_backup = {
            "client_id": client_id,
            "client_name": client_name,
            "generated_at": datetime.utcnow().isoformat(),
            "content": guide_text,
            "chunks_stored": chunks_stored,
            "source_folder": "brand_voice"
        }
        backup_path = brand_voice_dir / ".brand_voice_backup.json"
        backup_path.write_text(json.dumps(local_backup, indent=2), encoding="utf-8")
        logger.info(f"DEBUG: Local backup saved to {backup_path}")
    except Exception as e:
        logger.error(f"Failed to save local backup: {e}")

    return {
        "status": "success",
        "message": f"Brand voice generated and embedded into Qdrant",
        "file_saved": str(brand_voice_path),
        "chunks_stored": chunks_stored,
        "qdrant_collection": collection_name,
        "source_folder": "brand_voice",
        "drive_uploaded": drive_upload_success,
        "drive_url": drive_url,
    }
