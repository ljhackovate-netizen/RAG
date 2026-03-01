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
    client_drive_folder_id: str,   # The Drive folder ID to write back to
    direct_texts: list[str] = None, # Optional: direct texts to use instead of reading local files
) -> dict:
    """
    Generate brand voice guide from ingested documents and upload to Drive.
    """
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

    # 3. Save locally
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    file_name = f"brand_voice_guide_{timestamp}.txt"
    local_dir = Path(client_data_path) / client_id / "auto_generated"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / file_name
    local_path.write_text(guide_text, encoding="utf-8")
    logger.info(f"Saved brand voice locally: {local_path}")

    # 4. Upload to Google Drive (into auto_generated/ subfolder of client folder)
    drive_url = ""
    drive_error = ""
    if client_drive_folder_id:
        try:
            from core.drive.sync import get_drive_service
            service = get_drive_service()
            auto_gen_folder_id = _get_or_create_auto_generated_folder(service, client_drive_folder_id)
            drive_url = _upload_to_drive(service, auto_gen_folder_id, file_name, guide_text)
            logger.info(f"✅ Uploaded brand voice to Drive: {drive_url}")
            
            # 5. Ingest the generated brand voice into Qdrant for RAG
            try:
                from core.ingestion.pipeline import IngestionPipeline
                pipeline = IngestionPipeline()
                pipeline.ingest_file(
                    file_path_or_bytes=str(local_path),
                    client_id=client_id,
                    client_name=client_name,
                    source_folder="auto_generated"
                )
                logger.info("✅ Ingested auto-generated Brand Voice Guide into Qdrant")
            except Exception as e:
                logger.error(f"Failed to ingest brand voice into Qdrant: {e}")
                
        except Exception as e:
            drive_error = str(e)
            logger.error(f"Drive upload failed: {e}")

    return {
        "status": "success",
        "file_name": file_name,
        "local_path": str(local_path),
        "drive_url": drive_url,
        "drive_error": drive_error,
        "provider": result["provider"],
        "client_id": client_id,
    }
