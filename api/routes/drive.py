import os
import io
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

# LlamaIndex ONLY — no LangChain
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from core.ingestion.extractors import extract_text
from core.ingestion.chunker import get_chunk_config
from core.vector_store import vector_store_manager

router = APIRouter(prefix="/drive", tags=["drive"])
templates = Jinja2Templates(directory="templates")

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive"]

SUPPORTED_MIME = {
    "text/plain": ".txt",
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "text/csv": ".csv",
    "application/vnd.google-apps.document": ".txt",
    "application/vnd.google-apps.spreadsheet": ".csv",
}

EMBED_MODEL = FastEmbedEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    cache_dir="./.fastembed_cache",
    threads=2
)
VECTOR_SIZE = 384


def get_drive_service():
    sa_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "./secrets/cgn-service-account.json")
    if not Path(sa_path).exists():
        raise FileNotFoundError(f"Service account JSON not found at: {sa_path}")
    creds = service_account.Credentials.from_service_account_file(sa_path, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


@router.get("/", response_class=HTMLResponse)
async def drive_page(request: Request):
    return templates.TemplateResponse("drive.html", {"request": request})


@router.get("/connect")
async def connect_drive():
    try:
        service = get_drive_service()
        about = service.about().get(fields="user").execute()
        return {
            "connected": True,
            "email": about["user"]["emailAddress"]
        }
    except FileNotFoundError as e:
        return {"connected": False, "error": str(e)}
    except Exception as e:
        return {"connected": False, "error": f"Connection failed: {str(e)}"}


@router.get("/folders")
async def list_folders():
    service = get_drive_service()
    results = service.files().list(
        q="mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id, name, parents)",
        pageSize=200
    ).execute()
    folders = results.get("files", [])
    folder_map = {f["id"]: f["name"] for f in folders}
    enriched = []
    for f in folders:
        parent_id = (f.get("parents") or [None])[0]
        parent_name = folder_map.get(parent_id, "My Drive")
        enriched.append({
            "id": f["id"],
            "name": f["name"],
            "path": f"{parent_name} / {f['name']}",
        })
    return {"folders": enriched}


@router.get("/folders/{folder_id}/contents")
async def folder_contents(folder_id: str):
    service = get_drive_service()
    sf_q = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    subfolders = service.files().list(q=sf_q, fields="files(id,name)").execute().get("files", [])
    file_q = f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
    files = service.files().list(q=file_q, fields="files(id,name,mimeType,size,modifiedTime)").execute().get("files", [])
    total = len(files)
    for sf in subfolders:
        q = f"'{sf['id']}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
        total += len(service.files().list(q=q, fields="files(id)").execute().get("files", []))
    return {"subfolders": subfolders, "files": files, "total_files": total}


def _download_file(service, file_info: dict, dest_path: Path) -> bool:
    mime = file_info.get("mimeType", "")
    try:
        if mime == "application/vnd.google-apps.document":
            req = service.files().export_media(fileId=file_info["id"], mimeType="text/plain")
        elif mime == "application/vnd.google-apps.spreadsheet":
            req = service.files().export_media(fileId=file_info["id"], mimeType="text/csv")
        else:
            req = service.files().get_media(fileId=file_info["id"])
        buf = io.BytesIO()
        dl = MediaIoBaseDownload(buf, req, chunksize=1024 * 1024)
        done = False
        while not done:
            _, done = dl.next_chunk()
        dest_path.write_bytes(buf.getvalue())
        return True
    except Exception as e:
        logger.error(f"Download failed {file_info['name']}: {e}")
        return False


def _ingest_file_llamaindex(
    file_path: str,
    client_id: str,
    client_name: str,
    source_folder: str,
    qdrant_client: QdrantClient,
) -> int:
    source_folder = source_folder.lower()
    """
    Ingest a single file into Qdrant using LlamaIndex ONLY.
    Returns number of chunks stored.
    """
    # Extract text
    raw_text = extract_text(file_path)
    if not raw_text or len(raw_text.strip()) < 30:
        return 0

    # Create LlamaIndex Document with metadata
    doc = Document(
        text=raw_text,
        metadata={
            "client_id": client_id,
            "client_name": client_name,
            "file_name": Path(file_path).name,
            "source_folder": source_folder,
            "date_ingested": datetime.utcnow().isoformat(),
        },
        excluded_embed_metadata_keys=["date_ingested"],
    )

    # Get chunk config (adaptive, no domain keywords)
    chunk_cfg = get_chunk_config(file_path, source_folder)

    # LlamaIndex node parser
    node_parser = SentenceSplitter(
        chunk_size=chunk_cfg.chunk_size,
        chunk_overlap=chunk_cfg.chunk_overlap,
    )

    # Ensure Qdrant collection exists
    collection_name = vector_store_manager.ensure_collection(client_id)

    # LlamaIndex QdrantVectorStore
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index — this embeds and stores via LlamaIndex
    index = VectorStoreIndex.from_documents(
        [doc],
        storage_context=storage_context,
        embed_model=EMBED_MODEL,
        transformations=[node_parser],
        show_progress=False,
    )

    # Estimate chunks stored
    estimated_chunks = max(1, len(raw_text) // (chunk_cfg.chunk_size * 4))
    return estimated_chunks


@router.get("/sync-and-ingest-stream")
async def sync_and_ingest_stream(
    folder_id: str,
    client_id: str,
    client_name: str,
):
    """
    SSE endpoint: streams real-time progress during Drive sync + LlamaIndex ingestion.
    """
    async def event_stream():
        client_data_path = os.getenv("CLIENT_DATA_PATH", "./client_data")
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY") or None,
        )

        try:
            service = await asyncio.to_thread(get_drive_service)

            # ── PHASE 1: COLLECT ALL FILES TO DOWNLOAD ──────────────────
            all_files_to_download = []  # list of (file_info, local_path, subfolder_name)

            # Root-level files
            file_q = f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
            root_files = await asyncio.to_thread(
                lambda: service.files().list(q=file_q, fields="files(id,name,mimeType,size,modifiedTime)").execute().get("files", [])
            )
            for f in root_files:
                if f.get("mimeType") in SUPPORTED_MIME:
                    ext = SUPPORTED_MIME[f["mimeType"]]
                    stem = Path(f["name"]).stem
                    local_name = f["name"] if f["name"].endswith(ext) else stem + ext
                    local_path = Path(client_data_path) / client_id / local_name
                    all_files_to_download.append((f, local_path, "root"))

            # Subfolder files
            sf_q = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            subfolders = await asyncio.to_thread(
                lambda: service.files().list(q=sf_q, fields="files(id,name)").execute().get("files", [])
            )

            for sf in subfolders:
                if sf["name"].lower() == "auto_generated":
                    continue
                sf_file_q = f"'{sf['id']}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
                sf_files = await asyncio.to_thread(
                    lambda q=sf_file_q: service.files().list(q=q, fields="files(id,name,mimeType,size,modifiedTime)").execute().get("files", [])
                )
                for f in sf_files:
                    if f.get("mimeType") in SUPPORTED_MIME:
                        ext = SUPPORTED_MIME[f["mimeType"]]
                        stem = Path(f["name"]).stem
                        local_name = f["name"] if f["name"].endswith(ext) else stem + ext
                        local_path = Path(client_data_path) / client_id / sf["name"] / local_name
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        all_files_to_download.append((f, local_path, sf["name"]))

            total_files = len(all_files_to_download)

            # ── PHASE 2: DOWNLOAD ────────────────────────────────────────
            downloaded = []
            for i, (file_info, local_path, subfolder_name) in enumerate(all_files_to_download):
                ok = await asyncio.to_thread(_download_file, service, file_info, local_path)
                if ok:
                    downloaded.append((str(local_path), subfolder_name))
                yield f"data: {json.dumps({'step': 'downloading', 'file': file_info['name'], 'progress': i + 1, 'total': total_files, 'ok': ok})}\n\n"
                await asyncio.sleep(0.01)

            # ── PHASE 3: PARALLEL INGEST & BRAND VOICE ──────────────────
            total_chunks = 0
            
            # Start Brand Voice generation as a background task so it runs concurrently
            async def run_bv():
                try:
                    all_texts = []
                    for file_path, _ in downloaded:
                        text = await asyncio.to_thread(extract_text, file_path)
                        if text and len(text.strip()) > 30:
                            segment = f"=== {Path(file_path).name} ===\n{text[:3000]}"
                            all_texts.append(segment)
                    if not all_texts:
                        return {"status": "error", "message": "No text found to generate Brand Voice."}
                        
                    from core.generation.brand_voice_extractor import generate_brand_voice
                    res = await asyncio.to_thread(
                        generate_brand_voice,
                        client_id=client_id,
                        client_name=client_name,
                        client_data_path=client_data_path,
                        client_drive_folder_id=folder_id,
                        direct_texts=all_texts
                    )
                    return res
                except Exception as e:
                    return {"status": "error", "message": str(e)}

            bv_task = asyncio.create_task(run_bv())
            yield f"data: {json.dumps({'step': 'brand_voice', 'message': 'Generating Brand Voice concurrently...'})}\n\n"

            # Meanwhile, do the chunking via LlamaIndex
            for i, (file_path, subfolder_name) in enumerate(downloaded):
                fname = Path(file_path).name
                try:
                    chunks = await asyncio.to_thread(
                        _ingest_file_llamaindex,
                        file_path, client_id, client_name,
                        subfolder_name, qdrant_client
                    )
                    total_chunks += chunks
                    yield f"data: {json.dumps({'step': 'ingesting', 'file': fname, 'progress': i + 1, 'total': len(downloaded), 'chunks': chunks})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'step': 'ingesting', 'file': fname, 'progress': i + 1, 'total': len(downloaded), 'error': str(e)})}\n\n"
                await asyncio.sleep(0.01)

            # ── COMPLETE INGESTION ─────────────────────────────────────────────────
            yield f"data: {json.dumps({'step': 'complete', 'files_processed': len(downloaded), 'chunks_stored': total_chunks})}\n\n"

            # Save sync status locally
            status_path = Path(client_data_path) / client_id / ".sync_status.json"
            status_path.write_text(json.dumps({
                "last_sync": datetime.utcnow().isoformat() + "Z",
                "drive_folder_id": folder_id,
                "files_downloaded": len(downloaded),
            }, indent=2))
            
            # Wait for Brand Voice if still running
            if bv_task:
                bv_result = await bv_task
                if bv_result.get("status") == "success":
                    msg = "✅ Auto-generated Brand Voice, saved to Drive, and ingested to Qdrant!"
                    yield f"data: {json.dumps({'step': 'brand_voice_complete', 'message': msg})}\n\n"
                else:
                    if bv_result.get("message") == "No text found to generate Brand Voice.":
                        yield f"data: {json.dumps({'step': 'brand_voice_skip', 'message': bv_result.get('message')})}\n\n"
                    else:
                        yield f"data: {json.dumps({'step': 'brand_voice_error', 'message': bv_result.get('message', 'Unknown error')})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'step': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )

@router.get("/status/{client_id}")
async def sync_status(client_id: str):
    import json
    from pathlib import Path
    p = Path(os.getenv("CLIENT_DATA_PATH", "./client_data")) / client_id / ".sync_status.json"
    if not p.exists():
        return {"status": "never_synced"}
    return json.loads(p.read_text())

@router.post("/generate-brand-voice/{client_id}")
async def generate_brand_voice(client_id: str, client_name: str, drive_folder_id: str = ""):
    """Generate brand voice doc and upload it back to Drive"""
    client_data_path = os.getenv("CLIENT_DATA_PATH", "./client_data")
    
    # Auto-resolve drive_folder_id if empty
    if not drive_folder_id:
        import json
        from pathlib import Path
        status_path = Path(client_data_path) / client_id / ".sync_status.json"
        if status_path.exists():
            try:
                status_data = json.loads(status_path.read_text())
                drive_folder_id = status_data.get("drive_folder_id", "")
            except Exception:
                pass
                
    if not drive_folder_id:
        raise HTTPException(
            status_code=400, 
            detail="Could not determine Drive Folder ID. Please sync the client from Drive first."
        )

    from core.generation.brand_voice_extractor import generate_brand_voice as _gen
    result = _gen(client_id, client_name, client_data_path, drive_folder_id)
    return result
