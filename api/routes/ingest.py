from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
import os, shutil, tempfile
from pathlib import Path

from core.ingestion.pipeline import ingestion_pipeline
from core.vector_store import vector_store_manager

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/file")
async def ingest_file(
    client_id: str = Form(...),
    client_name: str = Form(...),
    source_folder: str = Form("uploads"),
    file: UploadFile = File(...),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        return ingestion_pipeline.ingest_file(tmp_path, client_id, client_name, source_folder)
    finally:
        os.unlink(tmp_path)


@router.post("/folder")
async def ingest_folder(
    background_tasks: BackgroundTasks,
    client_id: str = Form(...),
    client_name: str = Form(...),
    folder_path: str = Form(...),
):
    if not Path(folder_path).exists():
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")
    background_tasks.add_task(ingestion_pipeline.ingest_folder, folder_path, client_id, client_name)
    return {"status": "ingestion_started", "client_id": client_id}


@router.get("/stats/{client_id}")
async def stats(client_id: str):
    return vector_store_manager.stats(client_id)


@router.delete("/client/{client_id}")
async def clear_client(client_id: str):
    vector_store_manager.delete(client_id)
    return {"status": "cleared", "client_id": client_id}
