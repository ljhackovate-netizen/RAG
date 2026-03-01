from fastapi import APIRouter
from core.vector_store import vector_store_manager

router = APIRouter(prefix="/clients", tags=["clients"])


@router.get("/")
async def list_clients():
    """Return all clients that have ingested data in Qdrant."""
    client_ids = vector_store_manager.list_clients()
    return [
        {"id": c, **vector_store_manager.stats(c)}
        for c in client_ids
    ]


@router.get("/{client_id}/stats")
async def client_stats(client_id: str):
    return vector_store_manager.stats(client_id)
