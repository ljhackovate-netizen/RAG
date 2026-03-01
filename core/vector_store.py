"""
Qdrant vector store. One collection per client.
Collection name format: "client_{sanitized_client_id}"
All retrieval is strictly filtered by client_id payload — never mix clients.
"""
import os
import logging
import warnings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PayloadSchemaType
)
from core.embeddings import VECTOR_SIZE

logger = logging.getLogger(__name__)


def _sanitize(client_id: str) -> str:
    return client_id.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


class VectorStoreManager:
    def __init__(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Api key is used with an insecure connection")
            self.client = QdrantClient(
                url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                api_key=os.getenv("QDRANT_API_KEY") or None,
                timeout=30,
            )

    def collection_name(self, client_id: str) -> str:
        return f"client_{_sanitize(client_id)}"

    def ensure_collection(self, client_id: str) -> str:
        name = self.collection_name(client_id)
        existing = {c.name for c in self.client.get_collections().collections}
        if name not in existing:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            # Payload indexes for fast metadata filtering
            for field in ("client_id", "source_folder", "file_name"):
                self.client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            logger.info(f"Created Qdrant collection: {name}")
        return name

    def stats(self, client_id: str) -> dict:
        name = self.collection_name(client_id)
        try:
            info = self.client.get_collection(name)
            return {"chunks": info.points_count, "status": info.status.value, "name": name}
        except Exception:
            return {"chunks": 0, "status": "not_found", "name": name}

    def delete(self, client_id: str):
        name = self.collection_name(client_id)
        try:
            self.client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")
        except Exception as e:
            logger.warning(f"Could not delete {name}: {e}")

    def list_clients(self) -> list[str]:
        return [
            c.name.removeprefix("client_")
            for c in self.client.get_collections().collections
            if c.name.startswith("client_")
        ]


vector_store_manager = VectorStoreManager()
