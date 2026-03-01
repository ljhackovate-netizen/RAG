"""
FastEmbed embeddings — ONNX runtime, zero torch dependency.
Model: BAAI/bge-small-en-v1.5 → 384 dimensions
Downloads ~45MB on first run, cached in .fastembed_cache/
"""
from llama_index.embeddings.fastembed import FastEmbedEmbedding

VECTOR_SIZE = 384
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def get_embed_model():
    return FastEmbedEmbedding(
        model_name=EMBED_MODEL,
        cache_dir="./.fastembed_cache",
    )


embed_model = get_embed_model()
