import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(":memory:")
client.create_collection(
    collection_name="test",
    vectors_config={"size": 4, "distance": "Cosine"}
)

import numpy as np
query_vector = np.random.rand(4).tolist()

try:
    # Try query_points
    res = client.query_points(
        collection_name="test",
        query=query_vector,
        limit=1
    )
    print(f"query_points result type: {type(res)}")
    print(f"points in result: {hasattr(res, 'points')}")
    if hasattr(res, 'points'):
        print(f"points type: {type(res.points)}")
except Exception as e:
    print(f"query_points failed: {e}")

try:
    # Try search (just to be 100% sure it fails with a clean client)
    res = client.search(
        collection_name="test",
        query_vector=query_vector,
        limit=1
    )
    print(f"search result type: {type(res)}")
except Exception as e:
    print(f"search failed: {e}")
