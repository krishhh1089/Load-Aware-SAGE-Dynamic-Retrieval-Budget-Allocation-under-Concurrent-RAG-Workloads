import os
import pytest
import numpy as np

from retrieval.dense_retriever import DenseRetriever, QueryEmbedder
from retrieval.bm25 import ArtifactNotFoundError


def test_dense_missing_artifact_raises_error(tmp_path):
    empty_dir = str(tmp_path / "empty_faiss")
    os.makedirs(empty_dir, exist_ok=True)
    dense = DenseRetriever()
    with pytest.raises(ArtifactNotFoundError):
        dense.load_index(empty_dir)


def test_dense_save_and_load(tmp_path):
    index_dir = str(tmp_path / "faiss")
    d = 64
    num_vectors = 4

    vectors = np.random.randn(num_vectors, d).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    passages = [
        {"id": f"p_{i}", "text": f"Passage {i} text content", "title": f"Title {i}"}
        for i in range(num_vectors)
    ]

    DenseRetriever.save_index(vectors, passages, index_dir)
    assert os.path.exists(os.path.join(index_dir, "passages.json"))
    assert os.path.exists(os.path.join(index_dir, "vectors.npy"))

    # Load and test
    embedder = QueryEmbedder(dimension=d)
    dense = DenseRetriever(embedder=embedder, embedding_dim=d)
    dense.load_index(index_dir)
    assert dense.is_loaded

    results = dense.retrieve("query about artificial intelligence", top_k=2)
    assert len(results) == 2
    assert "score" in results[0]
    assert results[0]["retriever"] == "dense"
