import os
import pytest
from retrieval.bm25 import BM25Retriever, ArtifactNotFoundError


def test_bm25_missing_artifact_raises_error(tmp_path):
    empty_dir = str(tmp_path / "empty_bm25")
    os.makedirs(empty_dir, exist_ok=True)
    retriever = BM25Retriever()
    with pytest.raises(ArtifactNotFoundError):
        retriever.load_index(empty_dir)


def test_bm25_save_and_load(tmp_path):
    index_dir = str(tmp_path / "bm25")
    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.passages = [
        {"id": "doc1", "text": "Paris is the capital of France.", "title": "France"},
        {"id": "doc2", "text": "Berlin is the capital of Germany.", "title": "Germany"},
        {"id": "doc3", "text": "The European Union is located in Europe.", "title": "EU"}
    ]
    retriever.doc_len = [6, 6, 7]
    retriever.avg_dl = 6.33
    retriever.idf = {"paris": 1.2, "capital": 0.5, "france": 1.2}
    retriever.term_freqs = [
        {"paris": 1, "capital": 1, "france": 1},
        {"berlin": 1, "capital": 1, "germany": 1},
        {"european": 1, "union": 1}
    ]
    retriever.n_docs = 3

    # Save
    saved_path = BM25Retriever.save_index(retriever, index_dir)
    assert os.path.exists(saved_path)

    # Load into fresh retriever
    loaded = BM25Retriever()
    loaded.load_index(index_dir)
    assert loaded.is_loaded
    assert loaded.n_docs == 3

    # Retrieve
    results = loaded.retrieve("capital of France", top_k=2)
    assert len(results) > 0
    assert results[0]["id"] == "doc1"
    assert results[0]["score"] > 0

