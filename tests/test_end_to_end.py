import os
import pytest
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from config import Config
from main import SAGERuntimeSystem
from retrieval.bm25 import BM25Retriever
from retrieval.dense_retriever import DenseRetriever


def test_full_pipeline_end_to_end(tmp_path):
    # 1. Setup mock directories
    bm25_dir = str(tmp_path / "artifacts" / "indexes" / "bm25")
    faiss_dir = str(tmp_path / "artifacts" / "indexes" / "faiss")
    model_dir = str(tmp_path / "artifacts" / "models" / "random_forest")
    meta_dir = str(tmp_path / "artifacts" / "metadata")
    os.makedirs(bm25_dir, exist_ok=True)
    os.makedirs(faiss_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # 2. Create BM25 fixture
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.passages = [
        {"id": "doc1", "text": "Paris is the capital of France.", "title": "France"},
        {"id": "doc2", "text": "Berlin is the capital of Germany.", "title": "Germany"},
        {"id": "doc3", "text": "Rome is the capital of Italy.", "title": "Italy"}
    ]
    bm25.doc_len = [6, 6, 6]
    bm25.avg_dl = 6.0
    bm25.idf = {"paris": 1.5, "capital": 0.5, "france": 1.5, "berlin": 1.5, "germany": 1.5}
    bm25.term_freqs = [
        {"paris": 1, "capital": 1, "france": 1},
        {"berlin": 1, "capital": 1, "germany": 1},
        {"rome": 1, "capital": 1, "italy": 1}
    ]
    bm25.n_docs = 3
    BM25Retriever.save_index(bm25, bm25_dir)

    # 3. Create Dense fixture
    dim = 64
    vectors = np.random.randn(3, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    DenseRetriever.save_index(vectors, bm25.passages, faiss_dir)

    # 4. Create Random Forest fixture
    n_features = 10
    classes = [2, 5, 10]
    X_train = np.random.randn(20, n_features)
    y_train = np.random.choice(classes, size=20)
    rf = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump({"model": rf, "classes": classes, "temperature": 1.0}, os.path.join(model_dir, "sage_rf.joblib"))

    # 5. Build Config
    cfg_dict = {
        "artifacts": {
            "bm25_dir": bm25_dir,
            "faiss_dir": faiss_dir,
            "model_dir": model_dir,
            "feature_schema_path": "artifacts/metadata/feature_schema.json"
        },
        "retrieval": {
            "bm25": {"k1": 1.5, "b": 0.75, "top_candidates": 10},
            "dense": {"model_name": "BAAI/bge-m3", "embedding_dim": dim, "device": "cpu"},
            "rrf": {"k": 60, "bm25_weight": 1.0, "dense_weight": 1.0},
            "probe_depth": 2
        },
        "policy": {
            "candidate_budgets": classes,
            "fallback_budget": 2,
            "temperature": 1.0
        },
        "generation": {
            "provider": "mock",
            "model": "mock-llm",
            "max_context_tokens": 1000
        },
        "slo": {
            "target_slo_ms": 5000.0
        }
    }
    config = Config(cfg_dict)

    # 6. Execute full pipeline
    system = SAGERuntimeSystem(config=config)
    system.load_artifacts()
    assert system.is_initialized

    result = system.run_query("What is the capital of France?")
    assert result["query"] == "What is the capital of France?"
    assert result["selected_k"] in classes
    assert len(result["answer"]) > 0
    assert result["total_latency_ms"] > 0
    assert "retrieval_ms" in result["latency_breakdown_ms"]
    assert "policy_decision_ms" in result["latency_breakdown_ms"]
    assert "generation_ms" in result["latency_breakdown_ms"]
