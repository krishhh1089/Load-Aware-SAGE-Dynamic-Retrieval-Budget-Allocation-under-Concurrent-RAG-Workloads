from __future__ import annotations
import os
import json
from typing import List, Dict, Any, Optional
import numpy as np

from retrieval.bm25 import ArtifactNotFoundError

# On Windows, faiss-cpu SWIG bindings can trigger C-level STATUS_INVALID_HANDLE (0xc0000008).
# Therefore, FAISS is only imported if explicitly enabled via USE_FAISS=1 or in compatible environments.
FAISS_AVAILABLE = False
if os.environ.get("USE_FAISS", "0") == "1":
    try:
        import faiss
        FAISS_AVAILABLE = True
    except BaseException:
        FAISS_AVAILABLE = False


class QueryEmbedder:
    """Generates embeddings for user queries.
    Uses SentenceTransformer (e.g. BAAI/bge-m3 as specified in SAGE paper)
    or falls back to a deterministic projection if offline or testing.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu", dimension: int = 1024):
        self.model_name = model_name
        self.device = device
        self.dimension = dimension
        self._model = None
        self._is_initialized = False

    def initialize(self) -> None:
        if self._is_initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = self._model.get_sentence_embedding_dimension()
            self._is_initialized = True
        except Exception:
            self._is_initialized = True
            self._model = None

    def embed_query(self, query: str) -> np.ndarray:
        if not self._is_initialized:
            self.initialize()

        if self._model is not None:
            vec = self._model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
            return vec.astype(np.float32).reshape(1, -1)

        # Deterministic lightweight hash-projection fallback for offline testing
        rng = np.random.RandomState(abs(hash(query)) % (2**31))
        vec = rng.randn(1, self.dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class DenseRetriever:
    """Production Dense runtime retriever.
    Loads pre-built FAISS index or serialized vector matrix exported from Google Colab.
    """

    def __init__(
        self,
        embedder: Optional[QueryEmbedder] = None,
        model_name: str = "BAAI/bge-m3",
        embedding_dim: int = 1024,
        device: str = "cpu"
    ):
        self.embedder = embedder or QueryEmbedder(model_name=model_name, device=device, dimension=embedding_dim)
        self.passages: List[Dict[str, Any]] = []
        self.faiss_index = None
        self.vectors_matrix: Optional[np.ndarray] = None
        self.is_loaded: bool = False

    def load_index(self, index_dir: str = "artifacts/indexes/faiss") -> None:
        """Loads FAISS index artifact and passage mapping exported from Colab."""
        if not os.path.exists(index_dir):
            raise ArtifactNotFoundError(
                f"Dense index directory not found at: '{index_dir}'. "
                f"Please export the dense index and passage metadata from Google Colab into this directory."
            )

        metadata_file = os.path.join(index_dir, "passages.json")
        if not os.path.exists(metadata_file):
            raise ArtifactNotFoundError(
                f"Passage metadata file 'passages.json' not found in '{index_dir}'. "
                f"Export passage metadata corresponding to the index vectors."
            )

        with open(metadata_file, "r", encoding="utf-8") as f:
            self.passages = json.load(f)

        faiss_file = os.path.join(index_dir, "index.faiss")
        npy_file = os.path.join(index_dir, "vectors.npy")

        # 1. Try loading FAISS if available and index.faiss exists
        if FAISS_AVAILABLE and os.path.exists(faiss_file):
            try:
                import faiss
                self.faiss_index = faiss.read_index(faiss_file)
                self.is_loaded = True
                return
            except Exception as e:
                print(f"Warning: Failed to load FAISS index via faiss library ({e}). Falling back to vector matrix.")

        # 2. If vectors.npy exists, load as NumPy matrix
        if os.path.exists(npy_file):
            self.vectors_matrix = np.load(npy_file)
            self.is_loaded = True
            return

        # 3. If neither can be loaded, raise appropriate ArtifactNotFoundError
        if not os.path.exists(faiss_file) and not os.path.exists(npy_file):
            raise ArtifactNotFoundError(
                f"Neither 'index.faiss' nor 'vectors.npy' found in '{index_dir}'. "
                f"Please export the index from Google Colab."
            )

        # 4. If index.faiss exists but FAISS C++ library is not enabled/working
        if os.path.exists(faiss_file) and not FAISS_AVAILABLE:
            raise RuntimeError(
                f"FAISS index found at '{faiss_file}', but FAISS is disabled or incompatible on this OS. "
                f"Colab export script exports both 'index.faiss' and 'vectors.npy' for full cross-platform compatibility."
            )

        self.is_loaded = True

    def retrieve(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """Retrieve top_k passages via dense vector similarity."""
        if not self.is_loaded:
            raise RuntimeError("Dense index is not loaded. Call load_index() before retrieve().")

        q_vec = self.embedder.embed_query(query)

        # If FAISS index is loaded
        if self.faiss_index is not None:
            effective_k = min(top_k, self.faiss_index.ntotal, len(self.passages))
            if effective_k == 0:
                return []
            distances, indices = self.faiss_index.search(q_vec, effective_k)
            results = []
            for rank, (doc_idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
                if doc_idx < 0 or doc_idx >= len(self.passages):
                    continue
                p = self.passages[doc_idx]
                results.append({
                    "id": p.get("id", f"doc_{doc_idx}"),
                    "text": p.get("text", p.get("content", "")),
                    "title": p.get("title", ""),
                    "score": float(score),
                    "rank": rank,
                    "retriever": "dense",
                    "metadata": p.get("metadata", {})
                })
            return results

        # Fallback via vector matrix inner product
        if self.vectors_matrix is not None:
            effective_k = min(top_k, len(self.vectors_matrix), len(self.passages))
            if effective_k == 0:
                return []
            scores = np.dot(self.vectors_matrix, q_vec.T).squeeze()
            if scores.ndim == 0:
                scores = np.array([float(scores)])
            top_indices = np.argsort(scores)[::-1][:effective_k]

            results = []
            for rank, idx in enumerate(top_indices, start=1):
                p = self.passages[idx]
                results.append({
                    "id": p.get("id", f"doc_{idx}"),
                    "text": p.get("text", p.get("content", "")),
                    "title": p.get("title", ""),
                    "score": float(scores[idx]),
                    "rank": rank,
                    "retriever": "dense",
                    "metadata": p.get("metadata", {})
                })
            return results

        return []

    @classmethod
    def save_index(cls, vectors: np.ndarray, passages: List[Dict[str, Any]], output_dir: str) -> None:
        """Utility for test fixtures or export scripts."""
        os.makedirs(output_dir, exist_ok=True)
        # Always save vectors.npy for maximum compatibility
        np.save(os.path.join(output_dir, "vectors.npy"), vectors.astype(np.float32))

        # Also save index.faiss if faiss is enabled
        if FAISS_AVAILABLE:
            try:
                import faiss
                d = vectors.shape[1]
                idx = faiss.IndexFlatIP(d)
                idx.add(vectors.astype(np.float32))
                faiss.write_index(idx, os.path.join(output_dir, "index.faiss"))
            except Exception:
                pass

        with open(os.path.join(output_dir, "passages.json"), "w", encoding="utf-8") as f:
            json.dump(passages, f, indent=2)
