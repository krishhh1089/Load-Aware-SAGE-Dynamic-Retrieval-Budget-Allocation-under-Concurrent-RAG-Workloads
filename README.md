# SAGE: SLO-Load-Aware Adaptive Retrieval for Production RAG Systems
## Phase 1 — Original SAGE Runtime System

This repository implements the **local runtime system** for **SAGE (SLO-Aware Adaptive Retrieval for Production RAG Systems)**, based on the research paper by Muhammad Faizan Raza, Shuo (Luna) Yang, and Satish Mahadevan Srinivasan (Penn State, CoDIT 2026).

---

## 1. Project Architecture & Environment Separation

To ensure lightweight execution without overloading local machines, development is partitioned into two distinct environments:

```
+-----------------------------------------------------------------------------------------+
|                                ENVIRONMENT A: GOOGLE COLAB                              |
| (Heavy Data Preprocessing & Training - High Compute / A100 GPU)                         |
|                                                                                         |
|  1. Download Benchmark (Natural Questions / HotpotQA / UnSeenTimeQA)                    |
|  2. Passage Chunking & Preprocessing                                                    |
|  3. BGE-M3 Dense Embedding Generation & FAISS Index Construction                        |
|  4. BM25 Sparse Index Construction                                                      |
|  5. Exhaustive Oracle Budget Sweeps (k in {2, 3, 5, 7, 10, 15, 20, 25, 30})              |
|  6. Oracle Label Generation under SLO Deadline T = 5.0s                                 |
|  7. SAGE RandomForestClassifier Training (100 estimators, max depth 10)                 |
|  8. Validation-Set Temperature Calibration (T)                                          |
|  9. Export Artifacts -> artifacts.zip                                                   |
+-----------------------------------------------------------------------------------------+
                                           |
                                [ Export & Download ]
                                           v
+-----------------------------------------------------------------------------------------+
|                         ENVIRONMENT B: LOCAL RUNTIME APPLICATION                        |
| (Inference, Serving, Context Construction, Generation, Metrics - Local Laptop)          |
|                                                                                         |
|  1. Load Pre-Built Artifacts (BM25 Index, Dense Index, RF Model, Feature Schema)        |
|  2. Runtime Query Processing Pipeline:                                                  |
|     User Query                                                                          |
|       |--> Query Feature Extraction (lexical length, question indicators)              |
|       |--> Hybrid Retrieval Probe (BM25 + Dense)                                        |
|       |--> Reciprocal Rank Fusion (RRF with k=60)                                       |
|       |--> Retrieval Feature Extraction (top1 score, score drop s1-s2, agreement)       |
|       |--> Form SAGE Feature Vector phi(q)                                              |
|       |--> SAGE RF Policy Prediction: k* = argmax p_theta(k | phi(q))                  |
|       |--> Conservative Tie-Breaking & Temperature Scaling                              |
|       |--> Passage Selection (top-k* deduplicated passages)                             |
|       |--> Context Construction & Token Budgeting                                       |
|       |--> LLM Answer Generation (OpenAI API or Deterministic Mock)                     |
|       |--> Latency Breakdown & SLO Compliance Accounting                                |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Directory Layout & Colab Export Artifacts

The local runtime expects artifacts to be placed in the `artifacts/` directory as follows:

```
artifacts/
├── dataset/
│   └── passages.json               # Full corpus metadata (ID, title, text)
├── indexes/
│   ├── bm25/
│   │   └── bm25_index.joblib       # Pre-indexed BM25 data (IDF, doc lengths, freqs)
│   └── faiss/
│       ├── index.faiss             # Pre-built FAISS vector index (BGE-M3 embeddings)
│       ├── vectors.npy             # Dense embeddings matrix (cross-platform fallback)
│       └── passages.json           # Passage mapping matching vector order
├── models/
│   └── random_forest/
│       └── sage_rf.joblib          # Trained SAGE RandomForest model & metadata
└── metadata/
    ├── feature_schema.json         # Exact feature names, ordering, and data types
    └── model_metadata.json         # Hyperparameters, candidate budgets, temperature
```

### Colab Export Instructions

In your Google Colab training notebook, after completing index construction and Random Forest training, run the following export snippet:

```python
import os, json, joblib, numpy as np

# 1. Export BM25
os.makedirs("artifacts/indexes/bm25", exist_ok=True)
joblib.dump({
    "k1": bm25_indexer.k1,
    "b": bm25_indexer.b,
    "passages": bm25_indexer.passages,
    "doc_len": bm25_indexer.doc_len,
    "avg_dl": bm25_indexer.avg_dl,
    "doc_freqs": bm25_indexer.doc_freqs,
    "idf": bm25_indexer.idf,
    "term_freqs": bm25_indexer.term_freqs,
    "n_docs": bm25_indexer.n_docs
}, "artifacts/indexes/bm25/bm25_index.joblib")

# 2. Export Dense Index
os.makedirs("artifacts/indexes/faiss", exist_ok=True)
import faiss
faiss.write_index(faiss_index, "artifacts/indexes/faiss/index.faiss")
np.save("artifacts/indexes/faiss/vectors.npy", embeddings_matrix)
with open("artifacts/indexes/faiss/passages.json", "w", encoding="utf-8") as f:
    json.dump(passages, f)

# 3. Export Trained SAGE Random Forest Model
os.makedirs("artifacts/models/random_forest", exist_ok=True)
joblib.dump({
    "model": trained_rf_model,
    "classes": [2, 3, 5, 7, 10, 15, 20, 25, 30],
    "temperature": optimal_temperature,
    "feature_names": feature_schema["feature_names"]
}, "artifacts/models/random_forest/sage_rf.joblib")

# 4. Zip and download
!zip -r sage_artifacts.zip artifacts/
```

Extract the resulting `sage_artifacts.zip` into the root of this local repository.

---

## 3. SAGE Paper Methodology & Features

### Discrete Candidate Budgets
$$\mathcal{K} = \{2, 3, 5, 7, 10, 15, 20, 25, 30\}$$

### Hybrid Retrieval & Reciprocal Rank Fusion (RRF)
BM25 and BGE-M3 Dense candidate rankings are combined using RRF with constant $k = 60$:
$$RRF(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{w_m}{60 + \text{rank}_m(d)}$$

### Lightweight Probe Feature Vector $\phi(q)$
The runtime extracts 10 features matching `artifacts/metadata/feature_schema.json`:
1. `query_length_chars`: Total characters in query.
2. `query_length_tokens`: Total token/word count.
3. `query_has_question_mark`: Binary flag for punctuation `?`.
4. `query_avg_word_len`: Average word character length.
5. `top1_score`: Highest RRF score from probe retrieval.
6. `score_drop`: Rank gap $s_1 - s_2$ between rank 1 and 2 candidates.
7. `score_mean`: Arithmetic mean of top candidate retrieval scores.
8. `score_std`: Standard deviation of candidate scores.
9. `sparse_dense_agreement`: Jaccard overlap between top BM25 and Dense candidates.
10. `target_slo_ms`: Target latency Service Level Objective deadline in milliseconds.

### Policy Inference Decision
$$\pi_\theta(q) = \arg\max_{k \in \mathcal{K}} p_\theta(k \mid \phi(q))$$
With temperature scaling $p_\theta(k \mid \phi(q); T) = \text{softmax}(z_k / T)$ and conservative tie-breaking in favor of smaller budgets.

---

## 4. Local Installation & Setup

1. **Clone & Environment Setup**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

2. **Configure Environment Secrets**:
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   Add your `OPENAI_API_KEY` if testing real OpenAI generation. For offline execution, the system defaults to `provider: mock` in `configs/config.yaml`.

---

## 5. Running SAGE Inference

Once artifacts have been downloaded from Colab into `artifacts/`:

```bash
python main.py
```

Example output:
```json
{
  "query": "What is the capital of France and what is its population?",
  "answer": "Based on the retrieved context: Paris is the capital of France.",
  "selected_k": 3,
  "policy_confidence": 0.84,
  "passages_used": ["doc1", "doc4", "doc9"],
  "estimated_context_tokens": 142,
  "latency_breakdown_ms": {
    "retrieval_ms": 14.2,
    "feature_extraction_ms": 1.1,
    "policy_decision_ms": 0.8,
    "context_building_ms": 0.3,
    "generation_ms": 42.5,
    "total_ms": 58.9
  },
  "slo_met": true,
  "target_slo_ms": 5000.0
}
```

---

## 6. Running Unit Tests

The test suite runs with fast, small mock fixtures and does **not** download large datasets or train models locally:

```bash
python -m pytest tests/ -v
```

All 22 test cases cover:
- BM25 artifact loading, error handling, scoring.
- Dense index loading and vector inner product search.
- Reciprocal Rank Fusion ranking and tie-breaking.
- Feature extraction and schema validation.
- Temperature calibration and conservative tie-breaking.
- Policy model loading, feature validation, and prediction.
- Context builder deduplication, budget selection, and token limits.
- Mock and OpenAI LLM client integration.
- Exact Match, Recall@k, and SLO compliance evaluation.
- Full end-to-end inference pipeline execution.

---

## 7. Status & Limitations

- **Phase 1 Runtime Complete**: Retrieval, feature extraction, policy prediction, context construction, LLM generation, and evaluation interfaces are fully implemented.
- **Missing Colab Artifacts**: The full Natural Questions BM25 index, FAISS vector index, and trained Random Forest model must be generated in Google Colab and placed in `artifacts/`. The local application cleanly validates and prompts the user if artifacts are absent.
- **Phase 2 (Load-Aware Controller)**: To be implemented in the subsequent phase per project plan.

