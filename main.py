from __future__ import annotations
import os
import sys
import time
import json
import logging
from typing import Dict, Any, Optional

from config import Config, get_config
from retrieval.bm25 import ArtifactNotFoundError
from retrieval.hybrid_retriever import HybridRetriever
from features.feature_extractor import SAGEFeatureExtractor
from policy.predict import SAGEPolicyPredictor
from generation.context_builder import ContextBuilder
from generation.llm import SAGEGenerationManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SAGE-Runtime")


class SAGERuntimeSystem:
    """Original SAGE Runtime Pipeline.
    Implements the complete runtime inference flow using pre-built Colab artifacts:
    Query -> Query Features -> Hybrid Probe Retrieval (BM25 + Dense) -> RRF ->
    Retrieval Features -> Feature Vector -> SAGE RF Policy -> Budget k* ->
    Context Construction -> LLM Generation -> Metrics.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()

        # 1. Retrieval Layer
        retrieval_cfg = self.config.retrieval
        bm25_cfg = retrieval_cfg.get("bm25", {})
        dense_cfg = retrieval_cfg.get("dense", {})
        rrf_cfg = retrieval_cfg.get("rrf", {})

        from retrieval.bm25 import BM25Retriever
        from retrieval.dense_retriever import DenseRetriever, QueryEmbedder
        from retrieval.rrf import ReciprocalRankFusion

        bm25 = BM25Retriever(k1=bm25_cfg.get("k1", 1.5), b=bm25_cfg.get("b", 0.75))
        embedder = QueryEmbedder(
            model_name=dense_cfg.get("model_name", "BAAI/bge-m3"),
            device=dense_cfg.get("device", "cpu"),
            dimension=dense_cfg.get("embedding_dim", 1024)
        )
        dense = DenseRetriever(
            embedder=embedder,
            model_name=dense_cfg.get("model_name", "BAAI/bge-m3"),
            embedding_dim=dense_cfg.get("embedding_dim", 1024),
            device=dense_cfg.get("device", "cpu")
        )
        rrf = ReciprocalRankFusion(
            k=rrf_cfg.get("k", 60),
            bm25_weight=rrf_cfg.get("bm25_weight", 1.0),
            dense_weight=rrf_cfg.get("dense_weight", 1.0)
        )

        self.retriever = HybridRetriever(
            bm25=bm25,
            dense=dense,
            rrf=rrf,
            top_candidates=bm25_cfg.get("top_candidates", 50),
            probe_depth=retrieval_cfg.get("probe_depth", 2)
        )

        # 2. Feature Extractor
        schema_path = self.config.artifacts.get("feature_schema_path", "artifacts/metadata/feature_schema.json")
        self.feature_extractor = SAGEFeatureExtractor(schema_path=schema_path)

        # 3. Policy Predictor
        policy_cfg = self.config.policy
        model_path = os.path.join(
            self.config.artifacts.get("model_dir", "artifacts/models/random_forest"),
            "sage_rf.joblib"
        )
        self.policy_predictor = SAGEPolicyPredictor(
            model_path=model_path,
            temperature=policy_cfg.get("temperature", 1.0),
            fallback_budget=policy_cfg.get("fallback_budget", 5)
        )

        # 4. Context Builder
        gen_cfg = self.config.generation
        self.context_builder = ContextBuilder(
            max_context_tokens=gen_cfg.get("max_context_tokens", 2048)
        )

        # 5. LLM Generation Manager
        self.generator = SAGEGenerationManager(
            provider=gen_cfg.get("provider", "mock"),
            model=gen_cfg.get("model", "gpt-4o-mini"),
            temperature=gen_cfg.get("temperature", 0.2),
            max_tokens=gen_cfg.get("max_tokens", 150),
            api_key=gen_cfg.get("openai_api_key")
        )

        self.target_slo_ms = float(self.config.slo.get("target_slo_ms", 5000.0))
        self.is_initialized = False

    def load_artifacts(self) -> None:
        """Loads BM25, FAISS, and Random Forest artifacts exported from Colab."""
        art_cfg = self.config.artifacts
        bm25_dir = art_cfg.get("bm25_dir", "artifacts/indexes/bm25")
        faiss_dir = art_cfg.get("faiss_dir", "artifacts/indexes/faiss")

        logger.info("Loading retrieval index artifacts...")
        self.retriever.load_artifacts(bm25_dir=bm25_dir, faiss_dir=faiss_dir)

        logger.info("Loading trained SAGE Random Forest policy model...")
        self.policy_predictor.load_model()

        self.is_initialized = True
        logger.info("All SAGE runtime artifacts successfully loaded and validated.")

    def run_query(self, query: str, target_slo_override_ms: Optional[float] = None) -> Dict[str, Any]:
        """Executes the full SAGE inference pipeline for a single query."""
        if not self.is_initialized:
            self.load_artifacts()

        start_time = time.perf_counter()
        target_slo = target_slo_override_ms or self.target_slo_ms
        latency_breakdown: Dict[str, float] = {}

        # 1. Retrieval & Probe Extraction
        t0 = time.perf_counter()
        fused_candidates, raw_candidates = self.retriever.retrieve(
            query=query,
            top_k=50,
            return_individual=True
        )
        latency_breakdown["retrieval_ms"] = (time.perf_counter() - t0) * 1000.0

        # 2. Feature Extraction
        t1 = time.perf_counter()
        feature_vector, feature_dict = self.feature_extractor.extract_features(
            query=query,
            fused_candidates=fused_candidates[:10],
            bm25_candidates=raw_candidates.get("bm25") if raw_candidates else None,
            dense_candidates=raw_candidates.get("dense") if raw_candidates else None,
            target_slo_ms=target_slo
        )
        latency_breakdown["feature_extraction_ms"] = (time.perf_counter() - t1) * 1000.0

        # 3. Policy Prediction
        t2 = time.perf_counter()
        policy_decision = self.policy_predictor.predict_budget(feature_vector)
        selected_k = policy_decision["selected_k"]
        latency_breakdown["policy_decision_ms"] = (time.perf_counter() - t2) * 1000.0

        # 4. Context Construction
        t3 = time.perf_counter()
        context_str, selected_passages, token_estimate = self.context_builder.build_context(
            ranked_passages=fused_candidates,
            budget_k=selected_k
        )
        latency_breakdown["context_building_ms"] = (time.perf_counter() - t3) * 1000.0

        # 5. LLM Generation
        t4 = time.perf_counter()
        gen_output = self.generator.generate_response(query=query, context_string=context_str)
        latency_breakdown["generation_ms"] = (time.perf_counter() - t4) * 1000.0

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        latency_breakdown["total_ms"] = total_latency_ms

        slo_met = (total_latency_ms <= target_slo)

        result = {
            "query": query,
            "answer": gen_output.get("answer", ""),
            "selected_k": selected_k,
            "policy_confidence": policy_decision.get("confidence", 0.0),
            "passages_used": [p.get("id") for p in selected_passages],
            "estimated_context_tokens": token_estimate,
            "latency_breakdown_ms": {k: round(v, 2) for k, v in latency_breakdown.items()},
            "total_latency_ms": round(total_latency_ms, 2),
            "slo_met": slo_met,
            "target_slo_ms": target_slo,
            "features_summary": {k: round(v, 4) for k, v in feature_dict.items()}
        }
        return result


def main():
    logger.info("Initializing SAGE Runtime System...")
    try:
        system = SAGERuntimeSystem()
        system.load_artifacts()
    except ArtifactNotFoundError as e:
        logger.error(f"\n{'='*70}\n[ARTIFACT MISSING]\n{e}\n{'='*70}")
        print("\nNotice: The SAGE runtime requires pre-built artifacts exported from Google Colab.")
        print("Please follow the instructions in README.md to generate and copy artifacts.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start SAGE runtime: {e}")
        sys.exit(1)

    # Example query
    sample_query = "What is the capital of France and what is its population?"
    logger.info(f"Running inference for query: '{sample_query}'")
    response = system.run_query(sample_query)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()

