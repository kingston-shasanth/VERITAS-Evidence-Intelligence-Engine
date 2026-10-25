"""
Multilingual Dense Semantic Retriever for VERITAS-X.
Uses intfloat/multilingual-e5-small dense embeddings with FAISS IndexFlatIP
for cross-lingual semantic evidence retrieval (Cosine Similarity).
Features content-hash-based automatic cache invalidation and atomic persistence.
"""
import os
import json
import logging
import hashlib
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .evidence_corpus import EvidenceCorpus, EvidenceDocument

logger = logging.getLogger("veritas.dense_retriever")

DEFAULT_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), ".cache", "huggingface")
)


class DenseRetriever:
    """
    Multilingual semantic search engine powered by multilingual-e5 and FAISS.
    Features robust content-hash validation to automatically invalidate stale
    indices whenever corpus text or structure changes.
    """

    MODEL_NAME = "intfloat/multilingual-e5-small"
    DIMENSION = 384

    def __init__(
        self,
        corpus: EvidenceCorpus,
        model_name: str = MODEL_NAME,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        index_path: Optional[str] = None,
        meta_path: Optional[str] = None,
        model: Optional[SentenceTransformer] = None,
    ):
        self.corpus = corpus
        self.model_name = model_name
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.device = device
        self.index_path = index_path
        self.meta_path = meta_path

        # Derive default index and meta paths if corpus has a disk filepath
        if not self.index_path and hasattr(self.corpus, "filepath") and self.corpus.filepath:
            corpus_base = os.path.splitext(self.corpus.filepath)[0]
            self.index_path = f"{corpus_base}.faiss"
            self.meta_path = f"{corpus_base}_faiss_meta.json"

        os.environ["HF_HOME"] = self.cache_dir
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        # Check for locally cached model snapshot to avoid unnecessary remote HF network pings
        repo_dir_name = f"models--{self.model_name.replace('/', '--')}"
        local_cache_candidates = [
            os.path.join(self.cache_dir, "hub", repo_dir_name),
            os.path.expanduser(f"~/.cache/huggingface/hub/{repo_dir_name}"),
        ]
        if any(os.path.exists(p) for p in local_cache_candidates):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir, device=self.device)

        self.index: Optional[faiss.IndexFlatIP] = None
        self.doc_ids: List[str] = []

        self._build_index()

    @staticmethod
    def compute_corpus_hash(corpus: EvidenceCorpus) -> str:
        """
        Computes a deterministic SHA-256 hash of the evidence corpus content.
        Iterates through all documents sorted by doc_id to ensure a canonical
        fingerprint sensitive to document additions, deletions, and any textual edits.
        """
        hasher = hashlib.sha256()
        docs = sorted(corpus.get_all_documents(), key=lambda d: d.doc_id)
        for doc in docs:
            hasher.update(doc.doc_id.encode("utf-8"))
            hasher.update(b"\x1f")  # Unit separator
            hasher.update(doc.text.encode("utf-8"))
            hasher.update(b"\x1f")
            hasher.update(str(doc.source_dataset or "").encode("utf-8"))
            hasher.update(b"\x1f")
            hasher.update(str(doc.source_url or "").encode("utf-8"))
            hasher.update(b"\x1e")  # Record separator
        return hasher.hexdigest()

    def _is_cache_valid(self, meta: dict, current_hash: str) -> bool:
        """Validates metadata dictionary against current configuration and corpus content."""
        if not isinstance(meta, dict):
            logger.warning("FAISS cache invalid: metadata is not a valid JSON dictionary.")
            return False

        stored_hash = meta.get("corpus_hash")
        if not stored_hash or stored_hash != current_hash:
            logger.info(
                "FAISS cache invalid: corpus content hash mismatch (stored=%s, current=%s).",
                stored_hash,
                current_hash,
            )
            return False

        if meta.get("model_name") != self.model_name:
            logger.info("FAISS cache invalid: model_name mismatch (stored=%s, current=%s).", meta.get("model_name"), self.model_name)
            return False

        if meta.get("dimension") != self.DIMENSION:
            logger.info("FAISS cache invalid: dimension mismatch (stored=%s, expected=%d).", meta.get("dimension"), self.DIMENSION)
            return False

        if meta.get("corpus_len") != len(self.doc_ids):
            logger.info("FAISS cache invalid: corpus_len mismatch (stored=%s, current=%d).", meta.get("corpus_len"), len(self.doc_ids))
            return False

        if meta.get("doc_ids") != self.doc_ids:
            logger.info("FAISS cache invalid: document ID sequence mismatch.")
            return False

        return True

    def _build_index(self):
        """Loads cached FAISS index if valid, or encodes corpus passages and caches it atomically."""
        docs = self.corpus.get_all_documents()
        self.doc_ids = [d.doc_id for d in docs]

        if not docs:
            self.index = None
            return

        current_hash = self.compute_corpus_hash(self.corpus)

        # Attempt to load pre-built index from disk cache
        if self.index_path and self.meta_path and os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if self._is_cache_valid(meta, current_hash):
                    self.load_index(self.index_path, self.meta_path)
                    logger.info("Reused valid cached FAISS index from %s (hash: %s)", self.index_path, current_hash[:12])
                    return
                else:
                    logger.info("Cached FAISS index is invalid/stale. Rebuilding from corpus...")
            except Exception as e:
                logger.warning("FAISS cache invalid or unreadable (%s). Rebuilding from corpus...", e)

        # Prepend e5 standard passage prompt
        passage_texts = [f"passage: {d.text}" for d in docs]

        # Batch encode with L2 normalization (cosine similarity = inner product)
        embeddings = self.model.encode(
            passage_texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            device=self.device,
        )
        embeddings_np = np.array(embeddings, dtype=np.float32)

        self.index = faiss.IndexFlatIP(self.DIMENSION)
        self.index.add(embeddings_np)

        # Cache built index atomically if paths are configured
        if self.index_path and self.meta_path:
            try:
                self.save_index(self.index_path, self.meta_path, corpus_hash=current_hash)
            except Exception as e:
                logger.error("Failed to persist FAISS index cache: %s", e)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Encodes query with 'query: ' prompt and performs FAISS maximum inner-product search.
        """
        if self.index is None or not query or not query.strip():
            return []

        query_text = f"query: {query.strip()}"
        query_vec = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device,
        )
        query_vec_np = np.array(query_vec, dtype=np.float32)

        actual_k = min(top_k, self.index.ntotal)
        if actual_k <= 0:
            return []

        scores, indices = self.index.search(query_vec_np, actual_k)

        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0 or idx >= len(self.doc_ids):
                continue

            doc_id = self.doc_ids[idx]
            doc = self.corpus.get_document(doc_id)
            if doc is None:
                continue

            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "rank": rank,
                "text": doc.text,
                "source_dataset": doc.source_dataset,
                "source_url": doc.source_url,
                "claim_id": doc.claim_id,
                "metadata": doc.metadata,
            })

        return results

    def save_index(self, index_file: str, meta_file: str, corpus_hash: Optional[str] = None):
        """Persists the FAISS binary index and metadata safely using atomic replacement."""
        if self.index is None:
            return

        target_dir = os.path.dirname(os.path.abspath(index_file))
        os.makedirs(target_dir, exist_ok=True)

        if corpus_hash is None:
            corpus_hash = self.compute_corpus_hash(self.corpus)

        meta_data = {
            "corpus_hash": corpus_hash,
            "doc_ids": self.doc_ids,
            "model_name": self.model_name,
            "dimension": self.DIMENSION,
            "corpus_len": len(self.doc_ids),
        }

        rnd = uuid.uuid4().hex[:8]
        tmp_index = f"{index_file}.{os.getpid()}.{rnd}.tmp"
        tmp_meta = f"{meta_file}.{os.getpid()}.{rnd}.tmp"

        try:
            # 1. Write index to temporary file
            faiss.write_index(self.index, tmp_index)
            # 2. Write metadata to temporary file
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2)

            # 3. Atomically replace target files (os.replace is atomic on both Windows & POSIX in Python 3.3+)
            os.replace(tmp_index, index_file)
            os.replace(tmp_meta, meta_file)
        except Exception as e:
            logger.error("Failed to atomically persist FAISS cache: %s. Cleaning temp files.", e)
            for p in (tmp_index, tmp_meta):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            raise

    def load_index(self, index_file: str, meta_file: str):
        """Loads a pre-built FAISS index and doc_ids list with compatibility validation."""
        try:
            loaded_index = faiss.read_index(index_file)
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if loaded_index.d != self.DIMENSION:
                raise ValueError(f"Index dimension mismatch: {loaded_index.d} != {self.DIMENSION}")
            if loaded_index.ntotal != len(data["doc_ids"]):
                raise ValueError(f"Index vector count mismatch: {loaded_index.ntotal} != {len(data['doc_ids'])}")

            self.index = loaded_index
            self.doc_ids = data["doc_ids"]
        except Exception as e:
            logger.warning("load_index failed for %s: %s. Cache marked invalid.", index_file, e)
            raise
