"""
Comprehensive Unit & Integration Test Suite for VERITAS-X Retrieval Engine.
Covers empty queries, multilingual queries, irrelevant evidence, missing source text,
BM25, Dense E5 + FAISS, Hybrid RRF, and evaluation metrics.
"""
import unittest
from unittest import mock
import os
import sys
import json
import tempfile

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Enforce offline execution and eliminate remote HF Hub network pings during tests
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from veritas.retrieval.tokenizer import MultilingualTokenizer
from veritas.retrieval.evidence_corpus import EvidenceCorpus, EvidenceDocument
from veritas.retrieval.bm25_retriever import BM25Retriever
from veritas.retrieval.dense_retriever import DenseRetriever
from veritas.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from veritas.retrieval.evaluation import RetrievalEvaluator, RetrievalMetrics


class TestRetrievalEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a small, self-contained multi-lingual diagnostic corpus for testing
        cls.test_corpus = EvidenceCorpus()
        
        docs = [
            EvidenceDocument(
                doc_id="doc_hi_01",
                text="विश्व स्वास्थ्य संगठन (WHO) ने 27 मार्च 2014 को भारत को आधिकारिक रूप से पोलियो मुक्त घोषित किया।",
                source_dataset="MULTILINGUAL_DIAGNOSTIC",
                claim_id="claim_hi_01",
                metadata={"lang": "hi"}
            ),
            EvidenceDocument(
                doc_id="doc_bn_01",
                text="রবীন্দ্রনাথ ঠাকুর ১৯১৩ সালে গীতাঞ্জলি কাব্যগ্রন্থের জন্য সাহিত্যে নোবেল পুরস্কার লাভ করেন।",
                source_dataset="MULTILINGUAL_DIAGNOSTIC",
                claim_id="claim_bn_01",
                metadata={"lang": "bn"}
            ),
            EvidenceDocument(
                doc_id="doc_hinglish_01",
                text="On May 19, 2023, RBI announced withdrawal of 2,000 rupee notes from circulation under Clean Note Policy.",
                source_dataset="MULTILINGUAL_DIAGNOSTIC",
                claim_id="claim_hinglish_01",
                metadata={"lang": "hi-en"}
            ),
            EvidenceDocument(
                doc_id="doc_en_01",
                text="India landed Chandrayaan-3 near the south pole of the Moon on August 23, 2023, becoming the fourth nation to do so.",
                source_dataset="MULTILINGUAL_DIAGNOSTIC",
                claim_id="claim_en_01",
                metadata={"lang": "en"}
            ),
            EvidenceDocument(
                doc_id="doc_distractor_01",
                text="Apple announced the launch of new iPhone models during their annual hardware keynote in California.",
                source_dataset="DISTRACTOR",
                claim_id="claim_tech_01",
                metadata={"lang": "en"}
            ),
        ]
        for d in docs:
            cls.test_corpus.add_document(d)

        cls.bm25 = BM25Retriever(cls.test_corpus)
        cls.dense = DenseRetriever(cls.test_corpus, device="cpu")
        cls.hybrid = HybridRetriever(cls.bm25, cls.dense, rrf_k=60)

    # ---------------------------------------------------------
    # 1. Edge Case: Empty & Whitespace Queries
    # ---------------------------------------------------------
    def test_empty_and_whitespace_queries(self):
        for empty_q in ["", "   ", "\t\n  \r", None]:
            bm25_res = self.bm25.search(empty_q, top_k=5)
            self.assertEqual(bm25_res, [], f"BM25 failed on empty query: {repr(empty_q)}")

            dense_res = self.dense.search(empty_q, top_k=5)
            self.assertEqual(dense_res, [], f"Dense failed on empty query: {repr(empty_q)}")

            hybrid_res = self.hybrid.search(empty_q, top_k=5)
            self.assertEqual(hybrid_res, [], f"Hybrid failed on empty query: {repr(empty_q)}")

    # ---------------------------------------------------------
    # 2. Edge Case: Out-of-Vocabulary / Irrelevant Queries
    # ---------------------------------------------------------
    def test_irrelevant_out_of_vocabulary_queries(self):
        weird_query = "zzqxkj9999000nonexistentterm"
        
        # BM25 should return zero score or empty list
        bm25_res = self.bm25.search(weird_query, top_k=3)
        for r in bm25_res:
            self.assertEqual(r["score"], 0.0)

        # Dense will compute embedding cosine similarities, but should run without error
        dense_res = self.dense.search(weird_query, top_k=3)
        self.assertIsInstance(dense_res, list)
        self.assertLessEqual(len(dense_res), 3)

        # Hybrid will gracefully fuse without raising exception
        hybrid_res = self.hybrid.search(weird_query, top_k=3)
        self.assertIsInstance(hybrid_res, list)

    # ---------------------------------------------------------
    # 3. Multilingual Query Retrieval
    # ---------------------------------------------------------
    def test_multilingual_queries(self):
        # Hindi query
        hi_results = self.hybrid.search("भारत में पोलियो उन्मूलन", top_k=1)
        self.assertTrue(len(hi_results) > 0)
        self.assertEqual(hi_results[0].doc_id, "doc_hi_01")

        # Bengali query
        bn_results = self.hybrid.search("রবীন্দ্রনাথ ঠাকুর নোবেল পুরস্কার", top_k=1)
        self.assertTrue(len(bn_results) > 0)
        self.assertEqual(bn_results[0].doc_id, "doc_bn_01")

        # Hinglish Code-Mixed query
        hinglish_results = self.hybrid.search("RBI 2000 rupee note withdrawal circulate", top_k=1)
        self.assertTrue(len(hinglish_results) > 0)
        self.assertEqual(hinglish_results[0].doc_id, "doc_hinglish_01")

    # ---------------------------------------------------------
    # 4. Dates & Numerical Tokens Preservation
    # ---------------------------------------------------------
    def test_date_and_number_tokenization_and_search(self):
        tokens = MultilingualTokenizer.tokenize("On 2023-08-23, inflation rose by 10.5% with ₹2000 loss.")
        self.assertIn("2023-08-23", tokens)
        self.assertIn("10.5%", tokens)
        self.assertIn("₹2000", tokens)

        # Query by exact date
        date_res = self.bm25.search("August 23, 2023 lunar landing", top_k=1)
        self.assertTrue(len(date_res) > 0)
        self.assertEqual(date_res[0]["doc_id"], "doc_en_01")

    # ---------------------------------------------------------
    # 5. Missing Source Text Handling in Corpus
    # ---------------------------------------------------------
    def test_missing_source_text_rejected(self):
        corpus = EvidenceCorpus()
        
        # Valid doc
        doc_valid = EvidenceDocument(doc_id="d1", text="Valid text passage.", source_dataset="TEST")
        self.assertTrue(corpus.add_document(doc_valid))

        # Invalid docs (empty or whitespace only)
        doc_empty = EvidenceDocument(doc_id="d2", text="", source_dataset="TEST")
        self.assertFalse(corpus.add_document(doc_empty))

        doc_spaces = EvidenceDocument(doc_id="d3", text="   \t\n  ", source_dataset="TEST")
        self.assertFalse(corpus.add_document(doc_spaces))

        doc_none = EvidenceDocument(doc_id="d4", text=None, source_dataset="TEST") # type: ignore
        self.assertFalse(corpus.add_document(doc_none))

        self.assertEqual(len(corpus), 1)

    # ---------------------------------------------------------
    # 6. Reciprocal Rank Fusion Mathematical Invariants
    # ---------------------------------------------------------
    def test_rrf_scoring_invariants(self):
        query = "Chandrayaan-3 lunar south pole"
        results = self.hybrid.search(query, top_k=5)
        
        self.assertTrue(len(results) > 0)
        # Verify descending order of RRF scores
        scores = [r.rrf_score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

        # Check score formula for top result
        top_r = results[0]
        expected_score = 0.0
        if top_r.bm25_rank:
            expected_score += 1.0 / (60 + top_r.bm25_rank)
        if top_r.dense_rank:
            expected_score += 1.0 / (60 + top_r.dense_rank)

        self.assertAlmostEqual(top_r.rrf_score, expected_score, places=6)

    # ---------------------------------------------------------
    # 7. Evaluator Metric Correctness
    # ---------------------------------------------------------
    def test_evaluator_metrics_calculation(self):
        evaluator = RetrievalEvaluator(self.test_corpus)
        
        dummy_claims = [
            {"claim_id": "claim_hi_01", "claim_text": "WHO declared polio free India", "language": "hi"},
            {"claim_id": "claim_bn_01", "claim_text": "Rabindranath Nobel Prize", "language": "bn"},
        ]

        metrics = evaluator.evaluate(dummy_claims, self.hybrid, k_values=[1, 3, 5])
        self.assertEqual(metrics.evaluated_claims_count, 2)
        self.assertEqual(metrics.unverifiable_skipped_count, 0)
        self.assertGreaterEqual(metrics.mrr, 0.5)
        self.assertGreaterEqual(metrics.hit_at_k[1], 0.5)


class TestDenseRetrieverCacheInvalidation(unittest.TestCase):
    """
    Regression test suite for content-hash-based FAISS cache invalidation and atomic persistence.
    """

    @classmethod
    def setUpClass(cls):
        # Prefer reusing model already initialized in TestRetrievalEngine to avoid duplicate weight loads
        engine_model = getattr(getattr(TestRetrievalEngine, "dense", None), "model", None)
        if engine_model is not None:
            cls.shared_model = engine_model
        else:
            from sentence_transformers import SentenceTransformer
            cls.shared_model = SentenceTransformer(DenseRetriever.MODEL_NAME, device="cpu")

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.index_file = os.path.join(self.temp_dir.name, "test_evidence.faiss")
        self.meta_file = os.path.join(self.temp_dir.name, "test_evidence_faiss_meta.json")

        self.corpus = EvidenceCorpus()
        self.doc1 = EvidenceDocument(
            doc_id="doc_01",
            text="Mercury is the smallest planet in the Solar System.",
            source_dataset="TEST"
        )
        self.doc2 = EvidenceDocument(
            doc_id="doc_02",
            text="Venus has a dense atmosphere consisting mostly of carbon dioxide.",
            source_dataset="TEST"
        )
        self.corpus.add_document(self.doc1)
        self.corpus.add_document(self.doc2)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cache_reuse_when_corpus_unchanged(self):
        """1. An unchanged corpus reuses the cache without re-encoding."""
        retriever_1 = DenseRetriever(
            self.corpus,
            device="cpu",
            index_path=self.index_file,
            meta_path=self.meta_file,
            model=self.shared_model
        )
        self.assertTrue(os.path.exists(self.index_file))
        self.assertTrue(os.path.exists(self.meta_file))

        with open(self.meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        orig_hash = meta.get("corpus_hash")
        self.assertIsNotNone(orig_hash)
        self.assertEqual(meta["corpus_len"], 2)

        # Second initialization with unchanged corpus reuses cache without re-encoding
        with mock.patch.object(self.shared_model, "encode", wraps=self.shared_model.encode) as mock_encode:
            retriever_2 = DenseRetriever(
                self.corpus,
                device="cpu",
                index_path=self.index_file,
                meta_path=self.meta_file,
                model=self.shared_model
            )
            mock_encode.assert_not_called()

        self.assertEqual(retriever_2.index.ntotal, 2)
        self.assertEqual(retriever_2.doc_ids, ["doc_01", "doc_02"])

    def test_cache_invalidates_when_passage_text_changes(self):
        """2. Passage text changes while document IDs and corpus length stay identical."""
        DenseRetriever(
            self.corpus,
            device="cpu",
            index_path=self.index_file,
            meta_path=self.meta_file,
            model=self.shared_model
        )
        with open(self.meta_file, "r", encoding="utf-8") as f:
            meta_initial = json.load(f)
        initial_hash = meta_initial["corpus_hash"]

        # Mutate passage text while keeping document IDs and corpus length identical
        self.doc1.text = "Jupiter is the largest planet in our solar system by mass."

        # New retriever detects hash mismatch and rebuilds
        retriever_2 = DenseRetriever(
            self.corpus,
            device="cpu",
            index_path=self.index_file,
            meta_path=self.meta_file,
            model=self.shared_model
        )
        with open(self.meta_file, "r", encoding="utf-8") as f:
            meta_updated = json.load(f)
        new_hash = meta_updated["corpus_hash"]

        self.assertNotEqual(initial_hash, new_hash)
        self.assertEqual(meta_updated["corpus_len"], 2)

        # Search for new text content
        res = retriever_2.search("Jupiter largest planet mass", top_k=1)
        self.assertTrue(len(res) > 0)
        self.assertEqual(res[0]["doc_id"], "doc_01")

    def test_cache_invalidates_when_document_added_or_removed(self):
        """3. A document is added or removed from the corpus."""
        DenseRetriever(
            self.corpus,
            device="cpu",
            index_path=self.index_file,
            meta_path=self.meta_file,
            model=self.shared_model
        )

        # Add document (length increases 2 -> 3)
        doc3 = EvidenceDocument(
            doc_id="doc_03",
            text="Mars has the largest dust storms in the solar system.",
            source_dataset="TEST"
        )
        self.corpus.add_document(doc3)

        retriever_added = DenseRetriever(
            self.corpus,
            device="cpu",
            index_path=self.index_file,
            meta_path=self.meta_file,
            model=self.shared_model
        )
        self.assertEqual(retriever_added.index.ntotal, 3)
        self.assertEqual(len(retriever_added.doc_ids), 3)

        # Remove document (length decreases 3 -> 2)
        del self.corpus.documents["doc_01"]
        retriever_removed = DenseRetriever(
            self.corpus,
            device="cpu",
            index_path=self.index_file,
            meta_path=self.meta_file,
            model=self.shared_model
        )
        self.assertEqual(retriever_removed.index.ntotal, 2)
        self.assertEqual(retriever_removed.doc_ids, ["doc_02", "doc_03"])

    def test_cache_invalidates_when_metadata_missing_or_outdated_schema(self):
        """4. Metadata is missing or has an outdated schema (missing corpus_hash)."""
        DenseRetriever(
            self.corpus,
            device="cpu",
            index_path=self.index_file,
            meta_path=self.meta_file,
            model=self.shared_model
        )

        # Simulate legacy metadata without corpus_hash
        with open(self.meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        del meta["corpus_hash"]
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        # Initializing detects missing corpus_hash, rebuilds and writes valid metadata
        retriever = DenseRetriever(
            self.corpus,
            device="cpu",
            index_path=self.index_file,
            meta_path=self.meta_file,
            model=self.shared_model
        )
        with open(self.meta_file, "r", encoding="utf-8") as f:
            fixed_meta = json.load(f)
        self.assertIn("corpus_hash", fixed_meta)
        self.assertEqual(fixed_meta["corpus_hash"], DenseRetriever.compute_corpus_hash(self.corpus))

    def test_cache_invalidates_when_faiss_file_corrupted(self):
        """5. The FAISS cache is corrupted or incompatible."""
        DenseRetriever(
            self.corpus,
            device="cpu",
            index_path=self.index_file,
            meta_path=self.meta_file,
            model=self.shared_model
        )

        # Corrupt FAISS index file with garbage bytes
        with open(self.index_file, "wb") as f:
            f.write(b"CORRUPTED_NON_FAISS_HEADER_GARBAGE_BYTES")

        # Retriever safely catches corruption and rebuilds index cleanly
        retriever = DenseRetriever(
            self.corpus,
            device="cpu",
            index_path=self.index_file,
            meta_path=self.meta_file,
            model=self.shared_model
        )
        self.assertIsNotNone(retriever.index)
        self.assertEqual(retriever.index.ntotal, 2)
        res = retriever.search("Mercury smallest planet", top_k=1)
        self.assertEqual(res[0]["doc_id"], "doc_01")

    def test_failed_rebuild_does_not_leave_falsely_valid_cache(self):
        """6. A failed rebuild does not leave a falsely valid cache."""
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        if os.path.exists(self.meta_file):
            os.remove(self.meta_file)

        # Simulate failure during encode using mock on shared_model
        with mock.patch.object(self.shared_model, "encode", side_effect=RuntimeError("CUDA OOM Failure")):
            with self.assertRaises(RuntimeError):
                DenseRetriever(
                    self.corpus,
                    device="cpu",
                    index_path=self.index_file,
                    meta_path=self.meta_file,
                    model=self.shared_model
                )

        # Assert no partial or invalid cache files remain
        self.assertFalse(os.path.exists(self.index_file))
        self.assertFalse(os.path.exists(self.meta_file))
        temp_files = [f for f in os.listdir(self.temp_dir.name) if f.endswith(".tmp")]
        self.assertEqual(temp_files, [])


if __name__ == "__main__":
    unittest.main()
