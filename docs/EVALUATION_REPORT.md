# VERITAS-X — Evaluation & Benchmark Report

This document reports the empirical validation results, regression benchmarks, and documented operational limitations of the **VERITAS-X Claim Verification Platform**.

---

## 1. Core Model & Pipeline Benchmarks

VERITAS-X utilizes `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` as its core Natural Language Inference (NLI) backbone, paired with an offline hybrid retrieval corpus (BM25 + Dense Multilingual E5-small) and an optional live Tavily search engine.

### Verification Performance (Frozen Phase 3.2.5 Benchmark Split)

| Metric | Raw Baseline | Contextualized Q+A + Joint Premise | Calibrated + Selective Abstention |
| :--- | :--- | :--- | :--- |
| **Accuracy (Overall)** | 68.4% | 76.8% | **84.2%** (on covered claims) |
| **Macro F1-Score** | 0.612 | 0.724 | **0.819** |
| **Expected Calibration Error (ECE)** | 0.184 (uncalibrated) | 0.142 | **0.048** (Temperature Scaled) |
| **Selective Abstention Coverage** | 100.0% (forced) | 100.0% (forced) | **78.5%** (21.5% abstained for safety) |
| **Disparity Detection Rate** | 42.0% | 78.5% | **94.1%** on conflicting benchmark fixtures |

*Note: All evaluation artifacts are preserved in `data/processed/` and remain frozen to prevent data leakage and benchmark drift.*

---

## 2. Reasoning Regression Test Suite Results

The platform was subjected to targeted automated regression verification across five required edge-case reasoning patterns:

| ID | Test Scenario / Claim | Evidence Inputs | Expected Verdict | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **REG-01** | **Subjective Comparison Without Aligned Evidence**<br>*"Electric cars are fundamentally superior to hydrogen cars in long-distance comfort."* | Uninformative or neutral environmental articles | `INSUFFICIENT_EVIDENCE`<br>(Abstained: True) | `INSUFFICIENT_EVIDENCE`<br>(Abstained: True, Conf: 25.0%) | **PASSED** |
| **REG-02** | **Direct Endorsement with Distractor Suppression**<br>*"The New York Post endorsed Donald Trump for re-election in 2020."* | Direct NY Post editorial + Unrelated Trump articles | `VERIFIED_TRUE`<br>(No false refutation) | `VERIFIED_TRUE`<br>(Conf: 89.4%, Entail Mass Dominant) | **PASSED** |
| **REG-03** | **Unaligned Date / Condition Claim**<br>*"Amy Coney Barrett was confirmed to the Supreme Court on October 26, 2020."* | General biographical articles without confirmation date | `INSUFFICIENT_EVIDENCE`<br>(Abstained: True) | `INSUFFICIENT_EVIDENCE`<br>(Abstained: True, Conf: 25.0%) | **PASSED** |
| **REG-04** | **Aligned Direct Refutation (Myth Verification)**<br>*"Humans use only 10% of their brains."* | Live web / neuroscience consensus evidence | `VERIFIED_FALSE` | `VERIFIED_FALSE`<br>(Conf: 64.9%, Refute Mass Dominant) | **PASSED** |
| **REG-05** | **Genuine Opposing Evidence Disparity**<br>*"The George Washington Bridge was completely closed to traffic on September 9, 2013."* | 1 Port Authority supporting passage + 1 local news refuting passage | `CONFLICTING_OR_CHERRYPICKED`<br>(Disparity: True) | `CONFLICTING_OR_CHERRYPICKED`<br>(Disparity: True, Conf: 66.4%) | **PASSED** |

---

## 3. Automated Test Suite Metrics

All tests were executed using Python's standard `unittest` discovery and Node.js test runner:

- **Total Python Unit & Integration Tests**: 175 passed (0 failures, 0 errors, 100% pass rate).
  - `tests/test_nli_polarity_regressions.py`: 4 tests passed (snippet hierarchy, teaser question handling, direct entailment preservation, and myth debunking aggregation).
  - `tests/test_tavily_provider.py`: 21 tests passed (cost caps, network failures, keyless logging, timeouts).
  - `tests/test_api_adapter.py`: 28 tests passed (HTTP serialization, stance mappings, mode contracts).
  - `tests/test_verification.py`: 21 tests passed (evidence aggregation, abstention, calibration).
  - `tests/test_phase6_quality_assurance.py`: 9 tests passed (full end-to-end QA fixtures).
  - `tests/test_retrieval.py` & `tests/test_data_foundation.py`: 24 tests passed (BM25, E5, FAISS, corpus hashes).
  - `tests/test_threshold_tuner.py`, `tests/test_nli_wrapper.py`, etc.: 68 tests passed.
- **Frontend Automated Tests**: 17 tests passed (0 failures, 100% pass rate).
- **Frontend Production Build**: Vite v6.3.5 compiled cleanly in 1.02s.

---

## 4. Documented Operational Limitations

1. **Retrieval Scope**:
   - **Local Corpus Mode**: Restricted to the pre-indexed 1,047 verified passages. Novel or breaking events outside the index will legitimately return zero passages and abstain.
   - **Live Web Mode**: Relies on Tavily Basic Search API snippets (~200–400 characters). While sufficient for most factual propositions, complex claims requiring deep document crawling may lack comprehensive context.
2. **Free-Tier Search Quotas**:
   - Outgoing live searches are capped at 800 queries per calendar month to protect user accounts from billing overages. If the monthly ceiling is reached, the system automatically falls back to Local Corpus mode.
3. **Inference Latency**:
   - Running the DeBERTa-v3 cross-encoder on CPU requires ~1.5 to 3.0 seconds per claim verification. When an NVIDIA GPU with CUDA is detected, inference latency drops to ~180–350 ms.
4. **Probabilistic Nature of NLI**:
   - Calibrated confidence reflects statistical class probability given the retrieved premises, not absolute philosophical certainty. When evidence is insufficient or contradictory, VERITAS-X deliberately abstains to avoid hallucinations.
