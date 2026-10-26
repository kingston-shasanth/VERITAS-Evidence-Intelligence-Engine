# VERITAS-X: Multilingual Evidentiary Fact-Checking, Stance Disparity & Interactive NLP Telemetry

[![Python 3.10--3.13](https://img.shields.io/badge/Python-3.10--3.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.6.0+cu124](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-red.svg)](https://pytorch.org/)
[![CUDA 12.4/12.5](https://img.shields.io/badge/CUDA-RTX%20Accelerated%20%7C%20CPU%20Fallback-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![React 19 / Vite 6](https://img.shields.io/badge/Frontend-React%2019%20%2F%20Vite%206-61dafb.svg)](ui/veritas-x-advanced-ui/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Evaluation: Frozen Phase 3.2.5](https://img.shields.io/badge/Evaluation-Held--Out%20Test%20(N=256)-purple.svg)](data/processed/final_test_evaluation_phase_3_2_5.json)

**VERITAS-X** is an end-to-end, scientifically audited automated fact-verification system designed for real-world, multilingual discourse (English, Hindi, Bengali, and Code-Mixed Hinglish). It addresses critical failure modes in standard fact-checking pipelines: **unsupported binary classifications**, **single-source confirmation bias**, **spurious conflict hallucinations**, and **severe neural overconfidence**.

The system features a **transparent 11-stage pipeline** that couples linguistic preprocessing, domain-aware spelling normalization, and meaning-preservation guardrails with dense/sparse hybrid retrieval, cross-encoder natural language inference (NLI), stance disparity aggregation, and post-hoc temperature calibration ($T = 4.1039$). The entire verification trace is visualized in real time via an interactive, natural light-themed UI dashboard.

---

## 1. Key Audited Results

All benchmark metrics derive from the post-hoc audited evaluation on the untouched held-out test split ($N=256$, stratified 50/50 split from AVeriTeC and Multilingual Diagnostic claims). All baseline numbers have been reconciled against the underlying confusion matrices.

| Evaluation Metric | Phase 3.1 Baseline | Phase 3.2.5 Full-Test ($N=256$) | Difference vs. Baseline | Phase 3.2.5 Selective ($N=242$, Coverage: 94.53%) | Notes & Governance Restrictions |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Accuracy** | 57.42% (147/256) | **59.38%** (152/256) | **+1.95 pp (+1.96 pp rounded)** | **62.40%** (151/242) | Selective accuracy evaluates retained predictions |
| **Macro-F1 (4-Class)** | 44.24% | **43.34%** | **-0.90 percentage points** | **45.54%** | Retained Macro-F1 across 242 retained claims |
| **Macro-Precision** | 45.55% *(reconciled)* | **46.91%** | **+1.36 percentage points** | **47.76%** | Corrected from earlier 45.68% transcription error |
| **Macro-Recall** | 50.34% | **49.07%** | **-1.27 percentage points** | **50.60%** | Trade-off from conservative conflict thresholds |
| **Conflicting Precision** | 14.29% (4/28) | **28.57%** (2/7) | **+14.29 percentage points** | **28.57%** (2/7) | **Precision doubled (2x)**; false alarms cut by 79% |
| **Conflicting Recall** | 21.05% (4/19) | **10.53%** (2/19) | **-10.53 percentage points** | **11.76%** (2/17) | Primary system limitation (honest disclosure) |
| **Coverage ($g(x)=1$)** | 100.0% | 100.0% | — | **94.53%** (242/256) | Passes $\ge 85\%$ threshold (14 claims abstained) |
| **Error Rate / Selective Risk** | 42.58% *(Full error)* | **40.62%** *(Full error)* | **-1.95 pp (+1.96 pp rounded)** | **37.60%** *(Selective risk)* | Full error ($1-\text{Acc}$) vs. selective risk on retained claims ($1-\text{Acc}_{\text{ret}}$) |

> [!NOTE]
> **Comparison Caveat**: Full-test metrics ($N=256$) and selective metrics ($N=242$, Coverage: $94.53\%$) evaluate different sample spaces because selective classification abstains on 14 ambiguous claims. Multiclass temperature scaling (`softmax(logits / T)` with $T=4.1039$) improves probabilistic reliability (NLL, ECE) without altering rank order.

### Multiclass Calibration Diagnostics (Phase 3.2.5 Test Logits, $N=256$)

Evaluated on the exact same Phase 3.2.5 held-out test logits before and after temperature scaling ($T = 4.1039$, fitted strictly on validation split NLL):

| Calibration Metric | Uncalibrated ($T=1$) | Calibrated ($T=4.1039$) | Relative / Absolute Delta | Mathematical & Practical Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **Negative Log-Likelihood (NLL)** | 1.9077 | **0.9915** | **-48.0% relative** ($-0.9161$) | Penalizes overconfident incorrect predictions (cross-entropy) |
| **Multiclass Brier Score** | 0.5774 | **0.5226** | **-9.5% relative** ($-0.0548$) | Mean squared difference between probabilities and one-hot labels |
| **Expected Calibration Error (ECE)** | 23.10% | **10.60%** | **-12.50 percentage points** | 10 equal-width bins across $[0.0, 1.0]$; confidence vs. accuracy |

### Per-Class Evaluation Breakdown

| Class Label | Support (Full / Ret) | Phase 3.1 Baseline (P / R / F1) | Phase 3.2.5 Full-Test ($N=256$) (P / R / F1) | Phase 3.2.5 Retained ($N=242$) (P / R / F1) |
| :--- | :---: | :---: | :---: | :---: |
| `VERIFIED_TRUE` | 61 / 57 | 64.44% / 47.54% / **54.72%** | 57.69% / 49.18% / **53.10%** | 57.69% / 52.63% / **55.05%** |
| `VERIFIED_FALSE` | 157 / 150 | 81.45% / 64.33% / **71.89%** | 81.68% / 68.15% / **74.31%** | 81.68% / 71.33% / **76.16%** |
| `CONFLICTING_OR_CHERRYPICKED` | 19 / 17 | 14.29% / 21.05% / **17.02%** | **28.57%** / 10.53% / **15.38%** | **28.57%** / 11.76% / **16.67%** |
| `INSUFFICIENT_EVIDENCE` | 19 / 18 | 22.03% / 68.42% / **33.33%** | 19.70% / 68.42% / **30.59%** | 23.08% / 66.67% / **34.29%** |

#### Confusion Matrices

**Full Test Split ($N=256$, Rows = Ground Truth, Columns = Predicted `[TRUE, FALSE, CONF, INSUF]`):**
```
                  Pred_TRUE   Pred_FALSE   Pred_CONF   Pred_INSUF   Support
True_TRUE        :  [   30,         12,          2,         17    ]    61
True_FALSE       :  [   17,        107,          2,         31    ]   157
True_CONF        :  [    3,          9,          2,          5    ]    19
True_INSUF       :  [    2,          3,          1,         13    ]    19
```

**Retained Predictions ($N=242$, Selective Coverage: 94.53%, 14 Claims Abstained):**
```
                  Pred_TRUE   Pred_FALSE   Pred_CONF   Pred_INSUF   Support
True_TRUE        :  [   30,         12,          2,         13    ]    57 (4 abstained)
True_FALSE       :  [   17,        107,          2,         24    ]   150 (7 abstained)
True_CONF        :  [    3,          9,          2,          3    ]    17 (2 abstained)
True_INSUF       :  [    2,          3,          1,         12    ]    18 (1 abstained)
```

---

## 2. The 11-Stage End-to-End Pipeline

VERITAS-X decomposes verification into eleven transparent, inspectable processing stages:

```
+---------------------------------------------------------------------------------------------------+
|                                     VERITAS-X 11-STAGE PIPELINE                                   |
+---------------------------------------------------------------------------------------------------+
  [Stage 1: Input & Sanitization]      --> Unicode NFC, URL/mention preservation, control char strip
  [Stage 2: Linguistic Analysis]       --> Multilingual tokenization, POS tags, Named Entity Recognition
  [Stage 3: Spelling Correction]       --> Domain-aware SymSpell, entity protection, frequency weighting
  [Stage 4: Grammar & Syntax]          --> Universal Dependency parsing, Head-Dependent relations
  [Stage 5: Claim Extraction]          --> Subject-Predicate-Object decomposition, temporal/negation flags
  [Stage 6: Safety Guardrails]         --> Meaning preservation, edit distance gates, safe fallback
  [Stage 7: Evidence Retrieval]        --> Sparse BM25 + Dense Multilingual E5 + Reciprocal Rank Fusion (k=60)
  [Stage 8: NLI Verification]          --> MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli cross-encoder
  [Stage 9: Stance Disparity]          --> Cross-passage conflict detection, near-duplicate suppression
  [Stage 10: Calibration ($T=4.1039$)] --> Temperature scaling: p = softmax(z / 4.1039)
  [Stage 11: Final Adjudication]       --> 4-Class veracity verdict + Selective Abstention Gate
```

1. **Stage 1 — Input & Sanitization (`src/veritas/nlp/sanitizer.py`)**:
   Standardizes input via Unicode NFC normalization, strips invisible control characters and zero-width spaces, and isolates entities such as URLs, email addresses, and hashtags from corruption.
2. **Stage 2 — Linguistic Analysis (`src/veritas/nlp/linguistics.py`)**:
   Performs multilingual tokenization, part-of-speech (POS) tagging, and named entity recognition (NER) across English, Hindi, Bengali, and code-mixed scripts.
3. **Stage 3 — Spelling Correction (`src/veritas/nlp/spelling.py`)**:
   Applies domain-aware SymSpell normalization with strict entity protection (scientific names, proper nouns, dates, numbers, and URLs are never altered). Returns candidate rankings with edit distance, frequency scores, and confidence metrics.
4. **Stage 4 — Grammar & Syntax (`src/veritas/nlp/linguistics.py`)**:
   Builds syntactic dependency trees capturing subject-verb-object relationships, nominal modifiers, and clause structures.
5. **Stage 5 — Claim Extraction (`src/veritas/nlp/claim_extractor.py`)**:
   Extracts canonical propositions into structured `(subject, predicate, object)` triples, tagging polarity (affirmative vs. negated) and temporal anchors.
6. **Stage 6 — Safety Guardrails (`src/veritas/nlp/guardrails.py`)**:
   Computes semantic preservation scores between raw and corrected text. If the correction alters meaning or exceeds edit-distance thresholds, the engine rolls back safely to raw input.
7. **Stage 7 — Evidence Retrieval (`src/veritas/retrieval/`)**:
   Executes hybrid search across the local verified corpus (1,047 passages) and optional live web citations via:
   - Multilingual Okapi BM25 with strict zero-score document filtering.
   - Dense multilingual E5 (`intfloat/multilingual-e5-small`) with FAISS index search.
   - Reciprocal Rank Fusion ($k = 60$) to combine sparse and dense rankings.
8. **Stage 8 — NLI Verification (`src/veritas/verification/nli_model.py`)**:
   Evaluates premise-hypothesis pairs using `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, producing entailment, contradiction, and neutral logits. Also computes concatenated multi-premise joint inference.
9. **Stage 9 — Stance Disparity (`src/veritas/verification/evidence_aggregator.py`)**:
   Detects genuine evidentiary conflicts by requiring distinct, non-duplicate passages taking opposing stances ($e_i \ge 0.35, c_j \ge 0.50, i \ne j$). Suppresses duplicate passages via Jaccard lexical similarity ($\ge 0.75$) and discounts neutral noise.
10. **Stage 10 — Calibration (`src/veritas/verification/calibrator.py`)**:
    Applies frozen multiclass temperature scaling ($\hat{p} = \text{softmax}(\mathbf{z} / 4.1039)$), transforming overconfident neural outputs into well-calibrated probabilities.
11. **Stage 11 — Final Adjudication (`src/veritas/verification/engine.py`)**:
    Synthesizes calibrated probabilities into one of four domain classes (`VERIFIED_TRUE`, `VERIFIED_FALSE`, `CONFLICTING_OR_CHERRYPICKED`, `INSUFFICIENT_EVIDENCE`) and evaluates the selective abstention gate ($g(x) = 0$ if $\max p < 0.32$ or normalized Shannon entropy $> 0.94$).

---

## 3. Four-Class Veracity Taxonomy

VERITAS-X enforces an unambiguous distinction between substantive domain classes and meta-level classification decisions:

1. **`VERIFIED_TRUE`**: Retrieved evidence factually entails and substantiates the claim.
2. **`VERIFIED_FALSE`**: Retrieved evidence directly refutes or contradicts the claim.
3. **`CONFLICTING_OR_CHERRYPICKED`**: Independent, distinct evidence passages present contradictory stances, or selective quoting conceals critical counter-evidence. Requires active cross-passage disparity ($\tau_E \ge 0.35, \tau_C \ge 0.50, i \ne j$).
4. **`INSUFFICIENT_EVIDENCE`**: Retrieved evidence lacks sufficient informative mass ($< 0.10$), is dominated by neutral text, or contains no answers to intermediate questions.
5. **Selective Abstention (`ABSTAIN`)**: A meta-level decision ($g(x) \in \{0, 1\}$) triggered when prediction confidence is $< 0.32$ or normalized Shannon entropy is $> 0.94$. Abstained claims are deferred to human review and are **not** conflated with `INSUFFICIENT_EVIDENCE`.

---

## 4. Quickstart: One-Click Windows Launch

### Starting the Platform (`START_VERITAS_X.bat`)

Double-click `START_VERITAS_X.bat` in the repository root:

```cmd
START_VERITAS_X.bat
```

What the launcher does automatically:
1. **Resolves Environment**: Detects Python in `.venv\Scripts\python.exe` (or system PATH) and Node.js (`node`, `npm`).
2. **Installs UI Dependencies**: Runs `npm install` automatically if `ui/veritas-x-advanced-ui/node_modules/` is missing.
3. **Port Conflict Preflight**: Checks whether ports `8011` (Backend API) and `5173` (Frontend UI) are free using `scripts/launcher_helper.py`.
4. **Starts Backend API**: Launches `python -m api.server --port 8011 --background-warmup` in a dedicated terminal window (instant socket binding + background model pre-warming).
5. **Starts Frontend Dashboard**: Launches `npm run dev` in a dedicated terminal window.
6. **Monitors Readiness**: Polls `http://127.0.0.1:8011/api/health` and `http://127.0.0.1:5173/` until both are fully ready.
7. **Launches Web Browser**: Automatically opens default browser to `http://127.0.0.1:5173`.

### Stopping the Platform (`STOP_VERITAS_X.bat`)

To cleanly shut down all VERITAS-X services without terminating unrelated Python or Node processes:

```cmd
STOP_VERITAS_X.bat
```

Alternatively, run:
```powershell
python scripts/launcher_helper.py stop
```

---

## 5. Manual Startup (Cross-Platform)

If running manually from separate terminal sessions:

### Terminal 1 — Backend API Server
```powershell
# From project root:
.\.venv\Scripts\Activate.ps1   # On Linux/macOS: source .venv/bin/activate
python -m api.server --host 127.0.0.1 --port 8011 --background-warmup
```
- Available at `http://127.0.0.1:8011/api/health`.
- Options:
  - `--background-warmup` (default): Binds port 8011 immediately and loads PyTorch/DeBERTa/FAISS in a background thread.
  - `--lazy`: Defers model loading until the first verification request.

### Terminal 2 — Frontend UI
```powershell
# From project root:
cd ui/veritas-x-advanced-ui
npm install
npm run dev
```
- Available at `http://127.0.0.1:5173`.

> [!IMPORTANT]
> **Why `file://` Direct Launch Does Not Work**:
> Opening `index.html` directly via `file://` fails because modern browsers enforce CORS security policies blocking ES modules (`<script type="module">`) from the filesystem. Additionally, dynamic HTTP API requests (`fetch("http://127.0.0.1:8011/api/verify")`) require an HTTP origin. The application must be accessed through Vite dev server or a local static file server.

---

## 6. Fresh Environment Installation

### System Prerequisites
- **Python**: 3.10, 3.11, 3.12, or 3.13
- **Node.js**: v18.0.0 or later (with `npm` 9+)
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Hardware**: CUDA-compatible GPU (4GB+ VRAM recommended, e.g., RTX 2050/3060) or CPU (automatic fallback)

### Step 1: Clone and Configure Environment
```bash
git clone https://github.com/<your-org>/VERITAS-X.git
cd VERITAS-X

# Copy environment configuration
cp .env.example .env
```

### Step 2: Python Backend Virtual Environment
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Linux/macOS: source .venv/bin/activate

# Install all Python dependencies (includes PyTorch, Transformers, FAISS, SymSpellPy, etc.)
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Frontend Client Setup
```bash
cd ui/veritas-x-advanced-ui
npm install
cd ../..
```

### Step 4: Verify Offline Model Cache (Optional)
The pre-downloaded Hugging Face checkpoints are stored in `.cache/huggingface`:
- `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- `intfloat/multilingual-e5-small`

The backend automatically sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` when loading from `.cache/huggingface`, ensuring offline operation without external API calls.

---

## 7. API Reference Specification

### 1. `GET /api/health`
Lightweight readiness and model liveness probe.

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "service": "VERITAS-X Evidence Verification API",
  "version": "Phase 3.2.5",
  "device": "cuda",
  "model_ready": true,
  "model_loading": false,
  "warmup_state": "ready",
  "init_latency_ms": 15721.5
}
```

### 2. `POST /api/verify`
Submits a natural language claim for end-to-end evidence retrieval, NLI stance scoring, cross-passage disparity aggregation, calibrated adjudication, and NLP telemetry.

**Request Payload:**
```json
{
  "claim": "Pur watur frezes at 0 C",
  "evidence": [],
  "mode": "local",
  "include_processing_trace": true
}
```

**Response Payload (`200 OK`):**
```json
{
  "claim_id": "VX-20261026-001",
  "claim": "Pur watur frezes at 0 C",
  "normalized_claim": "Pure water freezes at 0 C",
  "verdict": "VERIFIED_TRUE",
  "confidence": 0.9412,
  "probabilities": {
    "VERIFIED_TRUE": 0.9412,
    "VERIFIED_FALSE": 0.0210,
    "CONFLICTING_OR_CHERRYPICKED": 0.0150,
    "INSUFFICIENT_EVIDENCE": 0.0228
  },
  "has_cross_passage_disparity": false,
  "abstained": false,
  "reason": "Retrieved evidence directly substantiates the claim proposition under standard physical conditions.",
  "evidence": [
    {
      "id": "E-01",
      "source": "Local Evidence Corpus",
      "text": "Pure water freezes at 0 degrees Celsius (32 degrees Fahrenheit) at standard atmospheric pressure.",
      "stance": "support",
      "score": 0.9412,
      "entailment_prob": 0.9412,
      "contradiction_prob": 0.0210,
      "neutral_prob": 0.0378,
      "is_informative": true,
      "doc_id": "doc_412"
    }
  ],
  "normalization_status": "CORRECTED",
  "normalization_corrections": [
    {
      "original_word": "Pur",
      "normalized_word": "Pure",
      "edit_distance": 1,
      "confidence": 0.95,
      "candidate_source": "symspell"
    },
    {
      "original_word": "watur",
      "normalized_word": "water",
      "edit_distance": 1,
      "confidence": 0.98,
      "candidate_source": "symspell"
    },
    {
      "original_word": "frezes",
      "normalized_word": "freezes",
      "edit_distance": 1,
      "confidence": 0.97,
      "candidate_source": "symspell"
    }
  ],
  "processing_trace_version": "1.0",
  "processing_trace": [
    {
      "stage": "input_received",
      "status": "completed",
      "summary": "Received raw user input.",
      "duration_ms": 0.04
    },
    {
      "stage": "sanitization",
      "status": "completed",
      "summary": "Unicode NFC normalization complete. No control characters found.",
      "duration_ms": 0.07
    },
    {
      "stage": "spelling_normalization",
      "status": "completed",
      "summary": "Applied 3 spelling correction(s). Meaning preservation score: 0.96.",
      "duration_ms": 4.45
    },
    {
      "stage": "meaning_preservation",
      "status": "completed",
      "summary": "Safety guardrails accepted correction safely.",
      "duration_ms": 0.38
    },
    {
      "stage": "tokenization",
      "status": "completed",
      "summary": "Tokenized into 6 tokens and 1 named entity.",
      "duration_ms": 0.06
    },
    {
      "stage": "linguistic_analysis",
      "status": "completed",
      "summary": "POS tagging and Universal Dependency syntax tree constructed.",
      "duration_ms": 0.12
    },
    {
      "stage": "claim_extraction",
      "status": "completed",
      "summary": "Extracted subject 'Pure water', predicate 'freezes', object 'at 0 C'.",
      "duration_ms": 0.11
    },
    {
      "stage": "retrieval",
      "status": "completed",
      "summary": "Hybrid BM25 + Dense E5 retrieval retrieved 5 evidence passages.",
      "duration_ms": 34.74
    },
    {
      "stage": "evidence_processing",
      "status": "completed",
      "summary": "Suppressed duplicates and evaluated topical relevance.",
      "duration_ms": 0.08
    },
    {
      "stage": "nli_verification",
      "status": "completed",
      "summary": "Cross-encoder DeBERTa-v3 evaluated stance across all passages.",
      "duration_ms": 142.24
    },
    {
      "stage": "final_result",
      "status": "completed",
      "summary": "Final adjudication complete: VERIFIED_TRUE (Confidence: 94.1%).",
      "duration_ms": 0.50
    }
  ],
  "device": "cuda",
  "latency_ms": 218.4
}
```

---

## 8. Automated Testing & Quality Verification

```powershell
# 1. Frontend Unit & Telemetry Test Suite (45 tests across 8 suites - 100% PASS)
cd ui/veritas-x-advanced-ui
npm test

# 2. Frontend Production Bundle Compilation (Vite 6 - 100% clean build)
npm run build
cd ../..

# 3. Core NLP, Telemetry, Adapter & Live Smoke Suites (72 tests - 100% PASS)
.\.venv\Scripts\python.exe -m unittest tests/test_nlp_spelling_normalization.py tests/test_nlp_linguistics_guardrails.py tests/test_api_nlp_integration_telemetry.py tests/test_api_adapter.py tests/test_live_smoke.py

# 4. Full Backend Test Discovery (328 tests: 325 pass, 3 frozen multi-hop edge cases)
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

---

## 9. Final Architecture Tree

```
VERITAS-X/
├── START_VERITAS_X.bat                  # One-click Windows launcher (preflight, dual-server, browser launch)
├── STOP_VERITAS_X.bat                   # Safe target-process shutdown script (ports 8011 & 5173)
├── README.md                            # Authoritative academic overview, architecture & operational guide
├── LICENSE                              # MIT License
├── requirements.txt                     # Core Python dependencies (PyTorch, Transformers, FAISS, SymSpellPy)
├── .gitignore                           # Production git exclusion rules
├── .env.example                         # Environment configuration template
│
├── api/                                 # HTTP REST API Layer
│   ├── __init__.py                      # Package initialization
│   ├── adapter.py                       # VeritasAPIAdapter (NLP integration, calibration, CORS, schema contract)
│   └── server.py                        # Threading HTTP Server (POST /api/verify, GET /api/health)
│
├── src/veritas/                         # Core Machine Learning & NLP Engine
│   ├── __init__.py                      # Root package initialization
│   ├── nlp/                             # Phase 4 NLP Pipeline Modules
│   │   ├── __init__.py                  # NLP package initialization
│   │   ├── sanitizer.py                 # Stage 1: Unicode NFC, control char stripping, entity shielding
│   │   ├── linguistics.py               # Stages 2 & 4: Multilingual tokenization, POS tagging, dependency parsing
│   │   ├── spelling.py                  # Stage 3: Domain-aware SymSpell spelling normalization
│   │   ├── claim_extractor.py           # Stage 5: Semantic claim structure & proposition extraction
│   │   ├── guardrails.py                # Stage 6: Meaning preservation scoring & safety guardrails
│   │   └── telemetry.py                 # Structured telemetry accumulator for UI animation playback
│   ├── retrieval/                       # Stage 7: Evidence Retrieval System
│   │   ├── __init__.py                  # Retrieval package initialization
│   │   ├── bm25_retriever.py            # Multilingual Okapi BM25 (zero-score filtered)
│   │   ├── dense_retriever.py           # Dense Multilingual E5 embedding retriever
│   │   ├── hybrid_retriever.py          # Reciprocal Rank Fusion (RRF, k=60)
│   │   ├── evidence_corpus.py           # FAISS index loader and passage metadata manager
│   │   ├── tavily_provider.py           # Live web search integration with credit limits
│   │   ├── tokenizer.py                 # Custom multilingual lexical tokenizer
│   │   └── evaluation.py                # Retrieval recall and MRR evaluation
│   ├── verification/                    # Stages 8-11: Adjudication Engine
│   │   ├── __init__.py                  # Verification package initialization
│   │   ├── nli_model.py                 # CrossEncoderNLI wrapper (MoritzLaurer/DeBERTa-v3)
│   │   ├── evidence_aggregator.py       # Stance disparity aggregator & neutral mass discounting
│   │   ├── calibrator.py                # TemperatureCalibrator (T = 4.1039) & selective abstention
│   │   ├── engine.py                    # End-to-end VerificationEngine orchestrator
│   │   ├── evaluation.py                # Multiclass evaluation metrics (NLL, ECE, Brier, F1)
│   │   └── temporal_analyzer.py         # Date delta audit and temporal anachronism detection
│   ├── data/                            # Benchmark Data Foundation
│   │   ├── __init__.py                  # Data package initialization
│   │   ├── schema.py                    # Canonical dataclasses & CanonicalVeracity taxonomy
│   │   ├── preprocessor.py              # Text cleaning, NFC normalization, script detection
│   │   ├── averitec_loader.py           # AVeriTeC JSONL parser and evidence extractor
│   │   ├── fever_loader.py              # FEVER dataset parser
│   │   ├── multilingual_loader.py       # Indic (Hindi, Bengali) and Hinglish loader
│   │   └── pipeline.py                  # End-to-end dataset builder pipeline
│   ├── graph/                           # Evidence Consensus Topology
│   │   ├── __init__.py                  # Graph package initialization
│   │   └── graph_builder.py             # Knowledge and consensus graph builder
│   ├── reasoning/                       # Multi-Hop Reasoning Modules
│   │   ├── __init__.py                  # Reasoning package initialization
│   │   ├── decomposer.py                # Complex claim sub-question decomposer
│   │   ├── multihop_verifier.py         # Multi-hop bridge reasoning verifier
│   │   └── types.py                     # Decomposition data structures
│   └── temporal/                        # Temporal Intelligence
│       ├── __init__.py                  # Temporal package initialization
│       └── temporal_analyzer.py         # Regex/ISO-8601 temporal parser and drift detector
│
├── ui/veritas-x-advanced-ui/            # Sole Active React 19 / Vite 6 Forensic Dashboard
│   ├── package.json                     # Frontend dependencies and npm scripts
│   ├── package-lock.json                # Authoritative npm lockfile
│   ├── vite.config.js                   # Vite configuration
│   ├── index.html                       # Application HTML root
│   ├── backend_contract.md              # Exact REST API payload contract specification
│   ├── README.md                        # UI architecture and component documentation
│   ├── dist/                            # Production build bundle (Vite 6 compiled)
│   ├── public/                          # Static assets (favicon.svg, veritas-symbol.svg)
│   ├── tests/                           # Frontend Jest/Testing-Library unit tests
│   │   ├── nlp_visualization_telemetry.test.js  # Telemetry parsing & stage rendering tests
│   │   └── ui_contract_and_voice.test.js       # Contract normalization & narrator tests
│   └── src/                             # React Source Code
│       ├── main.jsx                     # Application bootstrap
│       ├── App.jsx                      # Main workspace orchestration & page routing
│       ├── styles.css                   # Phase 4C natural, integrated light-theme design system
│       ├── lib/                         # Client Utility Libraries
│       │   ├── api.js                   # REST client & response normalizer
│       │   └── narrator.js              # Web Speech API briefing narrator
│       └── components/                  # UI Components
│           ├── ClaimStructureView.jsx   # Stage 5: Subject-Predicate-Object triple visualization
│           ├── ConfusionMatrix.jsx      # Interactive Phase 3.2.5 confusion matrix explorer
│           ├── EvidenceFlowView.jsx     # Stage 7 & 8: Evidence card flow & stance indicators
│           ├── EvidenceGraph.jsx        # Consensus network graph visualization
│           ├── ForensicReport.jsx       # 6-section forensic adjudication dossier
│           ├── Icons.jsx                # Light-theme SVG icon library
│           ├── InvestigationProgress.jsx# Radar telemetry progress & execution path
│           ├── MeaningPreservationPanel.jsx # Stage 6: Safety guardrail score gauges
│           ├── NLPVisualization.jsx     # Main 11-stage interactive animation coordinator
│           ├── OrbitHero.jsx            # Interactive orbital telemetry hero
│           ├── PlaybackControls.jsx     # Play/pause/step/speed animation toolbar
│           ├── ProcessingTimeline.jsx   # 11-stage progress tracker and stage navigator
│           ├── StageDetailsModal.jsx    # Deep-dive inspector modal for each stage
│           ├── SyntaxRelationshipView.jsx # Stage 4: Syntactic dependency tree viewer
│           ├── TokenInspector.jsx       # Stage 2: POS tag & token chip inspector
│           ├── VerificationReveal.jsx   # Stage 10 & 11: Calibrated verdict reveal
│           ├── VoiceNarrator.jsx        # Voice briefing controller
│           └── WordCorrectionAnimation.jsx # Stage 3: Animated spelling correction cards
│
├── scripts/                             # Utility & Benchmarking Scripts
│   ├── launcher_helper.py               # Runtime preflight, health polling, PID manager for .bat files
│   ├── demo_claim.py                    # Single-claim CLI demonstration runner
│   ├── run_demo_claims.py               # Batch live HTTP verification runner
│   ├── evaluate_heldout_test.py         # Frozen Phase 3.2.5 held-out test evaluation runner
│   ├── evaluate_verification.py         # End-to-end verification and calibration evaluator
│   ├── evaluate_retrieval.py            # Retrieval benchmark evaluation runner (BM25 vs Dense vs Hybrid)
│   ├── calibrate_and_abstain.py         # Validation calibration & selective abstention evaluator
│   ├── tune_thresholds.py               # Hyperparameter threshold tuner
│   ├── process_canonical_dataset.py     # Canonical dataset harmonization pipeline
│   ├── download_samples.py              # Sample dataset acquisition runner
│   ├── verify_live_api_and_workspaces.py # Live API workspace verification runner
│   ├── verify_live_comprehensive.py     # Multi-category regression verification runner
│   └── diagnose_and_improve.py          # Error profiling & diagnostic script
│
├── tests/                               # Comprehensive Automated Test Suites (25 test files)
│   ├── test_api_adapter.py              # Adapter mapping, contract, and regression tests (30 tests)
│   ├── test_api_nlp_integration_telemetry.py # Pipeline integration & telemetry tests (9 tests)
│   ├── test_nlp_spelling_normalization.py   # SymSpell, entity protection, Hinglish tests (22 tests)
│   ├── test_nlp_linguistics_guardrails.py    # Linguistics, claim extraction, guardrails tests (21 tests)
│   ├── test_live_smoke.py               # HTTP server health, CORS, and smoke tests (4 tests)
│   ├── test_calibrator.py               # Temperature scaling & selective abstention tests
│   ├── test_evidence_aggregator.py      # Stance disparity, deduplication, aggregation tests
│   ├── test_threshold_tuner.py          # Hyperparameter grid search tests
│   ├── test_nli_wrapper.py              # Cross-encoder label mapping & device fallback tests
│   ├── test_retrieval.py                # BM25 zero-filtering, dense FAISS, hybrid RRF tests
│   ├── test_data_foundation.py          # Dataset schema normalization & split tests
│   ├── test_general_evidence_reasoning.py # Multi-evidence reasoning tests
│   ├── test_systemic_root_cause_regressions.py # Edge case and bridge reasoning tests
│   ├── test_tavily_provider.py          # Web search provider & cost limit tests
│   └── ...                              # Remaining domain and regression test suites
│
├── docs/                                # Authoritative Documentation
│   ├── ARCHITECTURE.md                  # Comprehensive pipeline architecture & data flow
│   ├── data_dictionary.md               # Data dictionary for benchmark datasets & schemas
│   └── reproducibility.md               # Exact audit trail, commands, and seeds for Phase 3.2.5
│
└── data/processed/                      # Active Ground-Truth Datasets & Frozen Evaluation Baselines
    ├── canonical_claims.jsonl           # 1,111 harmonized canonical claims
    ├── evidence_corpus.jsonl            # 1,047 unique, verified ground-truth evidence passages
    ├── evidence_corpus.faiss            # FAISS index for dense multilingual E5 search
    ├── evidence_corpus_faiss_meta.json  # FAISS index metadata and doc_id mappings
    ├── validation_model_comparison.json # Phase 3.1 validation comparison cache (frozen)
    ├── test_model_evaluation.json       # Phase 3.1 held-out test baseline (frozen)
    ├── threshold_tuning_results.json    # Phase 3.2.3 hyperparameter tuning artifact (frozen)
    ├── calibration_and_abstention_results.json # Phase 3.2.4 calibration artifact (frozen)
    └── final_test_evaluation_phase_3_2_5.json  # Phase 3.2.5 final audited benchmark report (frozen)
```

---

## 10. Limitations & Forensic Disclosures

1. **Conflicting-Evidence Recall Trade-Off:**
   In Phase 3.2.5 held-out evaluations, conflicting-class recall is 10.53% (2/19 claims detected). Conservative thresholds were chosen to double precision to 28.57% and cut false alarms by 79%. Subtle or co-located conflicting claims default safely to `VERIFIED_FALSE` or `INSUFFICIENT_EVIDENCE`.
2. **Domain Boundaries & World Knowledge:**
   Textual NLI evaluates semantic entailment from documentary text. Unstated physical equivalences (e.g., equating "standard atmospheric pressure" to "sea level") are treated as `NEUTRAL` without domain-specific bridging text.
3. **Calibrated Confidence Semantics:**
   Confidence scores represent evidence alignment under post-hoc temperature scaling ($T=4.1039$). They quantify consistency with available citations, not an unconditional guarantee of real-world truth.

---

## 11. Dataset Attribution & Academic Citations

```bibtex
@inproceedings{schoch2023averitec,
  title     = {AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web},
  author    = {Schoch, Michael and Guo, Yufang and Vlachos, Andreas},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
  note      = {License: CC BY 4.0}
}

@inproceedings{thorne2018fever,
  title     = {FEVER: a large-scale dataset for Fact Extraction and VERification},
  author    = {Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Mittal, Arpit},
  booktitle = {NAACL-HLT},
  year      = {2018},
  note      = {License: CC BY-SA 3.0}
}

@inproceedings{laurer2023deberta,
  title     = {Less Annotating, More Classifying: Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and NLI},
  author    = {Laurer, Moritz and Atteveldt, Wouter van and Casas, Andreu and Welbers, Kasper},
  journal   = {Political Analysis},
  year      = {2023},
  note      = {Model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, License: MIT/Apache 2.0}
}

@article{wang2022e5,
  title     = {Text Embeddings by Weakly-Supervised Contrastive Pre-training},
  author    = {Wang, Liang and Yang, Nan and Huang, Xiaolong and Jiao, Binxing and Yang, Linjun and Jiang, Daxin and Majumder, Rangan and Wei, Furu},
  journal   = {arXiv preprint arXiv:2212.03533},
  year      = {2022},
  note      = {Model: intfloat/multilingual-e5-small, License: MIT}
}

@inproceedings{guo2017calibration,
  title     = {On Calibration of Modern Neural Networks},
  author    = {Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2017}
}

@inproceedings{geifman2017selective,
  title     = {Selective Classification for Deep Neural Networks},
  author    = {Geifman, Yonatan and El-Yaniv, Ran},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2017}
}
```

---

## 12. License

The VERITAS-X codebase is licensed under the [MIT License](LICENSE). Third-party datasets and pre-trained model weights remain governed by their respective upstream licenses (AVeriTeC: CC BY 4.0; FEVER: CC BY-SA 3.0; HuggingFace model checkpoints: MIT / Apache 2.0).
