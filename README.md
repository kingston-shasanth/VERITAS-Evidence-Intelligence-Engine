# VERITAS-X: Multilingual Evidentiary Fact-Checking, Stance Disparity & Uncertainty Calibration

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.6.0+cu124](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-red.svg)](https://pytorch.org/)
[![CUDA 12.4/12.5](https://img.shields.io/badge/CUDA-RTX%202050%20(4GB)-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Evaluation: Frozen Phase 3.2.5](https://img.shields.io/badge/Evaluation-Held--Out%20Test%20(N=256)-purple.svg)](data/processed/final_test_evaluation_phase_3_2_5.json)

**VERITAS-X** is an end-to-end, scientifically audited automated fact-checking and evidence verification system designed for real-world, multilingual discourse (English, Hindi, Bengali, and Code-Mixed Hinglish). It addresses critical failure modes in standard fact-checking pipelines: **unsupported binary classifications**, **single-source confirmation bias**, **spurious conflict hallucinations**, and **severe neural overconfidence**.

---

## 1. Key Audited Results

All metrics below derive from the post-hoc audited benchmark artifacts on the untouched, deterministic held-out test split ($N=256$, stratified 50/50 split from AVeriTeC and Multilingual Diagnostic claims). All baseline numbers have been reconciled against the underlying confusion matrices.

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
> **Comparison Caveat**: Full-test metrics ($N=256$) and selective metrics ($N=242$, Coverage: $94.53\%$) are **not strictly like-for-like comparisons**, because selective classification evaluates performance on a subset where the model declined 14 uncertain claims. They are reported separately to ensure transparency. Multiclass temperature scaling (`softmax(logits / T)` with $T=4.1039$) improves probabilistic reliability (NLL, ECE) but does not guarantee classification correctness on individual difficult claims.

### Multiclass Calibration Diagnostics (Phase 3.2.5 Test Logits, $N=256$)

The metrics below evaluate the effect of multiclass temperature scaling ($\hat{p} = \text{softmax}(\mathbf{z} / T)$ with scalar $T = 4.1039$, fitted strictly on validation split NLL) on the saved Phase 3.2.5 held-out test logits. These figures represent a **before-versus-after calibration evaluation on the exact same Phase 3.2.5 test logits**, not a comparison against the Phase 3.1 baseline (Phase 3.1 baseline logits were not calibrated). Temperature scaling adjusts probability confidence to reflect empirical accuracy without altering logit rank order or classification decisions.

| Calibration Metric | Phase 3.2.5 test logits — uncalibrated (T=1) | Phase 3.2.5 test logits — calibrated (T=4.1039) | Difference (Before vs. After Calibration) | Mathematical & Practical Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **Negative Log-Likelihood (NLL)** | 1.9077 (1.907682) | **0.9915** (0.991533) | **-48.0% relative** ($-0.9161$) | Penalizes overconfident incorrect predictions; cross-entropy loss |
| **Multiclass Brier Score** | 0.5774 (0.577370) | **0.5226** (0.522609) | **-9.5% relative** ($-0.0548$) | Mean squared difference between predicted class probabilities and one-hot ground truth |
| **Expected Calibration Error (ECE)** | 23.10% (0.231014) | **10.60%** (0.106013) | **-12.50 percentage points** | 10 equal-width bins across $[0.0, 1.0]$; measures confidence vs. empirical accuracy |


### Per-Class Evaluation Breakdown

| Class Label | Support (Full / Ret) | Phase 3.1 Baseline (P / R / F1) | Phase 3.2.5 Full-Test ($N=256$) (P / R / F1) | Phase 3.2.5 Retained ($N=242$, Coverage: 94.53%) (P / R / F1) |
| :--- | :---: | :---: | :---: | :---: |
| `VERIFIED_TRUE` | 61 / 57 | 64.44% / 47.54% / **54.72%** | 57.69% / 49.18% / **53.10%** | 57.69% / 52.63% / **55.05%** |
| `VERIFIED_FALSE` | 157 / 150 | 81.45% / 64.33% / **71.89%** | 81.68% / 68.15% / **74.31%** | 81.68% / 71.33% / **76.16%** |
| `CONFLICTING_OR_CHERRYPICKED` | 19 / 17 | 14.29% / 21.05% / **17.02%** | **28.57%** / 10.53% / **15.38%** | **28.57%** / 11.76% / **16.67%** |
| `INSUFFICIENT_EVIDENCE` | 19 / 18 | 22.03% / 68.42% / **33.33%** | 19.70% / 68.42% / **30.59%** | 23.08% / 66.67% / **34.29%** |

#### Confusion Matrices

**Full Test Split ($N=256$, Ground Truth Rows $\times$ Predicted Columns `[TRUE, FALSE, CONF, INSUF]`):**
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

## 2. System Architecture

```
+---------------------------------------------------------------------------------------------------+
|                                     VERITAS-X SYSTEM PIPELINE                                     |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                    [Natural Language Claim]
                         (English, Hindi, Bengali, Code-Mixed Hinglish)
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
              [Claim Preprocessing]                           [Temporal Analyzer]
            - Unicode NFC Normalization                     - Regex & ISO-8601 Parser
            - Date & Currency Preservation                  - Delta vs Claim Date Audit
            - Script Classification                         - Anachronism Drift Warning
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
                                     [Hybrid Retrieval Engine]
                         - Sparse BM25 (Custom Multilingual Okapi BM25)
                         - Dense Multilingual E5 (FAISS Cosine Similarity)
                         - Reciprocal Rank Fusion (RRF, k = 60)
                         - Local Verified Corpus (1,047 Passages)
                                                  |
                                                  v
                                      [Cross-Encoder NLI Engine]
                            (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
                            - Dynamic Label Mapping (resolve_nli_label_mapping)
                            - Pairwise Inference: (Passage Premise, Claim)
                            - Concatenated Joint Premise Inference
                                                  |
                                                  v
                               [Cross-Passage Stance Disparity Aggregator]
                            - Jaccard Lexical Near-Duplicate Suppression (0.75)
                            - Neutral Mass Noise Discounting (0.70)
                            - Stance Disparity: Distinct i != j, e_i >= 0.35, c_j >= 0.50
                            - Joint Premise De-weighting under Disparity (w = 0.40 vs 1.20)
                            - Conflict Ratio: 2 * min(E, C) / (E + C) >= 0.45
                            - Informative Mass Gate: total_mass >= 0.10
                                                  |
                                                  v
                                   [Four-Class Synthesized Logits]
                            [z_TRUE, z_FALSE, z_CONFLICTING, z_INSUFFICIENT]
                                                  |
                                                  v
                                [Multiclass Temperature Calibrator]
                                   p = softmax(z / T),  T = 4.1039
                                (Fitted strictly on validation split N=255)
                                                  |
                                                  v
                                   [Selective Abstention Engine]
                                   - Max probability < 0.320?
                                   - Normalized Shannon Entropy > 0.940?
                                                  |
                                  +---------------+---------------+
                                  |                               |
                           Retained (94.5%)               Abstained (5.5%)
                                  |                               |
                                  v                               v
                        [Categorical Assertion]        [Referred to Human Review]
                       (TRUE, FALSE, CONF, INSUF)      (Selective Deferral Option)
```

---

## 3. Four-Class Veracity Taxonomy

VERITAS-X enforces an unambiguous distinction between substantive domain classes and meta-level classification decisions:

1. **`VERIFIED_TRUE`**: Retrieved evidence factually entails and substantiates the claim.
2. **`VERIFIED_FALSE`**: Retrieved evidence directly refutes or contradicts the claim.
3. **`CONFLICTING_OR_CHERRYPICKED`**: Independent, distinct evidence passages present contradictory stances, or selective quoting conceals critical counter-evidence. Requires active cross-passage disparity ($\tau_E \ge 0.35, \tau_C \ge 0.50, i \ne j$).
4. **`INSUFFICIENT_EVIDENCE`**: The retrieved evidence lacks sufficient informative mass ($< 0.10$), is dominated by neutral text, or contains no answer to intermediate questions.
5. **Selective Abstention (`ABSTAIN`)**: A meta-level decision ($g(x) \in \{0, 1\}$) triggered when the model's prediction confidence is $< 0.32$ or normalized Shannon entropy is $> 0.94$. Abstained claims are deferred to human review and are **not** silently conflated with the domain class `INSUFFICIENT_EVIDENCE`.

---

## 4. One-Click Startup & Quickstart

### The One-Click Windows Launcher (`START_VERITAS_X.bat`)

VERITAS-X provides a root-level one-click launcher for Windows:

```
Double-click: START_VERITAS_X.bat
```

When launched, `START_VERITAS_X.bat`:
1. **Resolves the Project Root**: Determines the root directory using `%~dp0` regardless of invocation context or working directory.
2. **Detects Environment**: Automatically verifies the local Python virtual environment (`.venv\Scripts\python.exe`) and Node.js (`node`, `npm`).
3. **Preflight Port Conflict Detection**: Executes `scripts/launcher_helper.py preflight` to ensure ports `8000` (API) and `5173` (Frontend) are not in use.
4. **Launches Background Services**:
   - Starts the Python inference API (`python -m api.server --port 8000 --lazy`) in a dedicated terminal window.
   - Starts the React/Vite frontend (`npm run dev`) in a dedicated terminal window.
5. **Polls Genuine Readiness**: Waits for the HTTP health check (`http://127.0.0.1:8000/api/health`) and the Vite frontend (`http://127.0.0.1:5173`) using `scripts/launcher_helper.py wait-ready`.
6. **Launches Browser**: Automatically opens the default web browser to `http://127.0.0.1:5173`.
7. **Graceful Error Visibility**: Keeps terminal windows open with clear diagnostic instructions if Python, Node, or dependencies are missing.

### Safe Application Shutdown (`STOP_VERITAS_X.bat`)

To cleanly shut down the running application without terminating unrelated Python or Node processes:

```
Double-click: STOP_VERITAS_X.bat
```

Alternatively, run:
```powershell
python scripts/launcher_helper.py stop
```
This utility inspects active TCP listeners on ports `8000` and `5173`, identifies the specific PIDs (`python.exe` and `node.exe`), and safely terminates only the VERITAS-X processes. You may also press `Ctrl+C` in the respective server terminal windows.

---

### Important: Why `file://` Direct Launch Does Not Work

Directly opening `ui/veritas-x-advanced-ui/index.html` via `file://` in a web browser will fail because:
1. **ES Module Security (CORS)**: Modern React and Vite applications rely on native ES module syntax (`<script type="module" src="/src/main.jsx">`). Modern web browsers enforce Cross-Origin Resource Sharing (CORS) and strictly prohibit loading ES modules from the local filesystem (`file://`) origin.
2. **Dynamic REST API Networking**: The frontend must issue asynchronous HTTP requests (`fetch`) to the local Python verification backend (`http://127.0.0.1:8000/api/verify`). `file://` pages cannot reliably establish cross-protocol HTTP connections to local servers.
3. **Asset Resolution**: Vite bundles and resolves assets using absolute and root-relative URLs (`/src/...`, `/assets/...`) which fail to resolve under filesystem path roots.

The application must be accessed through the local HTTP server created by Vite (`http://127.0.0.1:5173`) or by serving the production build (`dist/`) via a local web server.

---

## 5. Primary System Limitation (Honest Disclosure)

### Conflicting-Class Recall on Held-Out Test Data
While the Phase 3.2.2 stance-disparity aggregator successfully **doubled conflicting precision from 14.29% to 28.57%** (+14.29 percentage points) and reduced false positive conflict alarms by **79%** (from 24 false alarms in Phase 3.1 down to 5 in Phase 3.2.5), **conflicting-class recall dropped from 21.05% (4/19) to 10.53% (2/19)** (-10.53 percentage points). Consequently, overall full-test Macro-F1 decreased from **44.24% to 43.34%** (-0.90 percentage points).

#### Root Cause:
The cross-passage disparity rule requires two distinct, non-duplicate passages taking opposing stances ($e_i \ge 0.35, c_j \ge 0.50, i \ne j$). In real-world web evidence, conflicting facts are frequently nuanced, implicit, or co-located within a single paragraph rather than split across cleanly segmented documents. When evidence is subtle, the conservative aggregator defaults to the dominant stance (`VERIFIED_FALSE` or `INSUFFICIENT_EVIDENCE`) rather than triggering a false alarm.

---

## 6. Verified Project Structure

```
VERITAS-X/
├── START_VERITAS_X.bat                        # Root one-click Windows launcher (preflight, dual-server, browser launch)
├── STOP_VERITAS_X.bat                         # Safe target-process shutdown script (ports 8000 & 5173)
├── README.md                                  # Academic overview, architecture & quickstart
├── LICENSE                                    # MIT License specification
├── requirements.txt                           # Core Python dependencies (PyTorch, Transformers, FAISS)
├── .gitignore                                 # Production repository exclusion rules
├── .env.example                               # Unified environment configuration template
├── api/
│   ├── __init__.py                            # API package initialization
│   ├── adapter.py                             # Thin live API adapter (contract adherence & CORS)
│   └── server.py                              # Threading HTTP server (POST /api/verify, GET /api/health)
├── ui/veritas-x-advanced-ui/                  # Sole active React 19 / Vite 6 Forensic Dashboard
│   ├── package.json                           # Frontend scripts and dependencies
│   ├── package-lock.json                      # Authoritative npm lockfile
│   ├── vite.config.js                         # Vite development and bundle configuration
│   ├── index.html                             # Single-page application root
│   ├── backend_contract.md                    # Exact REST payload contract specification
│   ├── .env.example                           # Frontend environment defaults
│   ├── .gitignore                             # Node & Vite build exclusions
│   ├── dist/                                  # Verified production build bundle
│   ├── tests/
│   │   └── ui_contract_and_voice.test.js      # Response normalization & voice briefing unit tests
│   └── src/
│       ├── main.jsx                           # Application bootstrap
│       ├── App.jsx                            # Forensic workspace orchestration (5 primary pages)
│       ├── styles.css                         # Unified midnight graphite design system
│       ├── lib/
│       │   ├── api.js                         # REST adapter & canonical response normalizer
│       │   └── narrator.js                    # Web Speech API script generator
│       └── components/
│           ├── ForensicReport.jsx             # 6-section forensic adjudication dossier
│           ├── InvestigationProgress.jsx      # Indeterminate pipeline radar & execution path
│           ├── VoiceNarrator.jsx              # Native Web Speech API voice briefing controller
│           ├── EvidenceGraph.jsx              # Consensus network & topology visualization
│           ├── ConfusionMatrix.jsx            # Audited Phase 3.2.5 confusion matrix explorer
│           ├── OrbitHero.jsx                  # Interactive orbital telemetry visualization
│           └── Icons.jsx                      # Forensic SVG icon library
├── src/veritas/
│   ├── data/                                  # Canonical schema, AVeriTeC/FEVER loaders, preprocessors
│   ├── retrieval/                             # Multilingual tokenizer, BM25 (zero-match filtered), Dense E5, FAISS, Hybrid RRF
│   └── verification/                          # CrossEncoderNLI wrapper, EvidenceAggregator, Calibrator, Engine (multi-premise)
├── scripts/
│   ├── launcher_helper.py                     # Cross-platform preflight, health polling, and safe process manager
│   ├── run_demo_claims.py                     # Live HTTP end-to-end demo verification runner
│   ├── demo_claim.py                          # Single-claim CLI & interactive demonstration
│   ├── evaluate_heldout_test.py               # Frozen one-shot evaluation on held-out test split (DO NOT RERUN)
│   ├── evaluate_verification.py               # End-to-end verification and calibration evaluator
│   ├── evaluate_retrieval.py                  # Retrieval benchmark evaluation (BM25 vs Dense vs Hybrid)
│   ├── calibrate_and_abstain.py               # Validation calibration & selective abstention evaluation
│   ├── tune_thresholds.py                     # Validation-only hyperparameter grid search (decoupled coverage)
│   ├── process_canonical_dataset.py           # Canonical benchmark dataset harmonization pipeline
│   ├── download_samples.py                    # Sample dataset acquisition runner
│   ├── verify_live_api_and_workspaces.py      # Automated 8-workspace and live API verification
│   └── verify_live_comprehensive.py           # End-to-end multi-category regression and claim verification
├── tests/
│   ├── test_api_adapter.py                    # Adapter mapping, contract, and regression tests (16 tests)
│   ├── test_live_smoke.py                     # HTTP server health, CORS, and smoke tests (4 tests)
│   ├── test_calibrator.py                     # Temperature scaling and selective abstention tests
│   ├── test_evidence_aggregator.py            # Stance disparity, deduplication, and aggregation tests
│   ├── test_threshold_tuner.py                # Hyperparameter constraint and grid search tests (4 tests)
│   ├── test_nli_wrapper.py                    # Cross-encoder label mapping and device fallback tests
│   ├── test_retrieval.py                      # BM25 zero-filtering, dense FAISS, and hybrid RRF tests
│   ├── test_data_foundation.py                # Dataset schema normalization and split tests
│   └── test_diagnose_checkpointing.py         # Checkpointing and diagnostic evaluation tests
└── data/processed/
    ├── canonical_claims.jsonl                 # 1,111 harmonized canonical claims
    ├── evidence_corpus.jsonl                  # 1,047 unique, verified ground-truth evidence passages
    ├── validation_model_comparison.json       # Phase 3.1 validation comparison cache (untouched)
    ├── test_model_evaluation.json             # Phase 3.1 held-out test evaluation baseline (untouched)
    ├── threshold_tuning_results.json          # Phase 3.2.3 tuning artifact (untouched)
    ├── calibration_and_abstention_results.json# Phase 3.2.4 calibration artifact (untouched)
    └── final_test_evaluation_phase_3_2_5.json # Phase 3.2.5 final test evaluation report (frozen benchmark)
```

---

## 7. Manual Startup, API Contracts & Testing

## 7. Manual Startup, API Contracts & Testing

### One-Click Quickstart (Recommended)

Double-click `START_VERITAS_X.bat` in the repository root. This:
1. Validates Python `.venv` and Node.js environment prerequisites.
2. Checks port availability (`8000` for Python API, `5173` for React UI) via `scripts/launcher_helper.py preflight`.
3. Binds HTTP port 8000 immediately and starts background model warming via `--background-warmup`.
4. Starts the Vite frontend on `http://127.0.0.1:5173`.
5. Monitors `/api/health` until `model_ready: true` is confirmed before launching the default browser.
6. To cleanly stop all services, run `STOP_VERITAS_X.bat` or press Ctrl+C.

### Manual Fallback Startup Commands

If launching manually from terminal sessions:

#### Terminal 1 — Backend API Server
```powershell
# From project root:
.\.venv\Scripts\Activate.ps1
python -m api.server --host 127.0.0.1 --port 8000 --background-warmup
```
- `--background-warmup` (default): Binds port 8000 immediately (<15 ms) and preloads DeBERTa-v3 and FAISS indices in a background thread, eliminating first-request latency.
- `--lazy`: Defers model loading until the first user verification request.
- Default local binding: `127.0.0.1:8000` (loopback only, private to host machine).

#### Terminal 2 — Frontend UI
```powershell
# From project root:
cd ui\veritas-x-advanced-ui
npm install
npm run dev
```
- Available at `http://127.0.0.1:5173`.

---

### API Endpoint Specification

#### 1. `POST /api/verify`
Submits a natural language claim for end-to-end evidence retrieval, NLI stance scoring, cross-passage disparity aggregation, and calibrated adjudication.

**Request Payload:**
```json
{
  "claim": "A thermonuclear bomb was detonated on the Moon during the Apollo 11 mission in July 1969.",
  "user_evidence": []
}
```

**Response Payload (`200 OK`, 11 Canonical Keys):**
```json
{
  "claim_id": "VX-001",
  "claim": "A thermonuclear bomb was detonated on the Moon during the Apollo 11 mission in July 1969.",
  "verdict": "VERIFIED_FALSE",
  "confidence": 0.3508,
  "probabilities": {
    "VERIFIED_TRUE": 0.2215,
    "VERIFIED_FALSE": 0.3508,
    "CONFLICTING_OR_CHERRYPICKED": 0.1240,
    "INSUFFICIENT_EVIDENCE": 0.3037
  },
  "has_cross_passage_disparity": false,
  "abstained": false,
  "reason": "The model assigned plurality probability to VERIFIED_FALSE with low calibrated confidence (35.1%), but no individual retrieved passage directly refutes the proposition (5 neutral/uninformative passage(s)). The claim lacks direct documentary refutation in the retrieved corpus.",
  "evidence": [
    {
      "id": "E-01",
      "source": "Local Evidence Corpus",
      "text": "Apollo 11 landed on the lunar surface on July 20, 1969...",
      "stance": "neutral",
      "score": 0.9421,
      "entailment_prob": 0.0210,
      "contradiction_prob": 0.0369,
      "neutral_prob": 0.9421,
      "is_informative": false,
      "doc_id": "doc_104"
    }
  ],
  "device": "cuda",
  "latency_ms": 128.4
}
```

#### 2. `GET /api/health`
Lightweight readiness and model liveness probe distinguishing HTTP socket binding from AI model readiness.

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

---

### Running Automated Test Suites

```powershell
# 1. Full Python Unit & Regression Test Suite (171 passed in ~30s)
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"

# 2. Reasoning Regression Verification (5/5 targeted scenarios passed)
.\.venv\Scripts\python.exe scratch/test_reasoning_regressions.py

# 3. Frontend Unit & Contract Tests (17 passed in ~94ms)
cd ui\veritas-x-advanced-ui
npm test

# 4. Frontend Production Build Verification (Vite v6.3.5)
npm run build
```

---

## 8. Multi-Mode Evidence Retrieval & Free-Tier Cost Controls

VERITAS-X supports three runtime evidence modes:
- **Local Corpus (`mode="local"`)**: Fully offline verification against the pre-indexed 1,047 verified passages (BM25 + Dense Multilingual E5 + FAISS + RRF). Requires zero credentials or internet access.
- **Live Web (`mode="web"`)**: Real-time web search citations retrieved via official Tavily Basic Search API, normalized into candidate evidence passages, and passed through DeBERTa-v3 NLI inference.
- **Hybrid (`mode="hybrid"`)**: Merged local corpus and web evidence with URL/hash deduplication to prevent double-counting.

### Free-Tier Cost Safeguards
- **Basic Search Only**: Advanced crawling and deep extraction disabled to strictly conserve credits (1 credit/search).
- **Persistent Ledger**: Hard application-side cap of **800 monthly credits** tracked atomically in `data/tavily_usage.json`.
- **Zero Retries**: Network timeouts or provider errors fail gracefully to prevent credit burn loops.
- **Permanent Lockout**: Automatic current-month freeze upon HTTP 402/429 billing errors.

For detailed data flow and deployment guides, see:
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): Complete data flow and pipeline specification.
- [docs/FINAL_RUN_GUIDE.md](docs/FINAL_RUN_GUIDE.md): Windows setup, startup/shutdown, and API configuration.
- [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md): Empirical benchmarks, reasoning regressions, and limitations.

---

## 9. Limitations & Forensic Disclosures

1. **Conflicting-Evidence Recall Trade-Off:**
   In Phase 3.2.5 held-out evaluations, conflicting-class recall is 10.53% (2/19 claims detected). Conservative thresholds were chosen to prevent false-alarm conflict declarations (precision doubled to 28.57%).
2. **Domain Boundaries & External World Knowledge:**
   Textual NLI models verify lexical and semantic entailment from documentary text. Unstated physical equivalences (e.g., equating "standard atmospheric pressure" to "sea level") are treated strictly as `NEUTRAL` without domain-specific bridging text.
3. **Calibrated Confidence vs. Ground Truth:**
   Confidence scores represent evidence alignment under post-hoc temperature scaling ($T=4.1039$). They quantify consistency with available citations, not an unconditional guarantee of real-world factuality.

### Core Defect Remediations & Engineering Upgrades

During productization, three verified runtime and tuning defects were resolved:

1. **Priority A — BM25 Zero-Score Document Filtering (`src/veritas/retrieval/bm25_retriever.py`)**:
   - *Previous Defect*: Out-of-vocabulary or zero-overlap queries returned arbitrary documents with 0.0 BM25 scores simply because they occupied top array indices.
   - *Remediation*: Added `if score > 0.0` filtering. Zero-match queries now return empty result sets (`[]`), preventing noisy false-positive passages from entering the Reciprocal Rank Fusion (RRF) pool.
2. **Priority B — Multi-Premise Joint Inference Parity (`src/veritas/verification/engine.py`)**:
   - *Previous Defect*: Multi-premise cross-encoder inference was evaluated in historical research experiments but omitted from the live verification engine loop.
   - *Remediation*: When multiple candidate evidence passages are retrieved, the engine builds a concatenated joint premise and evaluates it via a unified batch NLI pass, supplying `joint_nli_prediction` to `EvidenceAggregator`.
3. **Priority C — Decoupling Insufficient Evidence from Abstention (`scripts/tune_thresholds.py`)**:
   - *Previous Defect*: The hyperparameter tuning script conflated the domain verdict `INSUFFICIENT_EVIDENCE` with the meta-level decision to abstain ($g(x)=0$), artificially miscounting coverage.
   - *Remediation*: Coverage calculation was decoupled to reflect operational deferral criteria ($\max p < \tau_{\text{conf}}$ or $H_{\text{norm}} > \tau_{\text{ent}}$), with configurable coverage constraints (`min_coverage_threshold`).
4. **Offline Hugging Face Startup (`api/adapter.py`)**:
   - *Optimization*: Configured `HF_HUB_OFFLINE = 1` and `TRANSFORMERS_OFFLINE = 1` to prevent unauthenticated network polling to `huggingface.co`, ensuring instant local startup from `.cache/huggingface`.

---

## 10. Dataset Attribution & Academic Citations

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

## 11. License

The VERITAS-X codebase is licensed under the [MIT License](LICENSE). Third-party datasets and pre-trained weights remain governed by their respective upstream licenses (AVeriTeC: CC BY 4.0; FEVER: CC BY-SA 3.0; HuggingFace model checkpoints: MIT / Apache 2.0).
