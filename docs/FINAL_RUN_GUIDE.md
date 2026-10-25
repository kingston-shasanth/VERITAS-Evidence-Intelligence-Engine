# VERITAS-X — Final Run & Demonstration Guide

This guide provides comprehensive, step-by-step instructions for launching, operating, configuring, testing, and evaluating the **VERITAS-X Evidence-Grounded AI Fact-Verification Platform**.

---

## 1. System Architecture & Port Allocation

VERITAS-X integrates a multi-mode evidence retrieval pipeline, DeBERTa-v3 natural language inference (NLI), confidence calibration, relation alignment safeguards, and a modern React/Vite conversational interface.

| Service | Port | Endpoint / URL | Purpose |
| :--- | :--- | :--- | :--- |
| **Backend API** | `8011` | `http://127.0.0.1:8011/api/verify` | Multi-mode claim verification (Local, Live Web, Hybrid) & NLI aggregation |
| **Backend Health** | `8011` | `http://127.0.0.1:8011/api/health` | Model status, device (CPU/CUDA), and warmup readiness |
| **Frontend UI** | `5173` | `http://127.0.0.1:5173/` | Conversational verification interface & 8 research workspaces |

> [!NOTE]
> **Port Isolation**: Port 8011 is strictly dedicated to VERITAS-X to avoid conflicts with external services occupying port 8000.

---

## 2. Prerequisites & Environment

- **Python**: Version 3.10 to 3.13 (project virtual environment at `.\.venv\Scripts\python.exe`).
- **Node.js**: Version 18+ and `npm`.
- **Operating System**: Windows 10/11 (PowerShell or Command Prompt).
- **RAM**: Minimum 8 GB (16 GB recommended for DeBERTa cross-encoder in-memory inference).
- **Local Evidence Corpus**: Pre-indexed BM25 and dense E5 FAISS index (~1,047 passages).

---

## 3. Evidence Modes & Web Retrieval Setup

VERITAS-X natively supports three explicit evidence retrieval modes selectable via the UI or the `/api/verify` endpoint:

1. **Local Corpus (`local`)**:
   - Searches the offline 1,047 verified passages using BM25 keyword matching + dense multilingual E5 embeddings + FAISS with Reciprocal Rank Fusion (RRF, k=60).
   - **Requires zero internet connection and no API keys.**

2. **Live Web (`web`)**:
   - Executes real-time web queries through the official **Tavily Search API**.
   - Normalizes titles, URLs, text snippets, and publication dates into the evidence pipeline.
   - Passes live web snippets through the existing DeBERTa-v3 NLI model and relation-alignment safeguards.

3. **Hybrid (`hybrid`)**:
   - Combines retrieved offline corpus passages with live Tavily web search results.
   - Deduplicates identical or overlapping citations.
   - Prevents duplicate sources from artificially inflating verdict confidence.

### Configuring Tavily API Key (Backend-Only)
`TAVILY_API_KEY` is read strictly from backend environment variables or a local `.env` file. It is **never** embedded in frontend bundles, browser state, or application logs.

#### Windows PowerShell:
```powershell
$env:TAVILY_API_KEY="tvly-YOUR_ACTUAL_KEY_HERE"
```

#### Windows Command Prompt:
```cmd
set TAVILY_API_KEY=tvly-YOUR_ACTUAL_KEY_HERE
```

#### Local `.env` File (Repository Root):
Create or edit `.env` in `D:\VERITAS-X\.env`:
```env
TAVILY_API_KEY=tvly-YOUR_ACTUAL_KEY_HERE
```

> [!IMPORTANT]
> **Unconfigured / Offline Policy**: If no Tavily API key is configured and Live Web or Hybrid mode is selected, the engine honestly returns `retrieval_status: "unconfigured"` and abstains (`INSUFFICIENT_EVIDENCE`) rather than fabricating search results.

---

## 4. Quick Start (One-Click Launch)

### Option A: Windows Launcher (Recommended)
Double-click `START_VERITAS_X.bat` in the repository root:
```cmd
START_VERITAS_X.bat
```
This launcher automatically:
1. Detects `.\.venv\Scripts\python.exe`.
2. Inspects ports `8011` and `5173`.
3. Starts the backend server with background model warmup (`python -m api.server --port 8011 --background-warmup`).
4. Starts the Vite dashboard in `ui\veritas-x-advanced-ui`.
5. Polls `/api/health` until the DeBERTa model is warmed, then opens the browser at `http://127.0.0.1:5173/`.

### Option B: Manual Startup

**Terminal 1 — Backend Server:**
```powershell
.\.venv\Scripts\python.exe -m api.server --port 8011 --background-warmup
```

**Terminal 2 — Frontend:**
```powershell
cd ui\veritas-x-advanced-ui
npm run dev
```

---

## 5. Stopping the Application Safely

Double-click `STOP_VERITAS_X.bat` or run:
```cmd
STOP_VERITAS_X.bat
```
This script terminates **only** the processes listening on ports `8011` and `5173` that belong to VERITAS-X (recorded PIDs or Python/Node processes on those ports). It will **never** terminate unrelated processes on port 8000 or elsewhere.

---

## 6. Primary Conversational Interface & 8 Research Workspaces

The redesigned frontend employs a cohesive, warm editorial palette:
- **Canvas / Background**: `#F7F6F2`
- **Primary Surfaces**: `#FFFFFF` (Crisp cards & panels)
- **Secondary Surfaces**: `#EFEEE8` (Elevated secondary containers)
- **Primary Text**: `#202A31` (High-contrast editorial charcoal)
- **Secondary Text**: `#56636A` (Readable metadata slate)
- **Muted Text**: `#748087` (Muted labels & timestamps)
- **Borders**: `#E1E3DD` (Restrained clean borders)
- **Primary Accent**: `#176B5B` (Deep Forest Teal) / Background `#E8F4EF`
- **Secondary Accent**: `#C47745` (Warm Terracotta)
- **Semantic Verdicts**: Verified True (`#176B5B`), Verified False (`#B91C1C`), Conflicting (`#C47745`), Insufficient (`#56636A`).

### The 8 Workspaces:
1. **Claim Investigator** (`investigate`): Main conversational experience with natural language input, 3-way mode toggle (Local/Web/Hybrid), demonstration chips, conversational results, clickable web sources, and evidence filtering.
2. **Command Center** (`overview`): Executive operational overview, audited validation benchmarks (N=255), and session investigation history.
3. **Evidence Dossier** (`evidence`): Deep passage inspection, side-by-side comparative analysis, copyable citations, and stance breakdown.
4. **Evidence Graph** (`graph`): Interactive topology visualizing relationships between claims, decomposed subclaims, and retrieved sources.
5. **Temporal & Provenance Timeline** (`timeline`): Chronological reconstruction of event and publication dates; tracks outdated/superseded claims.
6. **Multi-Hop Reasoning** (`multihop`): Atomic claim decomposition showing conjunctions, causal dependencies, and individual subclaim support.
7. **Model Performance Lab** (`evaluation`): Audited confusion matrix, per-class precision/recall/F1, and temperature calibration curve.
8. **System Telemetry** (`system`): Hardware state (CUDA/CPU), inference latency, memory footprint, and monthly search credit usage.

---

## 7. Demonstration Claims & Reliability Safeguards

### Benchmark Test Cases:

1. **Direct Grounded Support**:
   - **Claim**: `"The New York Post endorsed Donald Trump's bid for re-election."`
   - **Mode**: Local Corpus
   - **Expected Verdict**: `VERIFIED_TRUE`
   - **Mechanism**: The direct endorsement passage is retrieved and decisive. Irrelevant topical articles mentioning Trump or other media do not trigger false conflicts.

2. **Refuted Myth with Debunking Snippets**:
   - **Claim**: `"Humans use only 10% of their brains."`
   - **Mode**: Live Web
   - **Expected Verdict**: `VERIFIED_FALSE` (confidence: ~96.5%)
   - **Mechanism**: Live web search returns multiple independent scientific debunking articles (STEAM Ahead, MIT McGovern, UAMS, Nature). Discontinuous search snippets containing teaser questions (`"Do we really use 10%...?"`) and headers (`"### Myth:"`) are resolved hierarchically without inverting the refuting assertions.

3. **Incomplete Condition / Date Verification**:
   - **Claim**: `"Amy Coney Barrett was confirmed as US Supreme Court Justice on October 26, 2020."`
   - **Mode**: Local Corpus
   - **Expected Verdict**: `INSUFFICIENT_EVIDENCE` (Abstained)
   - **Mechanism**: General hearing passages are retrieved, but none establish or disprove the specific confirmation date. The engine safely abstains instead of declaring a false refutation (`VERIFIED_FALSE`).

4. **Subjective Comparison**:
   - **Claim**: `"this veritas x is better than chatgpt"`
   - **Mode**: Local Corpus
   - **Expected Verdict**: `INSUFFICIENT_EVIDENCE` (Abstained)
   - **Mechanism**: Unverifiable subjective claim with zero documentary evidence. The engine detects uninformative evidence mass (< 0.10) and withholds an unsupported verdict.

5. **Genuine Cross-Passage Conflict**:
   - **Claim**: `"The municipal bridge was completely closed to vehicle traffic on Monday."`
   - **Mode**: User-supplied or Local with opposing bulletins
   - **Expected Verdict**: `CONFLICTING_OR_CHERRYPICKED`
   - **Mechanism**: Detects mutually exclusive statements from two official municipal bulletins regarding the closure.

---

## 8. Automated Test Execution

### Run Full Backend Test Suite (175 tests):
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```
- `test_nli_polarity_regressions.py` (4 tests): Verifies snippet hierarchy resolution, teaser question handling, direct entailment preservation, and end-to-end myth debunking aggregation.
- `test_tavily_provider.py` (21 tests): Mocks successful Tavily search, unconfigured key, 401 authorization error, 402 payment required, 429 quota reached, 800-credit monthly ceiling, network timeout, connection failure, malformed JSON, query sanitization, and snippet deduplication.
- `test_api_adapter.py` (28 tests): Validates API contracts, CORS headers, error codes, local mode, web mode, hybrid mode, and unconfigured key abstention policy.
- `test_verification.py` (21 tests): Stance classification, cross-passage disparity detection, relation-alignment, topic-mismatch safeguards, and selective coverage abstention.
- `test_phase6_quality_assurance.py` (9 tests): Full end-to-end QA fixtures, selective abstention, and accessibility reduced-motion styles.
- `test_retrieval.py` & `test_data_foundation.py` (24 tests): Dense Multilingual E5 embedding, FAISS index invalidation/rebuilding, BM25 keyword search, RRF merging, and data normalization.

### Run Frontend Contract Tests (17 tests):
```powershell
cd ui\veritas-x-advanced-ui
npm test
```

### Validate Production Build:
```powershell
cd ui\veritas-x-advanced-ui
npm run build
```

---

## 9. Operating Limits & Disclaimers

1. **Snippets vs. Full Articles**: Web search retrieval operates on search snippets provided by the Tavily API. Snippets provide high-density summaries, but cannot replace exhaustive document reading.
2. **Missing Metadata**: If an online source does not provide a published date or author, VERITAS-X will never invent or guess missing metadata.
3. **Model Plurality vs. Ground Truth**: Calibrated model confidence reflects multi-premise statistical alignment produced by DeBERTa-v3 cross-encoder inference. It is not an infallible guarantee of real-world truth.
4. **Local Corpus Scope**: The offline corpus comprises ~1,047 curated passages. Queries outside this domain will naturally yield `INSUFFICIENT_EVIDENCE` in local mode unless live web retrieval is enabled.
