# VERITAS-X: Academic Reproducibility Guide

This guide provides complete, step-by-step instructions to reproduce the experimental results, validation diagnostics, unit test suites, and demonstration workflows of **VERITAS-X**.

---

## 1. Hardware & System Specifications

The benchmark results reported in Phase 3.1 and Phase 3.2.5 were generated and audited on the following reference system:

| Specification | Primary Tested Environment | Minimum Supported |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Home / Pro (64-bit) | Windows 10/11 or Linux (Ubuntu 20.04+) |
| **CPU Architecture** | x86_64 (Intel Core i5 / AMD Ryzen) | 4 physical cores |
| **System Memory (RAM)** | 16 GB DDR4/DDR5 | 8 GB RAM |
| **GPU Accelerator** | NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM) | CPU-only (CUDA optional but recommended) |
| **GPU Compute Architecture** | Ampere (Compute Capability 8.6) | Pascal (CC 6.1+) |
| **NVIDIA Driver** | 555.97 (CUDA 12.5 driver capability) | 525+ (CUDA 12.0+) |
| **Python Version** | Python 3.13.2 (64-bit) | Python 3.10 – 3.13 |

---

## 2. Environment Setup & Dependency Installation

To prevent dependency drift, VERITAS-X runs within an isolated virtual environment (`.venv`).

### Step 2.1: Virtual Environment Creation
```powershell
# Open PowerShell in the workspace root: d:\VERITAS-X
cd d:\VERITAS-X

# Create an isolated virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
# On Linux / macOS: source .venv/bin/activate
```

### Step 2.2: CUDA-Enabled PyTorch & Core Libraries
Install PyTorch with CUDA 12.4 support, followed by the verification and NLP dependencies:

```powershell
# Install PyTorch with CUDA 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA acceleration
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
# Expected output: CUDA Available: True, Device: NVIDIA GeForce RTX 2050 Laptop GPU

# Install transformers, retrieval, and evaluation libraries
pip install transformers sentence-transformers rank-bm25 scikit-learn faiss-cpu numpy scipy tqdm
```

---

## 3. Dataset Directory Structure & Integrity

All benchmark claims and verified passages are stored in the `data/processed/` directory.

| File Path | Description | Record Count |
| :--- | :--- | :---: |
| [`data/processed/canonical_claims.jsonl`](file:///d:/VERITAS-X/data/processed/canonical_claims.jsonl) | Unified canonical claim schema across all sources | 1,111 claims |
| [`data/processed/evidence_corpus.jsonl`](file:///d:/VERITAS-X/data/processed/evidence_corpus.jsonl) | Local verified evidence passages answering Q&A | 1,047 unique passages |
| [`data/processed/validation_model_comparison.json`](file:///d:/VERITAS-X/data/processed/validation_model_comparison.json) | Phase 3.1 model diagnostic validation cache | 4 models evaluated |
| [`data/processed/test_model_evaluation.json`](file:///d:/VERITAS-X/data/processed/test_model_evaluation.json) | Phase 3.1 held-out test evaluation baseline | Untouched baseline ($N=256$) |
| [`data/processed/threshold_tuning_results.json`](file:///d:/VERITAS-X/data/processed/threshold_tuning_results.json) | Phase 3.2.3 validation grid search results | 1,536 combinations ($N=255$) |
| [`data/processed/calibration_and_abstention_results.json`](file:///d:/VERITAS-X/data/processed/calibration_and_abstention_results.json) | Phase 3.2.4 temperature scaling & abstention grid | Validation diagnostics ($N=255$) |
| [`data/processed/final_test_evaluation_phase_3_2_5.json`](file:///d:/VERITAS-X/data/processed/final_test_evaluation_phase_3_2_5.json) | Phase 3.2.5 final one-shot test evaluation | Held-out test split ($N=256$) |

### Dataset Partitioning & Stratification
- Evaluated claims: 511 claims from AVeriTeC (491) and Multilingual Diagnostic (20).
- Stratification procedure: `VeracityEvaluator.create_stratified_splits(eval_claims, val_ratio=0.5, random_seed=42)`
- **Validation Split ($N=255$)**: Used strictly for model selection, threshold tuning, and calibration fitting.
- **Held-Out Test Split ($N=256$)**: Used strictly for the one-shot final evaluation.
  - Class distribution: `VERIFIED_FALSE`: 157, `VERIFIED_TRUE`: 61, `CONFLICTING`: 19, `INSUFFICIENT`: 19.

---

## 4. Frozen Pipeline Parameters

All production and evaluation runs lock the following frozen hyperparameters:

```python
# CrossEncoder NLI Model Checkpoint
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Evidence Aggregator Thresholds (Phase 3.2.3)
FROZEN_AGGREGATOR_PARAMS = {
    "disparity_entail_threshold": 0.35,  # Min entailment for active supporting passage
    "disparity_contra_threshold": 0.50,  # Min contradiction for active refuting passage
    "conflict_threshold": 0.45,          # Min conflict balance ratio 2*min(E,C)/(E+C)
    "min_informative_mass": 0.10,        # Min mass to adjudicate beyond INSUFFICIENT
    "joint_weight_disparity": 0.40,      # De-weighted joint multiplier under disparity
    "joint_weight_default": 1.20,        # Default joint multiplier when passages agree
}

# Calibration & Selective Abstention (Phase 3.2.4)
CALIBRATION_TEMPERATURE = 4.1039         # Learned strictly on validation NLL
CONFIDENCE_THRESHOLD = 0.32              # Minimum confidence to assert categorical verdict
ENTROPY_THRESHOLD = 0.94                 # Maximum normalized Shannon entropy
```

---

## 5. Supported Commands

### 5.1 Run the Full Unit Test Suite (68 Tests)
Runs all unit and integration tests across data processing, retrieval, NLI label resolution, evidence aggregation, temperature calibration, and selective abstention:

```powershell
python -m unittest discover tests
```
*Expected Result:* `Ran 68 tests in ~18s: OK`

### 5.2 Run Single-Claim Demonstration
Verifies natural language claims using the frozen Phase 3.2.5 pipeline and local evidence corpus:

```powershell
# Curated safe demonstration fixtures:
python scripts/demo_claim.py --fixture 1   # Entailed scientific fact (VERIFIED_TRUE)
python scripts/demo_claim.py --fixture 2   # Contradicted claim (VERIFIED_FALSE)
python scripts/demo_claim.py --fixture 3   # Opposing evidence disparity (CONFLICTING)
python scripts/demo_claim.py --fixture 4   # Zero evidence in corpus (INSUFFICIENT_EVIDENCE)

# Ad-hoc claim verification:
python scripts/demo_claim.py --claim "Pure water freezes at 0 degrees Celsius at sea level."
```

### 5.3 Run Read-Only Post-Hoc Numerical Audit
Recomputes all confusion matrices, supports, precision, recall, F1, calibration metrics, and selective statistics from raw claim evaluations, and reconciles the baseline precision:

```powershell
python scratch/run_audit.py
```
*Expected Result:*
- Full-test accuracy: **59.375%** ($\Delta = 0.0000$)
- Full-test Macro-F1: **43.3439%** ($\Delta = 0.0000$)
- Full-test Macro-Precision: **46.910%** ($\Delta = 0.0000$)
- Full-test Macro-Recall: **49.070%** ($\Delta = 0.0000$)
- Calibration Recomputation (Phase 3.2.5 test logits: uncalibrated $T=1 \to$ calibrated $T=4.1039$):
  - NLL: **1.9077** $\to$ **0.9915** ($\Delta = 0.0000$)
  - Brier score: **0.5774** $\to$ **0.5226** ($\Delta = 0.0000$)
  - ECE: **23.10%** $\to$ **10.60%** ($\Delta = 0.0000$)
- Selective coverage: **94.53125%** (242/256, $\Delta = 0.0000$)
- Selective accuracy: **62.3967%** ($\Delta = 0.0000$)
- Selective risk: **37.6033%** ($\Delta = 0.0000$)
- Retained Macro-F1: **45.5387%** ($\Delta = 0.0000$)
- Retained Macro-Precision: **47.7550%** ($\Delta = 0.0000$)
- Retained Macro-Recall: **50.5991%** ($\Delta = 0.0000$)
- Phase 3.1 baseline macro-precision: **45.55%** (reconciled from the 45.68% transcription error)

---

## 6. Development & Training Reproduction Commands

> [!CAUTION]
> The commands below are **development-only pipelines** that were used during Phases 3.2.3 and 3.2.4. They must NOT be executed as part of standard demonstration or evaluation. The held-out test split must NEVER be evaluated more than once.

### 6.1 Validation Threshold Tuning (Phase 3.2.3)
Explores 1,536 hyperparameter combinations strictly on validation split ($N=255$):
```powershell
python scripts/tune_thresholds.py --canonical-path data/processed/canonical_claims.jsonl --output-file data/processed/threshold_tuning_results.json
```

### 6.2 Validation Calibration & Abstention Policy Tuning (Phase 3.2.4)
Learns temperature $T$ via NLL minimization and sweeps coverage-risk trade-offs strictly on validation split ($N=255$):
```powershell
python scripts/calibrate_and_abstain.py --output-file data/processed/calibration_and_abstention_results.json
```

---

## 7. Known Reproducibility Limitations

1. **VRAM Capacity Limits on Entry-Level GPUs**:
   The RTX 2050 has 4 GB of GDDR6 VRAM. Running batch sizes $> 32$ during dense cross-encoder inference can trigger CUDA out-of-memory errors. The code defaults to conservative batch sizes ($\le 16$).
2. **Local Corpus vs. Live Web Retrieval**:
   Retrieval is intentionally constrained to [`evidence_corpus.jsonl`](file:///d:/VERITAS-X/data/processed/evidence_corpus.jsonl) (1,047 passages). Arbitrary current-event claims not represented in the corpus will properly resolve to `INSUFFICIENT_EVIDENCE`.
3. **Minor Floating-Point Operating System Variance**:
   Minor differences ($< 10^{-6}$) in temperature scaling probabilities may occur across different PyTorch / cuDNN builds on Linux vs. Windows due to non-deterministic BLAS matrix operations, but class argmax decisions remain identical.
