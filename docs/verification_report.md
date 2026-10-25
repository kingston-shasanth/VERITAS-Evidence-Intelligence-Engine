# VERITAS-X: Evidence Verification, Stance Disparity & Uncertainty Calibration Report (Phase 3 & Phase 4)

---

## 1. Executive Summary & Objective

This report details the architectural modernization, hyperparameter tuning, post-hoc calibration, and held-out test evaluation of the **VERITAS-X Evidence Verification Pipeline**. 

The verification subsystem adjudicates the veracity of natural language claims given candidate evidence passages retrieved via hybrid BM25 + dense E5 reciprocal rank fusion (RRF) or human-verified question-answer premises. The system is designed to resolve three persistent failure modes in contemporary automated fact-checking:
1. **Severe False-Positive Conflict Rates**: Treating any non-entailing premise as active contradiction, which misclassifies nuanced false claims as conflicting.
2. **Joint Premise Contradiction Bias**: Concatenating multiple premises into a single context window, allowing contradiction bias to overwhelm subtle supporting evidence.
3. **Severe Neural Overconfidence**: Using uncalibrated raw softmax probabilities from deep neural models, yielding high Expected Calibration Error (ECE $> 20\%$).

All reported metrics derive from strictly separated evaluation tiers:
- **Validation Split ($N=255$)**: Used for model selection, threshold tuning, and temperature fitting.
- **Held-Out Test Split ($N=256$)**: Evaluated strictly once with all parameters frozen.
- **Post-Hoc Read-Only Audit**: Recomputed cell-by-cell from raw prediction outputs to confirm mathematical consistency.

---

## 2. Four-Class Veracity Taxonomy

VERITAS-X enforces an unambiguous four-class classification schema that preserves the forensic distinction between contradictory and unevidenced claims:

| Class | Forensic Definition | Activation Condition |
| :--- | :--- | :--- |
| **`VERIFIED_TRUE`** | Evidence factually entails and substantiates the claim. | $M_{\text{entail}} > M_{\text{contra}}$ and $M_{\text{total}} \ge m_{\text{min}}$ |
| **`VERIFIED_FALSE`** | Evidence directly refutes or contradicts the claim. | $M_{\text{contra}} > M_{\text{entail}}$ and $M_{\text{total}} \ge m_{\text{min}}$ |
| **`CONFLICTING_OR_CHERRYPICKED`** | Distinct evidence passages present opposing conclusions, or selective quoting conceals critical counter-evidence. | Active cross-passage disparity ($\tau_E \ge 0.35, \tau_C \ge 0.50, i \ne j$) and conflict ratio $\ge 0.45$ |
| **`INSUFFICIENT_EVIDENCE`** | Evidence lacks necessary facts, is wholly neutral, or mass falls below minimum threshold. | $M_{\text{total}} < m_{\text{min}} = 0.10$ |

### Meta-Level Selective Abstention (`ABSTAIN`)
Distinct from the substantive domain class `INSUFFICIENT_EVIDENCE`, **Selective Abstention** is a meta-level classification rejection mechanism ($g(x) \in \{0, 1\}$; Geifman & El-Yaniv, NeurIPS 2017). When the model's confidence is below threshold ($\theta_{\text{conf}} = 0.32$) or normalized entropy exceeds threshold ($\theta_H = 0.94$), the system declines to make ANY categorical assertion and routes the claim to human review. Abstained claims are evaluated by **Coverage** and **Selective Risk** (error rate among retained claims).

---

## 3. Architectural Design & Methodology

```
+---------------------------------------------------------------------------------------------------+
|                                  VERITAS-X VERIFICATION ENGINE                                    |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                    [Claim + Evidence Passages]
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
              [Temporal Forensics Engine]                 [Passage Lexical De-duplication]
            - Regex & ISO-8601 Parser                   - Jaccard Lexical Overlap (tau = 0.75)
            - Delta vs Claim Date Audit                 - Duplicate passage suppression
            - Anachronism Warning                       - Neutral mass filtering (tau = 0.70)
                         |                                                 |
                         |                                                 v
                         |                                      [Pairwise Cross-Encoder]
                         |                            (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
                         |                            Dynamic Label Mapping (resolve_nli_label_mapping)
                         |                            Premise: Passage | Hypothesis: Claim
                         |                                                 |
                         |                                                 v
                         |                            [Cross-Passage Stance Disparity Detector]
                         |                            - Active Supporter: e_i >= 0.35, e_i > c_i
                         |                            - Active Refuter:   c_j >= 0.50, c_j > e_j
                         |                            - Distinct Passages: i != j
                         |                            - De-weighted Joint Premise (w = 0.40 vs 1.20)
                         |                                                 |
                         |                                                 v
                         |                                 [Stance Consensus Aggregator]
                         |                                 - Informative Mass Accumulation
                         |                                 - Conflict Ratio: 2*min(E,C)/(E+C)
                         |                                 - Minimum Mass Gate: mass >= 0.10
                         |                                                 |
                         |                                                 v
                         |                                    [Four-Class Raw Logits]
                         |                                 [z_TRUE, z_FALSE, z_CONF, z_INSUF]
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
                                     [Multiclass Temperature Scaling]
                                        p_i = exp(z_i / T) / sum exp(z_j / T)
                                                T = 4.1039
                                                  |
                                                  v
                                     [Selective Classification Engine]
                                     - max(p) < 0.320?
                                     - Normalized Entropy > 0.940?
                                                  |
                                  +---------------+---------------+
                                  |                               |
                           Retained (94.5%)               Abstained (5.5%)
                                  |                               |
                                  v                               v
                        [Categorical Verdict]          [Deferred to Human Review]
                       (TRUE, FALSE, CONF, INSUF)      (Selective Risk Evaluation)
```

### 3.1 Cross-Encoder NLI & Dynamic Label Resolution
The primary inference backbone is `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (~86M parameters). 

> [!IMPORTANT]
> **Label Inversion Bug Mitigation**: Different fine-tuned NLI checkpoints assign different index mappings to classification heads. For example, `MoritzLaurer` assigns `{entailment: 0, neutral: 1, contradiction: 2}`, whereas legacy DeBERTa checkpoints assign `{contradiction: 0, entailment: 1, neutral: 2}`. Hardcoded index assumptions silently invert entailment and contradiction. VERITAS-X implements `resolve_nli_label_mapping`, which queries `model.config.id2label` dynamically and validates semantic keys explicitly.

### 3.2 Cross-Passage Stance-Disparity Detection
In standard summation-based aggregators, noisy single-passage neutral-contradiction fluctuations falsely trigger conflict ratios. The Phase 3.2.2 aggregator requires distinct evidence passages taking active opposing stances:
1. **Active Supporter**: Passage $p_i$ has entailment $e_i \ge \tau_E = 0.35$, $e_i > c_i$, and neutral $n_i < 0.70$.
2. **Active Refuter**: Passage $p_j$ has contradiction $c_j \ge \tau_C = 0.50$, $c_j > e_j$, and neutral $n_j < 0.70$.
3. **Distinct Evidence Condition**: The two passages must be distinct ($i \ne j$). A single document cannot satisfy both conditions.
4. **Joint Premise De-weighting**: When disparity is detected, the joint concatenated premise multiplier is reduced from $w_{\text{def}} = 1.20$ down to $w_{\text{disp}} = 0.40$, preventing joint contradiction bias from smothering the supporting passage.

### 3.3 Post-Hoc Multiclass Temperature Scaling
Synthesized evidence logits span $[-17.85, +11.00]$, producing uncalibrated overconfidence. We deploy multiclass temperature scaling (Guo et al., ICML 2017) using:
$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_{j=1}^4 \exp(z_j / T)}, \quad \text{with } T = 4.1039$$

#### Mathematical Properties & Distinction from Affine Platt Scaling:
- **Argmax & Rank Preservation**: For any positive scalar $T > 0$, $\arg\max_i (z_i / T) = \arg\max_i z_i$. Temperature scaling strictly preserves classification predictions and class orderings.
- **Distinction from Affine Platt Scaling**: Multiclass temperature scaling modifies logits using a single positive scalar divisor `softmax(logits / T)` with $T = 4.1039$, fitted via negative log-likelihood (NLL) minimization on validation data ($N=255$). In contrast, Platt Scaling (Platt, 1999) originally refers to logistic regression on binary decision margins ($P(y=1|f) = \frac{1}{1 + \exp(Af + B)}$), and matrix/vector scaling (often termed multiclass Platt scaling) fits affine transformations $\mathbf{W}\mathbf{z} + \mathbf{b}$, requiring $K(K+1)$ parameters and altering relative class rankings. Temperature scaling must not be conflated with affine Platt scaling.

---

## 4. Experimental Results & Benchmark Audit

### 4.1 Phase 3.1 Model Selection Diagnostic (Validation Split, $N=255$)
Four cross-encoder NLI architectures were evaluated across two passes: Pass 1 (raw answers alone) and Pass 2 (contextualized Q+A + joint premise):

| Model Checkpoint | Parameters | Val Acc (Pass 1 $\rightarrow$ Pass 2) | Val Macro-F1 (Pass 1 $\rightarrow$ Pass 2) | Selected? |
| :--- | :---: | :---: | :---: | :---: |
| Majority Class Baseline (`VERIFIED_FALSE`) | — | 61.18% $\rightarrow$ 61.18% | 18.98% $\rightarrow$ 18.98% | No |
| `cross-encoder/nli-deberta-v3-xsmall` | 22M | 53.73% $\rightarrow$ 45.10% | 35.11% $\rightarrow$ 32.44% | No |
| `cross-encoder/nli-distilroberta-base` | 82M | 53.73% $\rightarrow$ 49.80% | 31.26% $\rightarrow$ 33.21% | No |
| `cross-encoder/nli-deberta-v3-base` | 86M | 51.76% $\rightarrow$ 56.08% | 33.31% $\rightarrow$ 38.91% | No |
| **`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`** | 86M | **48.24% $\rightarrow$ 57.65%** | **37.36% $\rightarrow$ 43.16%** | **YES (Winner)** |

---

### 4.2 Phase 3.2.3 Validation Threshold Tuning ($N=255$)
A grid search over 1,536 hyperparameter combinations was conducted on cached validation NLI scores.

| Configuration | Accuracy | Macro-F1 | Macro-P | Macro-R | Conf. Precision | Conf. Recall | Coverage | Retained |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Default Aggregator (P3.2.2) | 57.25% | 43.27% | 48.24% | 50.12% | 10.3% | 15.8% | 71.76% | 183 |
| **Tuned Optimal Aggregator** | **59.22%** | **44.46%** | **49.87%** | **50.88%** | **33.3%** | **10.5%** | **70.59%** | **180** |

*Tuned values: $\tau_E=0.35, \tau_C=0.50, \tau_{\text{conf}}=0.45, m_{\text{min}}=0.10, w_{\text{disp}}=0.40, w_{\text{def}}=1.20$.*

---

### 4.3 Phase 3.2.4 Validation Calibration Diagnostics ($N=255$)
Scalar temperature $T = 4.1039$ was fitted strictly on validation split logits via bounded NLL minimization:

| Calibration Metric | Uncalibrated ($T=1.0$) | Calibrated ($T=4.1039$) | Relative Change |
| :--- | :---: | :---: | :---: |
| **Negative Log-Likelihood (NLL)** | 1.7542 | **0.9627** | **$-45.1\%$** |
| **Expected Calibration Error (ECE, 10 bins)** | 19.21% | **11.83%** | **$-38.4\%$** |
| **Multiclass Brier Score** | 0.5181 | **0.4988** | **$-3.7\%$** |

---

### 4.4 Phase 3.2.5 Final Held-Out Test Evaluation ($N=256$, Evaluated Strictly Once)

The final frozen pipeline was evaluated once on the untouched held-out test split ($N=256$). All metrics derive from [`data/processed/final_test_evaluation_phase_3_2_5.json`](file:///d:/VERITAS-X/data/processed/final_test_evaluation_phase_3_2_5.json):

#### Overall Performance Comparison
| Metric | Phase 3.1 Baseline | Phase 3.2.5 Full-Test ($N=256$) | Difference vs. Baseline | Phase 3.2.5 Selective ($N=242$, Coverage: 94.53%) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 57.42% (147/256) | **59.38%** (152/256) | **+1.95 pp (+1.96 pp rounded)** | **62.40%** (151/242) |
| **Macro-F1 (4-Class)** | 44.24% | **43.34%** | **-0.90 percentage points** | **45.54%** |
| **Macro-Precision** | 45.55% *(reconciled)* | **46.91%** | **+1.36 percentage points** | **47.76%** |
| **Macro-Recall** | 50.34% | **49.07%** | **-1.27 percentage points** | **50.60%** |
| **Coverage ($g(x)=1$)** | 100.0% | 100.0% | — | **94.53%** (242/256) |
| **Error Rate / Selective Risk** | 42.58% *(Full error)* | **40.62%** *(Full error)* | **-1.95 pp (+1.96 pp rounded)** | **37.60%** *(Selective risk)* |


#### Multiclass Calibration Diagnostics (Phase 3.2.5 Test Logits: Before vs. After Temperature Scaling, $N=256$)

The calibration metrics evaluate the effect of multiclass temperature scaling ($\sigma_{\text{SM}}(\mathbf{z}/T)$ with scalar $T=4.1039$, fitted strictly on validation split NLL) on the saved Phase 3.2.5 held-out test logits. These figures represent a **before-versus-after calibration evaluation on the exact same Phase 3.2.5 test logits**, not a comparison against the Phase 3.1 baseline (Phase 3.1 baseline logits were not calibrated). Temperature scaling adjusts prediction confidence to reflect empirical accuracy without altering logit rank order or classification decisions.

| Calibration Metric | Phase 3.2.5 test logits — uncalibrated (T=1) | Phase 3.2.5 test logits — calibrated (T=4.1039) | Difference (Before vs. After Calibration) |
| :--- | :---: | :---: | :---: |
| **Negative Log-Likelihood (NLL)** | 1.9077 (1.907682) | **0.9915** (0.991533) | **-48.0% relative** ($-0.9161$) |
| **Multiclass Brier Score** | 0.5774 (0.577370) | **0.5226** (0.522609) | **-9.5% relative** ($-0.0548$) |
| **Expected Calibration Error (ECE, 10 bins)** | 23.10% (0.231014) | **10.60%** (0.106013) | **-12.50 percentage points** |

#### Per-Class Performance Breakdown (Full Test $N=256$ vs. Retained Claims $N=242$, Coverage: 94.53%)
| Class Label | Support (Full / Ret) | Phase 3.1 Baseline (P / R / F1) | Phase 3.2.5 Full-Test ($N=256$) (P / R / F1) | Phase 3.2.5 Retained ($N=242$, Coverage: 94.53%) (P / R / F1) |
| :--- | :---: | :---: | :---: | :---: |
| `VERIFIED_TRUE` | 61 / 57 | 64.44% / 47.54% / **54.72%** | 57.69% / 49.18% / **53.10%** | 57.69% / 52.63% / **55.05%** |
| `VERIFIED_FALSE` | 157 / 150 | 81.45% / 64.33% / **71.89%** | 81.68% / 68.15% / **74.31%** | 81.68% / 71.33% / **76.16%** |
| `CONFLICTING_OR_CHERRYPICKED` | 19 / 17 | 14.29% / 21.05% / **17.02%** | **28.57%** / 10.53% / **15.38%** | **28.57%** / 11.76% / **16.67%** |
| `INSUFFICIENT_EVIDENCE` | 19 / 18 | 22.03% / 68.42% / **33.33%** | 19.70% / 68.42% / **30.59%** | 23.08% / 66.67% / **34.29%** |

#### Confusion Matrices
**Full Test Set ($N=256$, Rows = Ground Truth, Cols = Predicted `[TRUE, FALSE, CONF, INSUF]`):**
```
                  Pred_TRUE   Pred_FALSE   Pred_CONF   Pred_INSUF   Support
True_TRUE (61)   :  [   30,         12,          2,         17    ]    61
True_FALSE (157) :  [   17,        107,          2,         31    ]   157
True_CONF (19)   :  [    3,          9,          2,          5    ]    19
True_INSUF (19)  :  [    2,          3,          1,         13    ]    19
```

**Retained Predictions ($N=242$, After Selective Abstention, Coverage: 94.53%):**
```
                  Pred_TRUE   Pred_FALSE   Pred_CONF   Pred_INSUF   Support
True_TRUE (57)   :  [   30,         12,          2,         13    ]    57 (4 abstained)
True_FALSE (150) :  [   17,        107,          2,         24    ]   150 (7 abstained)
True_CONF (17)   :  [    3,          9,          2,          3    ]    17 (2 abstained)
True_INSUF (18)  :  [    2,          3,          1,         12    ]    18 (1 abstained)
```

---

## 5. Post-Hoc Mathematical Audit & Baseline Reconciliation

A read-only post-hoc numerical audit script ([`scratch/run_audit.py`](file:///d:/VERITAS-X/scratch/run_audit.py)) was executed directly on the saved artifacts:

1. **Exact Cell-by-Cell Alignment**: Every entry in the full-test and retained confusion matrices matches recomputed predictions with zero difference.
2. **Reconciliation of Phase 3.1 Baseline Precision**:
   - The Phase 3.1 test confusion matrix in [`test_model_evaluation.json`](file:///d:/VERITAS-X/data/processed/test_model_evaluation.json) has column sums $[45, 124, 28, 59]$ and diagonal entries $[29, 101, 4, 13]$.
   - This yields per-class precisions of $64.44\%$, $81.45\%$, $14.29\%$, and $22.03\%$.
   - The exact arithmetic macro-precision is $\mathbf{45.55\%}$. The value $45.68\%$ previously reported was an erroneous narrative transcription that is now formally reconciled.
3. **Probability & Calibration Verification**:
   - Computing $\text{softmax}(\mathbf{z} / 4.1039)$ on the raw Phase 3.2.5 test logits reproduced the saved calibrated probabilities with bitwise precision (max delta $= 0.0000 \times 10^0$).
   - Evaluating the same Phase 3.2.5 test logits before ($T=1$) and after temperature scaling ($T=4.1039$) confirmed: NLL fell from $1.9077$ to $0.9915$ ($-48.0\%$), Brier score fell from $0.5774$ to $0.5226$ ($-9.5\%$), and ECE fell from $23.10\%$ to $10.60\%$ ($-12.50\text{ pp}$). These are before-vs-after improvements on the same Phase 3.2.5 test logits, not comparisons to Phase 3.1.

---

## 6. Honest Limitation Analysis: Conflicting-Class Recall & Macro-F1

### The Precision-Recall Trade-off in Stance Disparity
- **Precision Gain**: Conflicting precision doubled from **14.29% to 28.57%** (+14.29 percentage points). False positive conflict alarms fell from 24 in Phase 3.1 down to **5** in Phase 3.2.5 (a $79\%$ reduction in false alarms).
- **Recall Deficit**: Conflicting recall dropped from **21.05% (4/19) down to 10.53% (2/19)** (-10.53 percentage points).
- **Macro-F1 Impact**: Due to the severe penalty on conflicting recall and slight shift in true-positive balance, overall full-test Macro-F1 decreased from **44.24% to 43.34%** (-0.90 percentage points), even while full-test accuracy increased from **57.42% to 59.38%** (+1.95 pp, +1.96 pp rounded). Under selective classification ($N=242$, coverage $94.53\%$), retained Macro-F1 recovered to **45.54%** (+1.30 percentage points vs. Phase 3.1 baseline).
- **Root Cause**: Real-world conflicting claims frequently embed evidence within a single paragraph or rely on implicit counter-claims that fail the strict distinct-passage thresholds ($\tau_E \ge 0.35, \tau_C \ge 0.50, i \ne j$). The conservative aggregator prioritized eliminating false alarms on false claims (reducing false-as-conflicting from 17 to 2), which suppressed detection on borderline conflicting examples.
- **Academic Recommendation**: Future work should explore intra-passage clause-level segmentation and discourse-level contradiction modeling rather than relying solely on whole-document cross-passage disparity.

---

## 7. Dataset Attribution & Academic References

1. **AVeriTeC**: Schoch, M., Guo, Y., & Vlachos, A. (2023). *AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web*. NeurIPS 2023. License: CC BY 4.0.
2. **FEVER**: Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). *FEVER: a large-scale dataset for Fact Extraction and VERification*. NAACL-HLT 2018. License: CC BY-SA 3.0.
3. **DeBERTa-v3 NLI**: Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K. (2023). *Less Annotating, More Classifying: Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and NLI*. Political Analysis. License: MIT / Apache 2.0.
4. **Multilingual E5**: Wang, L., et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training*. arXiv:2212.03533. License: MIT.
5. **Temperature Calibration**: Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML 2017.
6. **Selective Classification**: Geifman, Y., & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks*. NeurIPS 2017.
