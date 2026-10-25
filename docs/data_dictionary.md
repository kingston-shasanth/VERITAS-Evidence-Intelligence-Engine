# VERITAS-X: Data Dictionary, Provenance & Benchmark Specification

> **Documentation Index**: [Project README](../README.md) | [Phase 3/4 Verification Report](verification_report.md) | [Phase 2 Retrieval Report](retrieval_report.md) | [Reproducibility Guide](reproducibility.md)

---

## 1. Overview & Data Architecture

VERITAS-X establishes a unified forensic data standard designed to evaluate automated claim verification across diverse languages, real-world web environments, and temporal contexts. 

Rather than forcing incompatible datasets into naive binary classifications, VERITAS-X implements a **Canonical Schema** that preserves the underlying evidentiary structure, speaker context, temporal timestamps, and nuanced truth values (e.g., cherry-picking, missing context, and insufficiency).

```
+--------------------------------------------------------------------------------------------------+
|                                  DATA INGESTION & HARMONIZATION                                  |
|                                                                                                  |
|   [ FEVER (v1.0) ]           [ AVeriTeC (v1.0) ]       [ Indic / Multilingual Benchmarks ]       |
|   185k claims, Wikipedia     4.5k claims, Web Q&A      ~9k claims, En / Hi / Bn / Hinglish       |
|   Labels: SUPPORTS,          Labels: Supported,        Labels: SUPPORTS, REFUTES,                |
|   REFUTES, NOT ENOUGH INFO   Refuted, Conflicting,     NEI / Contextual Stance                   |
|                              Not Enough Evidence                                                 |
|          |                            |                                |                         |
|          v                            v                                v                         |
|   [ fever_loader.py ]        [ averitec_loader.py ]       [ multilingual_loader.py ]             |
|          \                            |                               /                          |
|           ------------------> [ preprocessor.py ] <-------------------                           |
|                               (NFC, Script Tag, Date)                                            |
|                                       |                                                          |
|                                       v                                                          |
|                        [ CanonicalClaim Data Model ]                                             |
|                        - claim_id, text, canonical_label                                         |
|                        - language, date, speaker, evidence                                       |
|                        - deterministic SHA-256 hash                                              |
|                                       |                                                          |
|                                       v                                                          |
|                    data/processed/canonical_claims.jsonl                                         |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Dataset Provenance, Attribution & Licensing

| Dataset | Research Paper & Authors | Venue & Year | Official Repository / Download URL | Licensing Terms | Access Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FEVER (v1.0)** | Thorne et al., *"FEVER: a large-scale dataset for Fact Extraction and VERification"* | NAACL-HLT 2018 | [fever.ai](https://fever.ai/dataset.html)<br>[sheffieldnlp/fever-naacl-2018](https://github.com/sheffieldnlp/fever-naacl-2018) | Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0) | Direct HTTP stream / Open download |
| **AVeriTeC (v1.0)** | Schlichtkrull, Guo, & Vlachos, *"AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web"* | NeurIPS 2023 | [github.com/MichSchli/AVeriTeC](https://github.com/MichSchli/AVeriTeC)<br>[huggingface.co/datasets/chenxwh/AVeriTeC](https://huggingface.co/datasets/chenxwh/AVeriTeC) | Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) | Direct Git / HTTP download |
| **IndicClaimBuster** | Pal, Jana, & Das, *"IndicClaimBuster: A Multilingual Claim Verification Dataset"* | IJCNLP-AACL 2025 | [github.com/pritampal98/indic-claim-buster](https://github.com/pritampal98/indic-claim-buster)<br>[aclanthology.org/2025.ijcnlp-long.133](https://aclanthology.org/2025.ijcnlp-long.133/) | Academic Research License (Paper CC-BY 4.0) | Restricted / Password Gated (Google Form request) |
| **VERITAS-X Multilingual Diagnostic Benchmark** | VERITAS-X Forensic Research Suite | 2026 | Local benchmark at `data/raw/multilingual/diagnostic_multilingual.json` | MIT / Academic Open Access | Bundled directly in workspace repository |

---

## 3. Label Mapping & Reconciled Topology

Different fact-checking initiatives adopt conflicting label names and semantics. VERITAS-X maps all incoming ground truths to a 4-tier forensic classification schema while preserving the untouched `raw_label`:

```
+----------------------------------+-----------------------------+-----------------------------+-----------------------------+
| Unified Canonical Label          | FEVER Raw Ground Truth      | AVeriTeC Raw Ground Truth   | Indic / Diagnostic Truth    |
+----------------------------------+-----------------------------+-----------------------------+-----------------------------+
| VERIFIED_TRUE                    | SUPPORTS                    | Supported                   | SUPPORTS / SUPPORTED / TRUE |
| VERIFIED_FALSE                   | REFUTES                     | Refuted                     | REFUTES / REFUTED / FALSE   |
| CONFLICTING_OR_CHERRYPICKED      | [N/A in FEVER v1]           | Conflicting Evidence /      | CONFLICTING                 |
|                                  |                             | Cherrypicking               |                             |
| INSUFFICIENT_EVIDENCE            | NOT ENOUGH INFO             | Not Enough Evidence         | NOT ENOUGH INFO / NEI       |
+----------------------------------+-----------------------------+-----------------------------+-----------------------------+
```

### Label Definitions:
1. **`VERIFIED_TRUE`**: The claim is fully affirmed by credible, non-contradicted empirical evidence.
2. **`VERIFIED_FALSE`**: The core assertion of the claim is directly contradicted or disproven by credible evidence.
3. **`CONFLICTING_OR_CHERRYPICKED`**: The claim contains an element of truth or genuine quotes, but omits critical context, relies on cherry-picked subsets of data, or is actively contested by conflicting official reports.
4. **`INSUFFICIENT_EVIDENCE`**: Available search evidence neither confirms nor refutes the claim beyond reasonable doubt, or the claim is fundamentally unverified (triggers calibrated model abstention).

---

## 4. Canonical Claim Specification (`CanonicalClaim`)

Exported records in `data/processed/canonical_claims.jsonl` strictly adhere to the following schema:

| Field Name | Type | Nullable | Description & Constraints |
| :--- | :--- | :---: | :--- |
| `claim_id` | `string` | No | Unique identifier with dataset prefix (e.g., `fever_137334`, `averitec_0`, `indic_hi_01`). |
| `claim_text` | `string` | No | Sanitized, NFC-normalized claim assertion. Whitespace and non-printable control characters stripped. |
| `source_dataset` | `string` | No | Enum: `FEVER`, `AVERITEC`, `INDIC_CLAIM_BUSTER`, `MULTILINGUAL_DIAGNOSTIC`. |
| `raw_label` | `string` | No | The untouched original ground-truth label string from the source dataset. |
| `canonical_label`| `string` | No | Enum: `VERIFIED_TRUE`, `VERIFIED_FALSE`, `CONFLICTING_OR_CHERRYPICKED`, `INSUFFICIENT_EVIDENCE`. |
| `language` | `string` | No | ISO 639-1 code: `en` (English), `hi` (Hindi), `bn` (Bengali), `hi-en` (Hinglish code-mixed). |
| `claim_date` | `string` | Yes | ISO 8601 formatted date string (`YYYY-MM-DD`) when the claim was first spoken/published. |
| `speaker` | `string` | Yes | The individual, organization, or medium that uttered the claim (e.g., `Donald Trump`, `WHO`). |
| `evidence` | `array` | No | List of `EvidenceItem` objects containing supporting or refuting snippets. |
| `metadata` | `object` | No | Key-value dictionary containing dataset-specific provenance (e.g., original URLs, justifications). |

---

## 5. Evidence Item Specification (`EvidenceItem`)

Each claim contains an array of evidence items extracted from Wikipedia sentences, investigative Q&A chains, or news passages:

| Field Name | Type | Nullable | Description |
| :--- | :--- | :---: | :--- |
| `text` | `string` | Yes | Extracted text snippet or answer (null in raw FEVER, populated from Wikipedia corpus). |
| `source_url` | `string` | Yes | URL to the original or archived evidence source (Wayback Machine snapshot). |
| `source_id` | `string` | Yes | Internal anchor or sentence identifier (e.g., `Soul_Food_-LRB-film-RRB-#s_0`, `q0_a0`). |
| `evidence_type` | `string` | No | Enum: `WIKIPEDIA_SENTENCE`, `WEB_EXTRACTIVE`, `WEB_ABSTRACTIVE`, `NEWS_PASSAGE`, `UNKNOWN`. |
| `metadata` | `object` | No | Auxiliary context (e.g., question text, medium type, annotation IDs). |

---

## 6. Preprocessing & Script Normalization Standards

To ensure cross-lingual robustness across South Asian scripts, `src/veritas/data/preprocessor.py` implements:

1. **Unicode NFC Composition:**
   Devanagari and Bengali text frequently suffer from encoding fragmentation (e.g., the letter `क़` can be represented as a precomposed character `U+0958` or decomposed into `क` `U+0915` + Nukta `U+093C`). Standardizing to **NFC** composition prevents vocabulary explosion and tokenization mismatches.
2. **Linguistic Script Classification:**
   Character-frequency analysis routes queries across Devanagari (`U+0900–U+097F`), Bengali (`U+0980–U+09FF`), and Latin (`U+0041–U+024F`).
3. **Code-Mixed (Hinglish) Identification:**
   Detects Romanized Hindi social media discourse using phonetic marker matching (`hai`, `nahi`, `aur`, `sarkar`, etc.).
4. **Deterministic SHA-256 Deduplication:**
   Computes a SHA-256 digest over lowercase, whitespace-normalized claim text to detect and eliminate duplicate records within and across splits.

---

## 7. Empirical Data Audit & Validation Findings

A full validation audit was executed across all raw sample datasets. Actual results:

```
=== Global Dataset Health Summary ===
Total Records Processed : 1,111
Valid Canonical Records  : 1,111
Invalid / Malformed      : 0
Duplicates Detected      : 0 (filtered during ingestion)
Missing Critical Fields  : 0

Canonical Label Distribution:
- VERIFIED_FALSE                : 513 (46.17%)
- VERIFIED_TRUE                 : 322 (28.98%)
- INSUFFICIENT_EVIDENCE         : 238 (21.42%)
- CONFLICTING_OR_CHERRYPICKED   :  38  (3.42%)

Language Breakdown:
- English (en)                  : 1,096 (98.65%)
- Hindi (hi)                    :     5  (0.45%)
- Bengali (bn)                  :     5  (0.45%)
- Code-Mixed Hinglish (hi-en)   :     5  (0.45%)
```

### Critical Empirical Finding: AVeriTeC Duplicates
During automated integrity testing of the official `MichSchli/AVeriTeC` development split (`dev.json`, 500 records), the VERITAS-X loader uncovered **9 duplicate claims** in the raw benchmark:
* Multiple identical claims relating to Nigeria's population at independence (e.g., Record 205 duplicates Record 203; Record 207 duplicates Record 204).
* Claims evaluating identical statements on domestic oil pricing in Nigeria.

The VERITAS-X deduplication engine filtered these duplicates, producing **491 clean, unique records** with full evidence integrity preserved.

---

## 8. Known Limitations, Biases & Ethical Considerations

1. **Synthetic Lexical Bias in FEVER:**
   Because FEVER claims were produced by crowdworkers mutating Wikipedia sentences, models can exploit spurious statistical artifacts (e.g., negative words like "never", "refused", "fails" strongly predict `REFUTES` regardless of evidence context).
2. **Real-World Label Imbalance in AVeriTeC:**
   Refuted claims account for over 60% of AVeriTeC dev set claims, reflecting human tendency to fact-check false rumors rather than verify true statements.
3. **IndicClaimBuster Gating:**
   The password requirement on `Dataset.zip` prevents unauthenticated CI/CD pipelines. Our dual strategy (submitting the research form + maintaining open multilingual diagnostic benchmarks) protects development velocity.
4. **Temporal Volatility of Evidence URLs:**
   Live web links decay over time. Forensic pipelines must store cached text snippets (`cached_source_url`) rather than issuing un-cached live HTTP GET requests during verification.

---

## 9. Physical File Map & Repository Data Paths

| File Path | Format | Size / Count | Role & Content |
| :--- | :---: | :---: | :--- |
| `data/raw/fever/sample_dev.jsonl` | JSONL | 600 records | Raw sampled FEVER development claims with Wikipedia sentence coordinate pointers. |
| `data/raw/averitec/sample_dev.json` | JSON | 500 records | Raw sampled AVeriTeC development claims with web Q&A evidence passages and URLs. |
| `data/raw/multilingual/diagnostic_multilingual.json` | JSON | 20 records | Diagnostic multilingual claims across Hindi, Bengali, Hinglish, and English. |
| `data/processed/canonical_claims.jsonl` | JSONL | 1,111 records (1.68 MB) | Unified canonical dataset conforming to `CanonicalClaim` schema with NFC normalization and SHA-256 deduplication. |
| `data/processed/evidence_corpus.jsonl` | JSONL | 1,047 passages (781 KB) | Extracted, deduplicated textual evidence passages used for local BM25 and dense FAISS indexing. |
| `data/processed/validation_model_comparison.json` | JSON | 405 KB | Phase 3.1 4-model validation comparison evaluation records on the 255-claim validation split. |
| `data/processed/test_model_evaluation.json` | JSON | 99.8 KB | Phase 3.1 baseline test evaluation records on the 256-claim held-out test split. |
| `data/processed/threshold_tuning_results.json` | JSON | 13.9 KB | Phase 3.2.3 27-point hyperparameter grid search results on the validation split. |
| `data/processed/calibration_and_abstention_results.json` | JSON | 15.5 KB | Phase 3.2.4 temperature scaling fit ($T=4.1039$) and selective abstention grid sweep results. |
| `data/processed/final_test_evaluation_phase_3_2_5.json` | JSON | 214.8 KB | Phase 3.2.5 final evaluation output containing test logits, calibrated probabilities, abstention flags, and metrics. |

---

## 10. Cross-Phase Navigation & Documentation

* **Downstream Verification Pipeline**: For experimental evaluation of four-class NLI verification, cross-passage stance-disparity detection, temperature scaling calibration, and selective abstention, see [Verification & Audit Report (Phases 3 & 4)](verification_report.md).
* **Sparse, Dense & Hybrid Retrieval**: For BM25, dense E5 FAISS index, and reciprocal rank fusion benchmarks across English and Indic scripts, see [Text Representation and Evidence Retrieval Report (Phase 2)](retrieval_report.md).
* **Hardware & Reproduction Guide**: For exact instructions on environment configuration, deterministic evaluation reproduction, and audit script usage, see [Reproducibility Guide](reproducibility.md).
* **Project Overview & Live Demo**: See [Root README](../README.md).

