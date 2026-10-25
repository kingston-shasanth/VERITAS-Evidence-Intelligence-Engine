"""
Interactive and Single-Claim Verification Demonstration for VERITAS-X.

Demonstrates the frozen Phase 3.2.5 verification pipeline:
- Hybrid Retrieval (Sparse BM25 + Dense Multilingual E5 + RRF) over the local evidence corpus
- Cross-Encoder NLI (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) with dynamic label mapping
- Cross-Passage Stance-Disparity Aggregator (tuned: tau_E=0.35, tau_C=0.50, tau_conf=0.45, m_min=0.10)
- Multiclass Temperature Calibration (T = 4.1039)
- Selective Classification / Abstention Engine (theta_conf = 0.32, theta_H = 0.94)

GOVERNANCE & SAFETY DISCLOSURES:
1. Evidence retrieval is strictly limited to the local offline reproducible index
   (1,047 verified passages in data/processed/evidence_corpus.jsonl). Live web retrieval
   is disabled to guarantee deterministic, leak-free reproducibility.
2. Demo fixtures use only safe non-test public facts and synthetic demonstrations.
   No held-out test split claims (N=256) are used as demo fixtures.
"""
import os
import sys
import argparse
import time
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from veritas.data.schema import CanonicalVeracity
from veritas.retrieval.evidence_corpus import EvidenceCorpus
from veritas.retrieval.bm25_retriever import BM25Retriever
from veritas.retrieval.dense_retriever import DenseRetriever
from veritas.retrieval.hybrid_retriever import HybridRetriever
from veritas.verification.nli_model import CrossEncoderNLI, resolve_device
from veritas.verification.evidence_aggregator import EvidenceAggregator
from veritas.verification.calibrator import TemperatureCalibrator, SelectiveAbstainer
from veritas.verification.temporal_analyzer import TemporalAnalyzer
from veritas.verification.engine import VerificationEngine

FROZEN_AGGREGATOR_PARAMS = {
    "disparity_entail_threshold": 0.35,
    "disparity_contra_threshold": 0.50,
    "conflict_threshold": 0.45,
    "min_informative_mass": 0.10,
    "joint_weight_disparity": 0.40,
    "joint_weight_default": 1.20,
}

FROZEN_TEMPERATURE = 4.1039
FROZEN_CONF_THRESH = 0.32
FROZEN_ENT_THRESH = 0.94

# Curated safe non-test demonstration fixtures
SAFE_DEMO_FIXTURES = [
    {
        "id": "demo_true",
        "title": "Scientific Law (Entailed Fact)",
        "claim": "Pure water freezes at 0 degrees Celsius at sea level.",
        "date": "2024-01-01",
        "explicit_evidence": [
            {
                "doc_id": "sci_ref_01",
                "text": "Under standard atmospheric pressure at sea level, the freezing point of pure water is precisely 0 degrees Celsius.",
                "source_url": "https://britannica.com/science/water",
            }
        ],
    },
    {
        "id": "demo_false",
        "title": "Direct Refutation (Contradicted Myth)",
        "claim": "The Earth is flat and supported on the back of giant elephants.",
        "date": "2024-01-01",
        "explicit_evidence": [
            {
                "doc_id": "sci_ref_02",
                "text": "Geodesy, circumnavigation, and orbital satellite photography prove that the Earth is an oblate spheroid, not flat.",
                "source_url": "https://nasa.gov/earth",
            }
        ],
    },
    {
        "id": "demo_conflicting",
        "title": "Cross-Passage Stance Disparity (Conflicting Evidence)",
        "claim": "The local municipal tax reform was beneficial to the community.",
        "date": "2024-01-01",
        "explicit_evidence": [
            {
                "doc_id": "report_pos",
                "text": "A comprehensive economic review confirmed that the local municipal tax reform was highly beneficial to the community, boosting small businesses.",
                "source_url": "https://chamber-commerce.org/report",
            },
            {
                "doc_id": "report_neg",
                "text": "An independent financial audit concluded that the municipal tax reform was not beneficial and caused severe financial distress to local residents.",
                "source_url": "https://citizens-audit.org/report",
            },
        ],
    },
    {
        "id": "demo_insufficient",
        "title": "Unsubstantiated Speculation (Insufficient Evidence)",
        "claim": "Extraterrestrial spacecraft landed in the central plaza yesterday.",
        "date": "2024-01-01",
        "explicit_evidence": [],  # Will rely on retrieval; zero matches in corpus
    },
]


def build_frozen_verification_engine(
    corpus_path: str,
    device_override: Optional[str] = None,
) -> VerificationEngine:
    """
    Constructs the VerificationEngine using the verified, frozen Phase 3.2.5 pipeline.
    Uses existing interfaces without modifying production source code.
    """
    device = resolve_device(device_override)
    print(f"[*] Initializing CrossEncoderNLI model on compute device: {device}...")
    t0 = time.time()
    nli_model = CrossEncoderNLI(
        model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device=device,
    )
    print(f"    NLI model loaded in {time.time() - t0:.2f}s (Device: {nli_model.device})")

    # Load local corpus and initialize hybrid retriever
    hybrid_retriever = None
    if os.path.exists(corpus_path):
        print(f"[*] Loading local reproducible EvidenceCorpus from: {corpus_path}...")
        corpus = EvidenceCorpus.load_from_jsonl(corpus_path)
        print(f"    Loaded {len(corpus)} verified evidence passages.")
        bm25 = BM25Retriever(corpus)
        dense = DenseRetriever(corpus, device=device)
        hybrid_retriever = HybridRetriever(bm25, dense, rrf_k=60)
        print("    Hybrid Retriever (BM25 + Dense Multilingual E5 + RRF) ready.")
    else:
        print(f"[!] Warning: Evidence corpus not found at {corpus_path}. Retrieval from corpus disabled.")

    # Instantiate frozen components
    frozen_aggregator = EvidenceAggregator(**FROZEN_AGGREGATOR_PARAMS)
    frozen_calibrator = TemperatureCalibrator(temperature=FROZEN_TEMPERATURE)
    frozen_abstainer = SelectiveAbstainer(
        confidence_threshold=FROZEN_CONF_THRESH,
        entropy_threshold=FROZEN_ENT_THRESH,
    )

    engine = VerificationEngine(
        nli_model=nli_model,
        retriever=hybrid_retriever,
        calibrator=frozen_calibrator,
        abstainer=frozen_abstainer,
        device=device,
    )
    # Inject frozen tuned aggregator
    engine.aggregator = frozen_aggregator

    return engine


def run_claim_demonstration(
    engine: VerificationEngine,
    claim_text: str,
    claim_date: Optional[str] = None,
    explicit_evidence: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 5,
):
    """Executes end-to-end verification and prints structured forensic output."""
    print("\n" + "=" * 80)
    print("VERITAS-X FORENSIC CLAIM ADJUDICATION")
    print("=" * 80)
    print(f"Claim Text : \"{claim_text}\"")
    if claim_date:
        print(f"Claim Date : {claim_date}")

    t_start = time.time()

    # Step 1: Evidence Acquisition
    if explicit_evidence is not None:
        passages = explicit_evidence
        print(f"\n[1. Evidence Acquisition] Provided {len(passages)} explicit evidence passage(s).")
    elif engine.retriever is not None:
        print(f"\n[1. Evidence Acquisition] Searching local reproducible corpus (top-{top_k} RRF)...")
        results = engine.retriever.search(claim_text, top_k=top_k)
        passages = [r.to_dict() for r in results]
        print(f"    Retrieved {len(passages)} candidate passage(s) via BM25 + Dense E5 RRF.")
    else:
        passages = []
        print("\n[1. Evidence Acquisition] No evidence provided and no retriever available.")

    # Disclosure banner
    if not explicit_evidence:
        print("    [Notice: Retrieval is restricted to the local offline corpus (1,047 verified passages).")
        print("             No live unauthenticated web queries are executed.]")

    # Step 2: Verification Engine Execution
    verdict = engine.verify_claim(
        claim_text=claim_text,
        claim_date=claim_date,
        candidate_passages=passages,
        top_k=top_k,
    )
    duration = time.time() - t_start

    # Step 3: Display Passages & Pairwise Stance Breakdown
    print("\n[2. Evidence Stance & Cross-Encoder NLI Breakdown]")
    if not verdict.evidence_breakdown:
        print("    (Zero informative passages available)")
    else:
        for idx, ev in enumerate(verdict.evidence_breakdown, 1):
            doc_id = ev.get("doc_id", f"ev_{idx}")
            txt = (ev.get("text") or "").strip()
            if len(txt) > 120:
                txt = txt[:117] + "..."
            e_p = ev.get("entailment_prob", 0.0)
            c_p = ev.get("contradiction_prob", 0.0)
            n_p = ev.get("neutral_prob", 0.0)
            supp = " [Active Supporter]" if ev.get("is_active_supporter") else ""
            ref = " [Active Refuter]" if ev.get("is_active_refuter") else ""
            print(f"    Passage {idx} ({doc_id}){supp}{ref}:")
            print(f"      Text: \"{txt}\"")
            print(f"      Stance Probabilities: Entailment={e_p*100:5.1f}% | Contradiction={c_p*100:5.1f}% | Neutral={n_p*100:5.1f}%")

    # Step 4: Stance Disparity Status
    print("\n[3. Cross-Passage Stance Disparity Detection]")
    # Check aggregator internals
    has_disparity = getattr(engine.aggregator, "has_cross_passage_disparity", False)
    # The aggregator returns AggregatedEvidenceVerdict internally during verify_claim
    print(f"    Aggregator Thresholds: Entailment >= {FROZEN_AGGREGATOR_PARAMS['disparity_entail_threshold']}, Contradiction >= {FROZEN_AGGREGATOR_PARAMS['disparity_contra_threshold']}")
    print(f"    Joint Premise Disparity Multiplier: {FROZEN_AGGREGATOR_PARAMS['joint_weight_disparity']} (vs Default {FROZEN_AGGREGATOR_PARAMS['joint_weight_default']})")

    # Step 5: Probabilities & Calibration
    print("\n[4. Multiclass Probability Distribution & Calibration]")
    print(f"    Learned Temperature : T = {FROZEN_TEMPERATURE:.4f} (Fitted strictly on validation split)")
    print("    Calibrated Class Probabilities:")
    for cname in engine.LABEL_NAMES:
        prob = verdict.class_probabilities.get(cname, 0.0)
        marker = " <--- Top Prediction" if cname == verdict.verdict.value else ""
        print(f"      {cname:<28s}: {prob*100:6.2f}%{marker}")

    # Step 6: Selective Abstention Decision
    print("\n[5. Selective Classification / Abstention Assessment]")
    print(f"    Confidence Threshold : {FROZEN_CONF_THRESH:.2f} | Entropy Threshold: {FROZEN_ENT_THRESH:.2f}")
    if verdict.was_abstained:
        print(f"    Abstention Status    : ABSTAINED (Decision deferred to human forensic review)")
        print(f"    Abstention Reason    : {verdict.abstention_reason}")
    else:
        print(f"    Abstention Status    : RETAINED (Model asserts categorical prediction)")

    # Step 7: Temporal Audit Status
    print("\n[6. Temporal Consistency Audit]")
    print(f"    Temporal Status      : {verdict.temporal_report.status.value}")
    if verdict.temporal_report.warning_message:
        print(f"    Temporal Warning     : {verdict.temporal_report.warning_message}")
        print(f"    Time Discrepancy     : {verdict.temporal_report.delta_days} days")

    # Final Verdict Summary
    print("\n" + "=" * 80)
    print(f"FINAL FORENSIC VERDICT : {verdict.verdict.value}")
    print(f"Calibrated Confidence  : {verdict.confidence*100:.2f}% (Uncalibrated: {verdict.uncalibrated_confidence*100:.2f}%)")
    print(f"Processing Duration    : {duration:.3f}s")
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VERITAS-X Frozen Phase 3.2.5 Claim Verification Demonstration."
    )
    parser.add_argument(
        "--claim",
        type=str,
        default=None,
        help="Claim text to verify. If omitted, launches interactive selection.",
    )
    parser.add_argument(
        "--evidence",
        type=str,
        default=None,
        help="Optional explicit evidence passage text (bypasses corpus search).",
    )
    parser.add_argument(
        "--claim-date",
        type=str,
        default=None,
        help="Optional claim publication date in YYYY-MM-DD format for temporal checking.",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "evidence_corpus.jsonl"),
        help="Path to local reproducible evidence corpus.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (auto-detects CUDA).",
    )
    parser.add_argument(
        "--fixture",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Run a specific curated safe non-test demonstration fixture (1=True, 2=False, 3=Conflicting, 4=Insufficient).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    engine = build_frozen_verification_engine(
        corpus_path=args.corpus_path,
        device_override=args.device,
    )

    if args.fixture is not None:
        fix = SAFE_DEMO_FIXTURES[args.fixture - 1]
        print(f"\n[*] Executing Safe Demo Fixture #{args.fixture}: {fix['title']}")
        run_claim_demonstration(
            engine=engine,
            claim_text=fix["claim"],
            claim_date=fix["date"],
            explicit_evidence=fix["explicit_evidence"] if fix["explicit_evidence"] else None,
        )
        return

    if args.claim:
        explicit_ev = None
        if args.evidence:
            explicit_ev = [{
                "doc_id": "cli_supplied_evidence",
                "text": args.evidence,
                "source_url": "user_cli_input",
            }]
        run_claim_demonstration(
            engine=engine,
            claim_text=args.claim,
            claim_date=args.claim_date,
            explicit_evidence=explicit_ev,
        )
        return

    # Interactive demonstration menu
    print("\n" + "=" * 80)
    print("VERITAS-X INTERACTIVE DEMONSTRATION MENU")
    print("=" * 80)
    print("Select an option:")
    print("  1. Run Curated Fixture 1: Scientific Law (VERIFIED_TRUE)")
    print("  2. Run Curated Fixture 2: Direct Refutation (VERIFIED_FALSE)")
    print("  3. Run Curated Fixture 3: Opposing Reports (CONFLICTING_OR_CHERRYPICKED)")
    print("  4. Run Curated Fixture 4: Unsubstantiated Speculation (INSUFFICIENT_EVIDENCE)")
    print("  5. Enter a custom claim interactively")
    print("  0. Exit")

    choice = input("\nEnter choice (0-5) [default=1]: ").strip()
    if not choice:
        choice = "1"

    if choice in ("1", "2", "3", "4"):
        fix = SAFE_DEMO_FIXTURES[int(choice) - 1]
        print(f"\n[*] Running Demo Fixture: {fix['title']}")
        run_claim_demonstration(
            engine=engine,
            claim_text=fix["claim"],
            claim_date=fix["date"],
            explicit_evidence=fix["explicit_evidence"] if fix["explicit_evidence"] else None,
        )
    elif choice == "5":
        user_claim = input("Enter claim text: ").strip()
        if not user_claim:
            print("No claim entered. Exiting.")
            return
        user_ev = input("Enter explicit evidence passage (press Enter to search local corpus): ").strip()
        explicit_ev = None
        if user_ev:
            explicit_ev = [{
                "doc_id": "interactive_user_evidence",
                "text": user_ev,
                "source_url": "user_prompt_input",
            }]
        run_claim_demonstration(
            engine=engine,
            claim_text=user_claim,
            explicit_evidence=explicit_ev,
        )
    else:
        print("Exiting demonstration.")


if __name__ == "__main__":
    main()
