import os
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from unittest import TestCase
from unittest.mock import MagicMock
from veritas.reasoning.multihop_verifier import MultiHopVerifier
from veritas.reasoning.types import SubClaim, SubClaimStatus
from veritas.data.schema import CanonicalVeracity
from veritas.verification.engine import ForensicVerdict


def make_sub_verdict(verdict, confidence=0.85, breakdown=None):
    return ForensicVerdict(
        claim_id="MOCK-SC",
        claim_text="Mock subclaim",
        verdict=verdict,
        confidence=confidence,
        uncalibrated_confidence=confidence,
        was_abstained=False,
        abstention_reason="Confidence sufficient",
        class_probabilities={
            verdict.value: confidence,
            CanonicalVeracity.INSUFFICIENT_EVIDENCE.value: 1.0 - confidence,
        },
        temporal_report=None,
        evidence_used_count=1,
        duplicates_filtered=0,
        irrelevant_passages_filtered=0,
        evidence_breakdown=breakdown or [
            {
                "text": "Evidence for subclaim",
                "entailment_prob": 0.90 if verdict == CanonicalVeracity.VERIFIED_TRUE else 0.05,
                "contradiction_prob": 0.90 if verdict == CanonicalVeracity.VERIFIED_FALSE else 0.05,
                "neutral_prob": 0.05,
                "is_joint": False,
            }
        ],
    )


class TestMultiHopVerifier(TestCase):

    def test_all_supported_subclaims_yield_verified_true(self):
        mock_engine = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = [
            [{"doc_id": "d1", "text": "Bridge closed Monday.", "source_url": "src1", "score": 0.88}],
            [{"doc_id": "d2", "text": "Bridge reopened Tuesday.", "source_url": "src2", "score": 0.91}],
        ]
        mock_engine.verify_claim.side_effect = [
            make_sub_verdict(CanonicalVeracity.VERIFIED_TRUE, 0.85),
            make_sub_verdict(CanonicalVeracity.VERIFIED_TRUE, 0.89),
        ]

        verifier = MultiHopVerifier(engine=mock_engine, retriever=mock_retriever)
        subclaims = [
            SubClaim(subclaim_id="SC-01", text="The bridge closed on Monday"),
            SubClaim(subclaim_id="SC-02", text="The bridge reopened on Tuesday", dependencies=["SC-01"]),
        ]

        result = verifier.verify_compound_claim(
            claim_text="The bridge closed on Monday and reopened on Tuesday.",
            subclaims=subclaims,
        )

        self.assertEqual(result["verdict"], CanonicalVeracity.VERIFIED_TRUE.value)
        self.assertGreater(result["confidence"], 0.80)
        self.assertEqual(result["coverage_ratio"], 1.0)
        self.assertEqual(result["supported_count"], 2)
        self.assertEqual(result["refuted_count"], 0)

    def test_refuted_subclaim_yields_verified_false_with_explanation(self):
        mock_engine = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = [
            [{"doc_id": "d1", "text": "Bridge closed Monday.", "source_url": "src1", "score": 0.88}],
            [{"doc_id": "d2", "text": "Bridge remained strictly closed all week.", "source_url": "src2", "score": 0.95}],
        ]
        mock_engine.verify_claim.side_effect = [
            make_sub_verdict(CanonicalVeracity.VERIFIED_TRUE, 0.85),
            make_sub_verdict(CanonicalVeracity.VERIFIED_FALSE, 0.92),
        ]

        verifier = MultiHopVerifier(engine=mock_engine, retriever=mock_retriever)
        subclaims = [
            SubClaim(subclaim_id="SC-01", text="The bridge closed on Monday"),
            SubClaim(subclaim_id="SC-02", text="The bridge reopened on Tuesday", dependencies=["SC-01"]),
        ]

        result = verifier.verify_compound_claim(
            claim_text="The bridge closed on Monday and reopened on Tuesday.",
            subclaims=subclaims,
        )

        self.assertEqual(result["verdict"], CanonicalVeracity.VERIFIED_FALSE.value)
        self.assertEqual(result["refuted_count"], 1)
        self.assertIn("Compound claim is refuted", result["explanation"])
        self.assertIn("SC-02", str([sc["subclaim_id"] for sc in result["subclaims"] if sc["status"] == "REFUTED"]))

    def test_partially_supported_with_unresolved_yields_insufficient_evidence(self):
        mock_engine = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = [
            [{"doc_id": "d1", "text": "Bridge closed Monday.", "source_url": "src1", "score": 0.88}],
            [],  # No evidence found for subclaim 2
        ]
        mock_engine.verify_claim.return_value = make_sub_verdict(CanonicalVeracity.VERIFIED_TRUE, 0.85)

        verifier = MultiHopVerifier(engine=mock_engine, retriever=mock_retriever)
        subclaims = [
            SubClaim(subclaim_id="SC-01", text="The bridge closed on Monday"),
            SubClaim(subclaim_id="SC-02", text="The secret treasure was hidden under the arch"),
        ]

        result = verifier.verify_compound_claim(
            claim_text="The bridge closed on Monday and the secret treasure was hidden under the arch.",
            subclaims=subclaims,
        )

        self.assertEqual(result["verdict"], CanonicalVeracity.INSUFFICIENT_EVIDENCE.value)
        self.assertEqual(result["unresolved_count"], 1)
        self.assertIn("Partially supported", result["explanation"])
