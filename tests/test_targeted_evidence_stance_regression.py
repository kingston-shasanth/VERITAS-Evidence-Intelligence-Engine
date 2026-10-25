"""
Targeted Regression Test Suite for VERITAS-X Evidence Stance & Verdict Aggregation.
Covers:
- Test A: Incognito tracking consistency (Claim A must not be VERIFIED_TRUE when evidence refutes it)
- Test B: Incognito IP visibility consistency (Claim B must remain VERIFIED_FALSE)
- Test C: Stance polarity matrix (entailment, explicit contradiction, neutral, negated/absolute terms)
- Test D: Aggregation consistency (refutations cannot contribute to support count; verdict strictly follows eligible evidence)
- Test E: Abstention (insufficient eligible evidence does not produce a confident definitive verdict)
"""
import os
import sys
import unittest
from typing import List, Dict, Any
from unittest.mock import MagicMock

# Ensure repo root and src/ are on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from veritas.data.schema import CanonicalVeracity
from veritas.verification.engine import VerificationEngine
from veritas.verification.evidence_aggregator import EvidenceAggregator
from veritas.verification.nli_model import (
    CrossEncoderNLI,
    normalize_elliptical_contrast,
    _SEMANTIC_REFUTATION,
)


class TestTargetedIncognitoRegression(unittest.TestCase):
    """
    Tests A and B: Reproductions of Claim A and Claim B using the actual NLI pipeline
    and fixed passage hierarchy resolution.
    """

    @classmethod
    def setUpClass(cls):
        # Force offline HuggingFace usage to avoid network requests
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        cls.nli = CrossEncoderNLI()
        cls.engine = VerificationEngine(nli_model=cls.nli)

    def test_a_incognito_tracking_not_verified_true(self):
        """
        Test A: Claim A 'Using incognito mode prevents websites from tracking a user's activity.'
        must NOT evaluate to VERIFIED_TRUE when passages explicitly state websites/trackers can observe activity.
        Must evaluate to VERIFIED_FALSE with refuting evidence.
        """
        claim_a = "Using incognito mode prevents websites from tracking a user's activity."

        passages_a = [
            {
                "doc_id": "doc_eset",
                "text": (
                    "### Is Incognito Mode completely private?\n\n"
                    "No, Incognito mode only prevents local storage of temporary data such as browsing history and cookies. "
                    "Your ISP, employer, and websites you visit can still track your activity.\n\n"
                    "### Can websites still track me if I’m using Incognito Mode?\n\n"
                    "Yes, websites can use tracking methods like fingerprinting and IP logging to monitor your activity, even in Incognito mode. "
                    "[...] Incognito mode is a private browsing feature found in most modern web browsers. "
                    "It prevents your browsing history, cookies, and site data from being stored on your device during a session. "
                    "However, it’s important to note that incognito mode does not make you anonymous online. "
                    "While it helps keep your activity private from others using the same device, your internet service provider (ISP), "
                    "websites, and network administrators can still track your online activity."
                ),
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            },
            {
                "doc_id": "doc_kaspersky",
                "text": (
                    "When using incognito mode, at the end of your browser session, your browser will clear cookies from your device. "
                    "This helps to prevent companies from tracking you and may reduce your exposure to targeted ads and dynamic pricing. "
                    "However, not all browsers block third party cookies by default, so even though cookies won’t be stored on your device, "
                    "your activity may still be tracked by [...] While incognito mode increases your online privacy, it doesn’t completely protect you from internet tracking. "
                    "When using incognito mode, third parties can still gather data about you when you visit a website, such as your location, browser, "
                    "operating system, and other information. [...] When using incognito mode, your browsing activity is never entirely invisible, "
                    "whether that’s from your internet service provider or the websites you visit."
                ),
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            },
            {
                "doc_id": "doc_techlusive",
                "text": (
                    "Incognito mode only hides your activity from your own device - not from websites, ISPs, Wi-Fi owners, or trackers. "
                    "Privacy isn't a button."
                ),
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            },
            {
                "doc_id": "doc_firefox",
                "text": (
                    "In Chrome, incognito mode does the same thing. In either case, your actions could be visible to the websites you visit, "
                    "your employer or school, or your internet service provider (ISP). Also, if you sign into any accounts, "
                    "your browsing activity may be saved to that account. [...] It won't keep your internet service provider (ISP) "
                    "from seeing where you've been online. It won't stop websites from seeing your physical location."
                ),
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            },
            {
                "doc_id": "doc_reddit",
                "text": "In Incognito, none of your browsing history, cookies and site data, or information entered in forms are saved on your device. This means your",
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            },
        ]

        verdict = self.engine.verify_claim(claim_text=claim_a, candidate_passages=passages_a)

        # Epistemic invariant: Verdict must NOT be VERIFIED_TRUE
        self.assertNotEqual(verdict.verdict, CanonicalVeracity.VERIFIED_TRUE)
        # Should be correctly classified as VERIFIED_FALSE
        self.assertEqual(verdict.verdict, CanonicalVeracity.VERIFIED_FALSE)

        # Confirm evidence counts
        refuting_items = [
            e for e in verdict.evidence_breakdown
            if e.get("stance") == "refute" and e.get("is_decision_eligible", False) and not e.get("is_joint", False)
        ]
        supporting_items = [
            e for e in verdict.evidence_breakdown
            if e.get("stance") == "support" and e.get("is_decision_eligible", False) and not e.get("is_joint", False)
        ]

        self.assertGreaterEqual(len(refuting_items), 3, "At least 3 passages must be classified as refute")
        self.assertEqual(len(supporting_items), 0, "No passages stating websites can still track should be counted as support")

    def test_b_incognito_ip_visibility_remains_verified_false(self):
        """
        Test B: Claim B 'Private browsing mode automatically hides a user's IP address from the websites they visit.'
        must remain VERIFIED_FALSE when evidence explicitly states the IP address remains visible.
        """
        claim_b = "Private browsing mode automatically hides a user's IP address from the websites they visit."

        passages_b = [
            {
                "doc_id": "doc_itechpost",
                "text": "Routes traffic through another server and can hide the user's usual public IP address from websites. Private browsing: Helps separate a session.",
                "is_relevant": True,
                "is_informative": False,
                "is_claim_aligned": True,
            },
            {
                "doc_id": "doc_kaspersky_ip",
                "text": (
                    "Private browsing doesn't hide your identity or your activity. "
                    "Your IP address is visible to every site you visit, your ISP can see which sites you visit and when. "
                    "[...] No, incognito mode doesn't hide your IP address. Every site you visit, your internet service provider, "
                    "and any network your device is connected to can all see your real IP."
                ),
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            },
            {
                "doc_id": "doc_brave",
                "text": (
                    "Browsing in Incognito (or 'private') mode simply means that your browsing history in that browser window is erased when closed. "
                    "Brave gives two options to hide your IP address while browsing: private windows with Tor, and our built-in Firewall + VPN."
                ),
                "is_relevant": True,
                "is_informative": False,
                "is_claim_aligned": True,
            },
            {
                "doc_id": "doc_mcafee",
                "text": (
                    "The terms “private search” and “incognito mode” sound great. But while your history is erased on your device, "
                    "it’s still visible to the outside world. Even when you’re in incognito mode, websites, your ISP, and your network "
                    "can still see your IP address and browsing history. [...] An incognito browser refers to a private browser mode session "
                    "where your activity, such as sites visited, cookies, and form data, is not saved on your device after you close the window or tabs. "
                    "This mode prevents local storage of your browsing history but does not hide your activity from your internet service provider, "
                    "employer, or the websites you visit.\n\n"
                    "### What does private browser mode hide? [...] "
                    "When you search the internet, your web browser automatically saves the history of your searches. "
                    "In incognito mode, however, it deletes this information when you end the session."
                ),
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            },
            {
                "doc_id": "doc_hp",
                "text": (
                    "It does not hide your activity from your employer or school. "
                    "It does not hide your IP address from websites. Every website you visit can see your IP address, "
                    "which reveals your approximate location and can be used to identify you across sessions."
                ),
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            },
        ]

        verdict = self.engine.verify_claim(claim_text=claim_b, candidate_passages=passages_b)

        self.assertEqual(verdict.verdict, CanonicalVeracity.VERIFIED_FALSE)
        refuting_items = [
            e for e in verdict.evidence_breakdown
            if e.get("stance") == "refute" and e.get("is_decision_eligible", False) and not e.get("is_joint", False)
        ]
        supporting_items = [
            e for e in verdict.evidence_breakdown
            if e.get("stance") == "support" and e.get("is_decision_eligible", False) and not e.get("is_joint", False)
        ]
        self.assertGreaterEqual(len(refuting_items), 3)
        self.assertEqual(len(supporting_items), 0)


class TestStancePolarityAndSemantics(unittest.TestCase):
    """
    Test C: Stance polarity controlled examples:
    - Evidence entails claim
    - Evidence explicitly contradicts claim
    - Evidence is related but neutral
    - Claim with absolute terms ('always', 'never', 'guarantees')
    - Elliptical contrast normalization
    """

    @classmethod
    def setUpClass(cls):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        cls.nli = CrossEncoderNLI()

    def test_c1_evidence_entails_claim(self):
        premise = "The James Webb Space Telescope was successfully launched on December 25, 2021 from French Guiana."
        hypothesis = "The James Webb Space Telescope was launched in December 2021."
        res = self.nli.predict_pair(premise, hypothesis)
        self.assertEqual(res["predicted_label"], "ENTAILMENT")
        self.assertGreater(res["entailment"], 0.85)

    def test_c2_evidence_contradicts_claim(self):
        premise = "Neuroscientists have demonstrated that humans use virtually all of their brain, debunking the myth that 90% is dormant."
        hypothesis = "Humans only use 10% of their brains."
        res = self.nli.predict_pair(premise, hypothesis)
        self.assertEqual(res["predicted_label"], "CONTRADICTION")
        self.assertGreater(res["contradiction"], 0.85)

    def test_c3_evidence_is_neutral(self):
        premise = "Electricity generation from solar panels reached record highs in Western Europe during the summer months."
        hypothesis = "Electricity generation and consumption cause economic growth."
        res = self.nli.predict_pair(premise, hypothesis)
        self.assertEqual(res["predicted_label"], "NEUTRAL")
        self.assertGreater(res["neutral"], 0.50)

    def test_c4_claim_with_absolute_term_contradicted(self):
        premise = "While private browsing offers local privacy, it does not guarantee anonymity or prevent remote tracking."
        hypothesis = "Private browsing guarantees complete anonymity online."
        res = self.nli.predict_pair(premise, hypothesis)
        self.assertEqual(res["predicted_label"], "CONTRADICTION")
        self.assertGreater(res["contradiction"], 0.80)

    def test_c5_elliptical_contrast_normalization(self):
        raw_text = "Incognito mode only hides activity from your device - not from websites or trackers."
        normalized = normalize_elliptical_contrast(raw_text)
        self.assertIn("does not hide activity", normalized)
        self.assertIn("from websites or trackers", normalized)

        # When evaluated through NLI with the normalized clause, contradiction to 'prevents websites from tracking' must dominate
        claim = "Using incognito mode prevents websites from tracking a user's activity."
        res = self.nli.predict_pair(raw_text, claim)
        self.assertEqual(res["predicted_label"], "CONTRADICTION")
        self.assertGreater(res["contradiction"], 0.80)


class TestAggregationConsistencyAndAbstention(unittest.TestCase):
    """
    Tests D & E: Mocked unit tests for aggregation consistency and abstention logic.
    """

    def setUp(self):
        self.aggregator = EvidenceAggregator(
            jaccard_dedup_threshold=0.75,
            neutral_threshold=0.70,
            min_informative_mass=0.20,
            conflict_threshold=0.40,
        )

    def test_d_refuting_evidence_cannot_silently_contribute_to_support(self):
        """
        Test D: Contradictory evidence cannot contribute to support count or entailment mass.
        """
        passages = [
            {"doc_id": "p1", "text": "Evidence refuting claim A.", "is_relevant": True, "is_informative": True, "is_claim_aligned": True, "is_decision_eligible": True},
            {"doc_id": "p2", "text": "Another passage refuting claim A.", "is_relevant": True, "is_informative": True, "is_claim_aligned": True, "is_decision_eligible": True},
        ]
        nli_preds = [
            {"entailment": 0.02, "contradiction": 0.95, "neutral": 0.03, "predicted_label": "CONTRADICTION"},
            {"entailment": 0.05, "contradiction": 0.90, "neutral": 0.05, "predicted_label": "CONTRADICTION"},
        ]

        agg_verdict = self.aggregator.aggregate(
            candidate_passages=passages,
            nli_predictions=nli_preds,
        )

        self.assertEqual(agg_verdict.canonical_verdict, CanonicalVeracity.VERIFIED_FALSE)
        support_count = sum(1 for e in agg_verdict.evidence_breakdown if e.get("stance") == "support")
        refute_count = sum(1 for e in agg_verdict.evidence_breakdown if e.get("stance") == "refute")

        self.assertEqual(support_count, 0)
        self.assertEqual(refute_count, 2)
        self.assertGreater(agg_verdict.contradiction_mass, agg_verdict.entailment_mass)

    def test_e_abstention_on_insufficient_eligible_evidence(self):
        """
        Test E: Insufficient eligible evidence does not produce a confident definitive verdict.
        """
        passages = [
            {"doc_id": "p1", "text": "Completely off-topic passage.", "is_relevant": False, "is_informative": False, "is_claim_aligned": False, "is_decision_eligible": False},
            {"doc_id": "p2", "text": "Another irrelevant passage.", "is_relevant": False, "is_informative": False, "is_claim_aligned": False, "is_decision_eligible": False},
        ]
        nli_preds = [
            {"entailment": 0.05, "contradiction": 0.05, "neutral": 0.90, "predicted_label": "NEUTRAL"},
            {"entailment": 0.08, "contradiction": 0.04, "neutral": 0.88, "predicted_label": "NEUTRAL"},
        ]

        agg_verdict = self.aggregator.aggregate(
            candidate_passages=passages,
            nli_predictions=nli_preds,
        )

        self.assertEqual(agg_verdict.canonical_verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)

    def test_e_engine_abstains_when_zero_eligible_support_for_claimed_true(self):
        """
        Test E: VerificationEngine must abstain to INSUFFICIENT_EVIDENCE if an aggregation
        would otherwise produce VERIFIED_TRUE with 0 eligible supporting passages.
        """
        mock_nli = MagicMock()
        mock_nli.predict_batch.side_effect = lambda pairs: [
            {"predicted_label": "NEUTRAL", "entailment": 0.22, "contradiction": 0.05, "neutral": 0.73}
            for _ in pairs
        ]
        engine = VerificationEngine(nli_model=mock_nli)
        claim = "Some unverified proposition."
        passages = [
            {"doc_id": "p1", "text": "Neutral context 1", "is_relevant": True, "is_informative": True, "is_claim_aligned": True},
            {"doc_id": "p2", "text": "Neutral context 2", "is_relevant": True, "is_informative": True, "is_claim_aligned": True},
        ]

        verdict = engine.verify_claim(claim_text=claim, candidate_passages=passages)
        self.assertNotEqual(verdict.verdict, CanonicalVeracity.VERIFIED_TRUE)
        self.assertEqual(verdict.verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertTrue(verdict.was_abstained)


if __name__ == "__main__":
    unittest.main()
