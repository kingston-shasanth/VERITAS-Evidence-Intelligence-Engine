"""
Regression test suite for evidence classification, snippet hierarchy, topical relevance,
and relevance-vs-stance distinction in VERITAS-X.

Verifies:
1. Multi-snippet web passages with introductory definitions and ellipses [...] are NOT
   incorrectly labeled as CONTRADICTION when the passage as a whole supports the claim.
2. Proper nouns ending in 's' (e.g. 'James') are not misclassified as verbs in claim decomposition.
3. Legitimate debunking passages (e.g. 10% brain myth) retain their decisive CONTRADICTION stance.
4. Unrelated/off-topic passages retrieved in Combined mode have is_relevant=False, score=0.0,
   are never counted as decisive support or contradiction, and do not trigger false disparities.
5. Live web simulation with 4 supporting passages + the actual problematic passage produces
   VERIFIED_TRUE with 0 refuters and no cross-passage disparity.
"""
import unittest
import os
import sys

# Ensure src/ and repo root are on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

from veritas.data.schema import CanonicalVeracity
from veritas.verification.nli_model import CrossEncoderNLI
from veritas.verification.engine import VerificationEngine, _decompose_claim
from veritas.verification.evidence_aggregator import EvidenceAggregator
from api.adapter import VeritasAPIAdapter


class TestEvidenceClassificationRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize the verification engine and adapter once on CPU with local retriever."""
        cls.adapter = VeritasAPIAdapter(device_override="cpu")
        cls.engine = cls.adapter.engine
        cls.nli = cls.engine.nli

    def test_jwst_problematic_snippet_classified_as_entailment(self):
        """
        The actual problematic passage from the Live Web test containing [...]
        must be classified as ENTAILMENT, not CONTRADICTION.
        """
        claim = "NASA's James Webb Space Telescope launched in December 2021"
        problematic_passage = (
            "The James Webb Space Telescope (JWST) is a space telescope [...] "
            "It was launched on 25 December 2021 on an Ariane 5 rocket from Kourou, French Guiana."
        )

        pred = self.nli.predict_pair(problematic_passage, claim)
        self.assertEqual(
            pred["predicted_label"],
            "ENTAILMENT",
            f"Expected ENTAILMENT but got {pred['predicted_label']} with scores {pred}"
        )
        self.assertGreater(pred["entailment"], 0.80)
        self.assertLess(pred["contradiction"], 0.10)

    def test_jwst_all_supporting_variants_are_entailment(self):
        """All typical phrasing variants of the JWST launch must predict ENTAILMENT."""
        claim = "NASA's James Webb Space Telescope launched in December 2021"
        variants = [
            "The James Webb Space Telescope was launched on December 25, 2021 [...] Development began in 1996 for a launch initially planned for 2007.",
            "NASA's James Webb Space Telescope lifted off on an Ariane 5 rocket on Dec. 25, 2021 [...] It launched from Europe's Spaceport in French Guiana.",
            "The James Webb Space Telescope launched on Christmas Day, December 25, 2021.",
            "The James Webb Space Telescope (JWST) is an ESA/NASA/CSA space telescope launched on 25 December 2021 on Ariane flight VA256.",
        ]
        pairs = [(v, claim) for v in variants]
        preds = self.nli.predict_batch(pairs)
        for v, p in zip(variants, preds):
            self.assertEqual(
                p["predicted_label"],
                "ENTAILMENT",
                f"Variant '{v[:60]}...' expected ENTAILMENT, got {p['predicted_label']}"
            )

    def test_debunking_passage_with_ellipses_retains_contradiction(self):
        """
        A legitimate debunking passage with [...] (e.g. brain myth) must still
        predict CONTRADICTION and not be weakened.
        """
        hypothesis = "Humans use only 10% of their brains."
        debunk_passage = (
            "Do humans really use just 10% of their brains? "
            "### Myth: Humans Only Use 10% of Their Brains [...] "
            "The myth that humans only use 10% of their brains is FALSE. "
            "Neuroimaging shows humans use virtually 100% of their brain throughout the day."
        )
        pred = self.nli.predict_pair(debunk_passage, hypothesis)
        self.assertEqual(pred["predicted_label"], "CONTRADICTION")
        self.assertGreater(pred["contradiction"], 0.70)

    def test_proper_nouns_ending_in_s_do_not_distort_claim_decomposition(self):
        """
        Claims with names like 'James' must not treat 'James' as a verb or sever
        the subject at 'James'.
        """
        claim = "NASA's James Webb Space Telescope launched in December 2021"
        sub_subj, sub_pred, date_num, verb_tokens = _decompose_claim(claim)

        # Subject must include 'james', 'webb', 'telescope'
        self.assertTrue(
            {"james", "webb"}.issubset(sub_subj),
            f"Expected 'james' and 'webb' in subject tokens, got: {sub_subj}"
        )
        # Verb tokens must be 'launched', not 'james'
        self.assertIn("launched", verb_tokens)
        self.assertNotIn("james", verb_tokens)

        # Topical relevance check on passage without 'NASA'
        passage_no_nasa = "The James Webb Space Telescope was launched on December 25, 2021."
        self.assertTrue(
            VerificationEngine.is_topically_relevant(claim, passage_no_nasa),
            "Passage with James Webb Space Telescope must be topically relevant to the claim"
        )
        self.assertTrue(
            VerificationEngine.has_claim_relation_alignment(claim, passage_no_nasa),
            "Passage must be aligned with the launch relation"
        )

    def test_unrelated_passage_topical_filtering(self):
        """Unrelated local passages must be tagged as not topically relevant."""
        claim = "NASA's James Webb Space Telescope launched in December 2021"
        unrelated_1 = "The municipal water authority increased wastewater treatment fees in December 2021 following local council votes."
        unrelated_2 = "Commercial office space leasing rates in downtown Chicago fell during the fourth quarter of 2021."

        self.assertFalse(VerificationEngine.is_topically_relevant(claim, unrelated_1))
        self.assertFalse(VerificationEngine.is_topically_relevant(claim, unrelated_2))

    def test_live_web_simulation_produces_unified_support(self):
        """
        Simulate the actual 5 Live Web test passages (4 supporting + 1 problematic).
        Verifies:
        - Verdict is VERIFIED_TRUE.
        - Exactly 0 refuting passages.
        - Disparity is False.
        """
        claim = "NASA's James Webb Space Telescope launched in December 2021"
        passages = [
            {
                "doc_id": "web_01",
                "text": "NASA's James Webb Space Telescope launched at 7:20 a.m. EST Saturday, Dec. 25, 2021, on an Ariane 5 rocket from Europe's Spaceport in French Guiana.",
                "source_type": "web",
                "source_title": "NASA Release",
                "source_url": "https://www.nasa.gov/press-release/james-webb-space-telescope-launch",
            },
            {
                "doc_id": "web_02",
                "text": "Targeted for launch on December 25, 2021, NASA's James Webb Space Telescope is the premier space science observatory of the next decade.",
                "source_type": "web",
                "source_title": "NASA Science",
                "source_url": "https://science.nasa.gov/mission/webb/",
            },
            {
                "doc_id": "web_03",
                "text": "Ariane flight VA256 carried the James Webb Space Telescope into space on December 25, 2021, successfully deploying the observatory into its planned transfer trajectory.",
                "source_type": "web",
                "source_title": "Arianespace Mission",
                "source_url": "https://www.arianespace.com/mission/ariane-flight-va256/",
            },
            {
                "doc_id": "web_04",
                "text": "NASA's James Webb Space Telescope launched on December 25, 2021.",
                "source_type": "web",
                "source_title": "Spaceflight Now",
                "source_url": "https://spaceflightnow.com/2021/12/25/jwst-launch/",
            },
            {
                "doc_id": "web_05",
                "text": "The James Webb Space Telescope (JWST) is a space telescope [...] It was launched on 25 December 2021 on an Ariane 5 rocket from Kourou, French Guiana.",
                "source_type": "web",
                "source_title": "Wikipedia - JWST",
                "source_url": "https://en.wikipedia.org/wiki/James_Webb_Space_Telescope",
            },
        ]

        report = self.adapter.verify(
            claim=claim,
            mode="web",
            candidate_passages=passages,
        )

        self.assertEqual(report["verdict"], CanonicalVeracity.VERIFIED_TRUE.value)
        self.assertFalse(report["has_cross_passage_disparity"])
        self.assertFalse(report["abstained"])

        # Check evidence breakdown
        evidence = report["evidence"]
        self.assertEqual(len(evidence), 5)
        for ev in evidence:
            self.assertEqual(
                ev["stance"], "support",
                f"Passage {ev['id']} ({ev['source_title']}) was unexpectedly '{ev['stance']}' instead of 'support'"
            )

        # Confirm reason explains consistent support
        self.assertIn("supports the claim", report["reason"])
        self.assertNotIn("disagreement", report["reason"].lower())
        self.assertNotIn("disparity", report["reason"].lower())

    def test_combined_mode_with_unrelated_local_passage(self):
        """
        Simulate Combined mode with 4 supporting web passages and 1 unrelated local passage.
        Verifies:
        - Verdict is VERIFIED_TRUE.
        - The unrelated passage is assigned score 0.0 and stance neutral.
        - The unrelated passage is sorted to the bottom.
        - It is not counted in supporting or refuting tallies.
        """
        claim = "NASA's James Webb Space Telescope launched in December 2021"
        passages = [
            {
                "doc_id": "web_01",
                "text": "NASA's James Webb Space Telescope launched on December 25, 2021.",
                "source_type": "web",
                "source_title": "NASA Launch News",
            },
            {
                "doc_id": "web_02",
                "text": "The James Webb Space Telescope was launched on 25 December 2021 on an Ariane 5 rocket.",
                "source_type": "web",
                "source_title": "ESA Release",
            },
            {
                "doc_id": "web_03",
                "text": "The James Webb Space Telescope (JWST) is a space telescope [...] It was launched on 25 December 2021.",
                "source_type": "web",
                "source_title": "Wikipedia Summary",
            },
            {
                "doc_id": "web_04",
                "text": "Targeted for launch on December 25, 2021, NASA's James Webb Space Telescope lifted off successfully.",
                "source_type": "web",
                "source_title": "Spaceflight Digest",
            },
            {
                "doc_id": "local_01",
                "text": "The municipal water authority increased residential tax rates in December 2021 following local council votes.",
                "source_type": "local",
                "source_title": "Municipal Tax Records",
            },
        ]

        report = self.adapter.verify(
            claim=claim,
            mode="hybrid",
            candidate_passages=passages,
        )

        self.assertEqual(report["verdict"], CanonicalVeracity.VERIFIED_TRUE.value)
        self.assertFalse(report["has_cross_passage_disparity"])

        evidence = report["evidence"]
        self.assertEqual(len(evidence), 5)

        # Unrelated local passage check
        local_ev = next(e for e in evidence if e["doc_id"] == "local_01")
        self.assertFalse(local_ev["is_relevant"])
        self.assertFalse(local_ev["contributed_to_decision"])
        self.assertEqual(local_ev["score"], 0.0)
        self.assertEqual(local_ev["stance"], "neutral")

        # 4 web passages must be supporting
        web_evs = [e for e in evidence if e["doc_id"] != "local_01"]
        for we in web_evs:
            self.assertEqual(we["stance"], "support")
            self.assertTrue(we["is_relevant"])
            self.assertGreater(we["score"], 0.80)

    def test_nasa_science_timeline_explicit_launch_produces_support(self):
        """
        Regression test for Live Web mode where NASA Science mission timeline passage
        explicitly mentions launch on Dec 25, 2021 alongside background mission text.
        Must produce 'support' (never 'refute' / CONTRADICTS CLAIM), 0 refuting passages,
        and verdict VERIFIED_TRUE.
        """
        claim = "NASA's James Webb Space Telescope launched in December 2021"
        passages = [
            {
                "doc_id": "web_nasa_science_01",
                "text": (
                    "NASA Science - James Webb Space Telescope Mission Timeline: "
                    "Webb studies every phase in the history of our Universe, ranging from the first luminous "
                    "glows after the Big Bang, to the formation of solar systems capable of supporting life on Earth. "
                    "Webb launched on Dec. 25th 2021. It does not orbit around the Earth like the Hubble Space Telescope, "
                    "it orbits the Sun 1.5 million kilometers away."
                ),
                "source_type": "web",
                "source_title": "NASA Science",
                "source_url": "https://science.nasa.gov/mission/webb/",
            },
            {
                "doc_id": "web_nasa_release_02",
                "text": "NASA's James Webb Space Telescope launched at 7:20 a.m. EST Saturday, Dec. 25, 2021, on an Ariane 5 rocket.",
                "source_type": "web",
                "source_title": "NASA Release",
                "source_url": "https://www.nasa.gov/press-release/james-webb-space-telescope-launch",
            },
            {
                "doc_id": "web_esa_03",
                "text": "The James Webb Space Telescope (JWST) is an ESA/NASA/CSA space telescope launched on 25 December 2021 on Ariane flight VA256.",
                "source_type": "web",
                "source_title": "ESA Spacecraft Operations",
                "source_url": "https://www.esa.int/Science_Exploration/Space_Science/Webb",
            },
            {
                "doc_id": "web_spaceflight_04",
                "text": "The James Webb Space Telescope launched on Christmas Day, December 25, 2021.",
                "source_type": "web",
                "source_title": "Spaceflight Now",
                "source_url": "https://spaceflightnow.com/2021/12/25/jwst-launch/",
            },
        ]

        report = self.adapter.verify(
            claim=claim,
            mode="web",
            candidate_passages=passages,
        )

        self.assertEqual(report["verdict"], CanonicalVeracity.VERIFIED_TRUE.value)
        self.assertFalse(report["has_cross_passage_disparity"])
        self.assertFalse(report["abstained"])

        # Crucially: NASA Science passage MUST be support, NOT refute
        nasa_ev = next(e for e in report["evidence"] if e["doc_id"] == "web_nasa_science_01")
        self.assertEqual(nasa_ev["stance"], "support", f"NASA Science was unexpectedly '{nasa_ev['stance']}' instead of 'support'")
        self.assertTrue(nasa_ev["is_relevant"])

        # Exactly 0 refuting passages across the entire report
        refuting_count = len([e for e in report["evidence"] if e["stance"] == "refute"])
        self.assertEqual(refuting_count, 0, f"Found unexpected refuting passages: {[e for e in report['evidence'] if e['stance'] == 'refute']}")

    def test_local_mode_off_topic_passages_tagged_is_relevant_false(self):
        """
        Regression test for Local mode: When verifying JWST launch in local mode (not in local corpus),
        all 5 retrieved passages must have is_relevant == False, score == 0.0,
        stance == 'neutral', and verdict == INSUFFICIENT_EVIDENCE.
        """
        claim = "NASA's James Webb Space Telescope launched in December 2021"
        report = self.adapter.verify(claim=claim, mode="local")

        self.assertEqual(report["verdict"], CanonicalVeracity.INSUFFICIENT_EVIDENCE.value)
        evidence = report["evidence"]
        self.assertGreater(len(evidence), 0)

        for ev in evidence:
            self.assertFalse(ev["is_relevant"], f"Passage {ev['id']} should be marked is_relevant=False")
            self.assertEqual(ev["score"], 0.0, f"Passage {ev['id']} score should be 0.0")
            self.assertEqual(ev["stance"], "neutral")
            self.assertFalse(ev["contributed_to_decision"])


if __name__ == "__main__":
    unittest.main()

