"""
Comprehensive regression test suite for general-purpose evidence reasoning,
classification, topical relevance, informativeness, and consensus aggregation in VERITAS-X.

Tests 16 distinct semantic and structural categories across unfamiliar claims,
domains, sentence structures, and source types without relying on claim-specific rules.
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


class TestGeneralEvidenceReasoning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize verification engine once on CPU for fast deterministic testing."""
        cls.adapter = VeritasAPIAdapter(device_override="cpu")
        cls.engine = cls.adapter.engine
        cls.nli = cls.engine.nli

    # -------------------------------------------------------------------------
    # Category 1: Directly Supported and Directly Contradicted Claims
    # -------------------------------------------------------------------------
    def test_direct_entailment_astronomy_claim(self):
        """Passage providing direct factual affirmation predicts ENTAILMENT."""
        claim = "The Earth orbits the Sun."
        passage = "Heliocentric astronomical observations confirm that the Earth orbits the Sun once every 365.25 days."
        pred = self.nli.predict_pair(passage, claim)
        self.assertEqual(pred["predicted_label"], "ENTAILMENT")
        self.assertGreater(pred["entailment"], 0.80)
        self.assertLess(pred["contradiction"], 0.10)

    def test_direct_contradiction_anatomy_claim(self):
        """Passage asserting mutually exclusive biological fact predicts CONTRADICTION."""
        claim = "Adult humans normally possess three biological lungs."
        passage = "Human anatomy consists of two lungs in the thoracic cavity: a right lung with three lobes and a left lung with two lobes."
        pred = self.nli.predict_pair(passage, claim)
        self.assertEqual(pred["predicted_label"], "CONTRADICTION")
        self.assertGreater(pred["contradiction"], 0.75)

    # -------------------------------------------------------------------------
    # Category 2: Negation and Double Negation
    # -------------------------------------------------------------------------
    def test_negation_in_claim_entailed_by_negative_fact(self):
        """A negative claim ('do not cause') is entailed by a passage affirming the negative."""
        claim = "Vaccines do not cause autism."
        passage = "Extensive peer-reviewed epidemiological research across millions of children demonstrates that vaccines do not cause autism."
        pred = self.nli.predict_pair(passage, claim)
        self.assertEqual(pred["predicted_label"], "ENTAILMENT")
        self.assertGreater(pred["entailment"], 0.80)

    def test_double_negation_entailed_by_affirmative_event(self):
        """A double negative claim ('did not fail to land') is entailed by affirmative success."""
        claim = "The Apollo 11 lunar module did not fail to land on the Moon."
        passage = "On July 20, 1969, the Apollo 11 Lunar Module Eagle successfully landed on the Moon, allowing Neil Armstrong and Buzz Aldrin to walk on the lunar surface."
        pred = self.nli.predict_pair(passage, claim)
        self.assertEqual(pred["predicted_label"], "ENTAILMENT")
        self.assertGreater(pred["entailment"], 0.70)

    # -------------------------------------------------------------------------
    # Category 3: Universal and Absolute Claims ("never", "always", "all")
    # -------------------------------------------------------------------------
    def test_absolute_never_claim_refuted_by_counterexample(self):
        """
        Claim asserting 'never' (e.g. lightning never strikes twice) is refuted by
        concrete counterexamples without requiring explicit meta-words like 'myth'.
        """
        claim = "Lightning never strikes the same place twice."
        passage = (
            "The Empire State Building in New York City is struck by lightning an average of 25 times per year, "
            "and has been struck up to eight times during a single thunderstorm."
        )
        pred = self.nli.predict_pair(passage, claim)
        self.assertEqual(pred["predicted_label"], "CONTRADICTION")
        self.assertGreater(pred["contradiction"], 0.75)

    def test_universal_all_claim_refuted_by_exception(self):
        """Claim asserting 'all mammals lay eggs' or 'all mammals give birth' refuted by monotremes."""
        claim = "All mammals give birth to live young rather than laying eggs."
        passage = "Monotremes, including the platypus and the echidna, are mammals that lay eggs rather than giving birth to live young."
        pred = self.nli.predict_pair(passage, claim)
        self.assertEqual(pred["predicted_label"], "CONTRADICTION")
        self.assertGreater(pred["contradiction"], 0.75)

    # -------------------------------------------------------------------------
    # Category 4: Dates, Quantities, Comparisons, and Temporal Relations
    # -------------------------------------------------------------------------
    def test_comparative_elevation_claim_entailed(self):
        """Comparative claim ('is taller than') entailed by comparative measurements."""
        claim = "Mount Everest is taller than K2."
        passage = "Mount Everest stands at 8,848.86 meters above sea level, while K2, the world's second-highest mountain, stands at 8,611 meters."
        pred = self.nli.predict_pair(passage, claim)
        self.assertEqual(pred["predicted_label"], "ENTAILMENT")
        self.assertGreater(pred["entailment"], 0.75)

    def test_historical_adoption_date_entailed(self):
        """Historical adoption date claim verified by exact date passage."""
        claim = "The United States Declaration of Independence was approved in July 1776."
        passage = "The Second Continental Congress formally adopted the Declaration of Independence in Philadelphia on July 4, 1776."
        pred = self.nli.predict_pair(passage, claim)
        self.assertEqual(pred["predicted_label"], "ENTAILMENT")
        self.assertGreater(pred["entailment"], 0.80)

    # -------------------------------------------------------------------------
    # Category 5: Multiword Subjects and Proper Nouns
    # -------------------------------------------------------------------------
    def test_multiword_organization_subject_decomposition(self):
        """Proper nouns like 'World Health Organization' decompose cleanly into subject entity and predicate."""
        claim = "The World Health Organization declared the end of the emergency in May 2023."
        sub_subj, sub_pred, date_num, verb_tokens = _decompose_claim(claim)
        self.assertTrue({"world", "health", "organization"}.issubset(sub_subj))
        self.assertIn("declared", verb_tokens)
        self.assertTrue({"2023", "may"}.issubset(date_num))

    # -------------------------------------------------------------------------
    # Category 6: Direct Quotations and Attributed Statements
    # -------------------------------------------------------------------------
    def test_attributed_quote_refutation_hierarchy(self):
        """Passage quoting a flat Earth claim but refuting it in substantive body predicts CONTRADICTION."""
        claim = "The Earth is flat."
        passage = (
            "Modern flat earth advocates assert: 'The Earth is a flat plane encircled by ice.' [...] "
            "However, orbital satellite photography, geodesy, and global circumnavigation conclusively prove "
            "that the Earth is a spherical body."
        )
        pred = self.nli.predict_pair(passage, claim)
        self.assertEqual(pred["predicted_label"], "CONTRADICTION")
        self.assertGreater(pred["contradiction"], 0.75)

    # -------------------------------------------------------------------------
    # Category 7: Figurative Language and Dictionary Definitions
    # -------------------------------------------------------------------------
    def test_lexical_figurative_definition_supported(self):
        """Claim about figurative dictionary usage supported by lexicographical explanation."""
        claim = "Modern English dictionaries recognize that 'literally' can be used figuratively for emphasis."
        passage = (
            "Major lexicographical authorities including Merriam-Webster and the Oxford English Dictionary "
            "include a definition of 'literally' denoting virtual or metaphorical emphasis rather than strict literal truth."
        )
        pred = self.nli.predict_pair(passage, claim)
        self.assertEqual(pred["predicted_label"], "ENTAILMENT")
        self.assertGreater(pred["entailment"], 0.75)

    def test_dictionary_pronunciation_header_is_uninformative(self):
        """Pure pronunciation / lexical headers fail informativeness."""
        header = "lightning /ˈlaɪt.nɪŋ/ noun"
        is_inf, reason = VerificationEngine.is_passage_informative(header)
        self.assertFalse(is_inf)
        self.assertTrue("insufficient" in reason or "lexical" in reason)

    # -------------------------------------------------------------------------
    # Category 8: Relevant Neutral Background Information
    # -------------------------------------------------------------------------
    def test_relevant_neutral_background_not_decisive(self):
        """
        Passage about Albert Einstein that does NOT mention the Nobel Prize or photoelectric effect
        is topically relevant to Einstein, but unaligned with the specific assertion.
        """
        claim = "Albert Einstein received the 1921 Nobel Prize in Physics for his work on the photoelectric effect."
        neutral_passage = (
            "Albert Einstein was born in Ulm, Kingdom of Württemberg, in 1879. "
            "In 1905, he published four groundbreaking papers in the Annalen der Physik including special relativity."
        )
        is_rel = VerificationEngine.is_topically_relevant(claim, neutral_passage)
        self.assertTrue(is_rel, "Should be topically relevant to Albert Einstein")
        is_aligned = VerificationEngine.has_claim_relation_alignment(claim, neutral_passage)
        self.assertFalse(is_aligned, "Must not be aligned with the 1921 Nobel Prize assertion")

        pred = self.nli.predict_pair(neutral_passage, claim)
        self.assertGreater(pred["neutral"], 0.60)

    # -------------------------------------------------------------------------
    # Category 9: Topically Unrelated Passages with Keyword Overlap
    # -------------------------------------------------------------------------
    def test_misleading_keyword_overlap_rejected_as_irrelevant(self):
        """Passage sharing words like 'space', 'launched', '2021' but about wastewater is marked irrelevant."""
        claim = "NASA's James Webb Space Telescope launched in December 2021."
        coincidental_passage = (
            "The regional water utility launched a wastewater expansion initiative in December 2021, "
            "allocating municipal office space and reservoir funding across three suburban districts."
        )
        is_rel = VerificationEngine.is_topically_relevant(claim, coincidental_passage)
        self.assertFalse(is_rel, "Passage lacking James Webb Space Telescope subject must be irrelevant")

    # -------------------------------------------------------------------------
    # Category 10: Truncated Sentences, Ellipses, and Anaphoric Pronouns
    # -------------------------------------------------------------------------
    def test_truncated_fragment_rejected_by_informativeness(self):
        """A truncated sentence ending in 'because...' is uninformative."""
        fragment = "The spacecraft was launched into high orbit because..."
        is_inf, reason = VerificationEngine.is_passage_informative(fragment)
        self.assertFalse(is_inf)
        self.assertEqual(reason, "truncated_dangling_fragment")

    def test_context_preservation_for_anaphoric_snippets(self):
        """
        When passage has ellipses separating the definition and launch fact,
        referential context is preserved and predicts ENTAILMENT.
        """
        claim = "The Hubble Space Telescope was placed into orbit in April 1990."
        passage = (
            "The Hubble Space Telescope is a space observatory. [...] "
            "It was carried into orbit by Space Shuttle Discovery in April 1990."
        )
        pred = self.nli.predict_pair(passage, claim)
        self.assertEqual(pred["predicted_label"], "ENTAILMENT")
        self.assertGreater(pred["entailment"], 0.75)

    # -------------------------------------------------------------------------
    # Category 11: Multi-Passage Consensus vs Disparity
    # -------------------------------------------------------------------------
    def test_multiple_independent_supporting_passages_synthesize_verified_true(self):
        """3 independent supporting passages produce VERIFIED_TRUE with no disparity."""
        claim = "The speed of light in a vacuum is approximately 300,000 kilometers per second."
        passages = [
            {
                "doc_id": "c1",
                "text": "The speed of light in vacuum, commonly denoted c, is a universal physical constant exactly equal to 299,792,458 m/s, or roughly 300,000 km/s.",
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            },
            {
                "doc_id": "c2",
                "text": "In physics, light travels through empty space at approximately 300,000 kilometers each second.",
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            },
            {
                "doc_id": "c3",
                "text": "Measurements confirm light moves in a vacuum at close to 300,000 km per second.",
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            },
        ]
        verdict = self.engine.verify_claim(claim_text=claim, candidate_passages=passages)
        self.assertEqual(verdict.verdict, CanonicalVeracity.VERIFIED_TRUE)
        self.assertFalse(verdict.was_abstained)
        self.assertGreater(verdict.confidence, 0.70)

    def test_genuine_cross_passage_disparity_detected(self):
        """When one passage confirms a claim and another contradicts it, disparity is triggered."""
        claim = "Clinical trials demonstrated that Drug Alpha reduced viral shedding duration."
        passages = [
            {
                "doc_id": "p_sup",
                "text": "A Phase II double-blind trial demonstrated that Drug Alpha significantly reduced viral shedding duration by 3.5 days compared to placebo.",
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            },
            {
                "doc_id": "p_ref",
                "text": "A large-scale multicenter Phase III evaluation found Drug Alpha failed to reduce viral shedding duration, observing no difference compared to placebo controls.",
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            },
        ]
        verdict = self.engine.verify_claim(claim_text=claim, candidate_passages=passages)
        self.assertEqual(verdict.verdict, CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED)
        self.assertGreater(verdict.confidence, 0.50)

    # -------------------------------------------------------------------------
    # Category 12: Mixed Evidence (Supporting, Neutral Background, and Irrelevant)
    # -------------------------------------------------------------------------
    def test_mixed_evidence_filters_irrelevant_and_retains_support(self):
        """Mixed evidence retains supporting stance without letting irrelevant or neutral distort counts."""
        claim = "Alexander Fleming discovered penicillin in 1928."
        passages = [
            {
                "doc_id": "m_sup",
                "text": "Scottish physician Alexander Fleming discovered penicillin at St Mary's Hospital in London in September 1928.",
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            },
            {
                "doc_id": "m_neut",
                "text": "St Mary's Hospital is an NHS general hospital located in Paddington, London, founded in 1845.",
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": False,
            },
            {
                "doc_id": "m_irrel",
                "text": "The price of commercial agricultural fertilizer declined during the late autumn harvest season.",
                "is_relevant": False,
                "is_informative": True,
                "is_claim_aligned": False,
            },
        ]
        res = self.adapter.verify(claim=claim, user_evidence=passages)
        self.assertEqual(res["verdict"], "VERIFIED_TRUE")
        ev_list = res["evidence"]
        self.assertEqual(len(ev_list), 3)

        # Check evidence item statuses
        sup_item = next(e for e in ev_list if e["doc_id"] == "m_sup")
        neut_item = next(e for e in ev_list if e["doc_id"] == "m_neut")
        irrel_item = next(e for e in ev_list if e["doc_id"] == "m_irrel")

        self.assertEqual(sup_item["stance"], "support")
        self.assertTrue(sup_item["is_decision_eligible"])

        self.assertEqual(neut_item["stance"], "neutral")
        self.assertFalse(neut_item["is_decision_eligible"])

        self.assertEqual(irrel_item["stance"], "neutral")
        self.assertEqual(irrel_item["score"], 0.0)
        self.assertFalse(irrel_item["is_relevant"])

    # -------------------------------------------------------------------------
    # Category 13: Empty Retrieval and Insufficient Evidence
    # -------------------------------------------------------------------------
    def test_empty_retrieval_yields_insufficient_evidence(self):
        """Zero retrieved evidence passages immediately yields INSUFFICIENT_EVIDENCE with 1.0 confidence."""
        verdict = self.engine.verify_claim(claim_text="Unverifiable obscure claim with no facts.", candidate_passages=[])
        self.assertEqual(verdict.verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertTrue(verdict.was_abstained)
        self.assertEqual(verdict.evidence_used_count, 0)

    # -------------------------------------------------------------------------
    # Category 14: Supplied-Evidence Workflow
    # -------------------------------------------------------------------------
    def test_user_supplied_evidence_workflow(self):
        """User-supplied evidence array executes end-to-end through the adapter."""
        claim = "Titan is the largest moon of Saturn."
        evidence = [
            "Titan is the largest moon of Saturn and the second-largest natural satellite in the Solar System.",
        ]
        res = self.adapter.verify(claim=claim, user_evidence=evidence)
        self.assertEqual(res["verdict"], "VERIFIED_TRUE")
        self.assertEqual(res["mode"], "user_supplied")
        self.assertEqual(len(res["evidence"]), 1)
        self.assertEqual(res["evidence"][0]["stance"], "support")

    # -------------------------------------------------------------------------
    # Category 15: Serialization and Contract Integrity
    # -------------------------------------------------------------------------
    def test_response_serialization_contains_required_fields(self):
        """Response payload must contain all contract fields."""
        claim = "Water expands when it freezes into ice."
        evidence = ["Unlike most substances, water expands upon freezing, resulting in ice being less dense than liquid water."]
        res = self.adapter.verify(claim=claim, user_evidence=evidence)
        for key in ["claim_id", "claim", "verdict", "confidence", "probabilities", "has_cross_passage_disparity", "abstained", "reason", "evidence"]:
            self.assertIn(key, res, f"Missing required payload key: {key}")

        ev = res["evidence"][0]
        for ev_key in ["id", "source", "source_type", "stance", "score", "is_informative", "is_relevant", "is_claim_aligned", "is_decision_eligible", "contributed_to_decision"]:
            self.assertIn(ev_key, ev, f"Missing required evidence key: {ev_key}")

    # -------------------------------------------------------------------------
    # Category 16: General Failure Case Fixtures (A through E)
    # -------------------------------------------------------------------------
    def test_case_a_quantitative_contradiction_included_and_refutes(self):
        """
        Case A: A relevant quantitative contradiction must NOT be excluded as off-topic.
        Claim: An adult human normally has three lungs.
        Passage establishing two lungs is topically relevant, relation-aligned, and refutes the claim.
        A passage about drinking three liters of water is NOT relation-aligned.
        """
        claim = "An adult human normally has three lungs."
        passage_con = "An adult human normally has two lungs located in the thoracic cavity."
        passage_off = "Adult humans require three liters of drinking water daily for kidney function."

        # Verify relevance and alignment
        self.assertTrue(VerificationEngine.is_topically_relevant(claim, passage_con))
        self.assertTrue(VerificationEngine.has_claim_relation_alignment(claim, passage_con))
        self.assertTrue(VerificationEngine.is_decision_eligible(claim, passage_con))

        # Negative control: matching the number 'three' alone does NOT establish alignment
        self.assertFalse(VerificationEngine.has_claim_relation_alignment(claim, passage_off))
        self.assertFalse(VerificationEngine.is_decision_eligible(claim, passage_off))

        # End-to-end verification via API adapter
        res = self.adapter.verify(claim=claim, user_evidence=[passage_con, passage_off])
        self.assertEqual(res["verdict"], "VERIFIED_FALSE")
        ev_con = next(e for e in res["evidence"] if "two lungs" in e["text"])
        self.assertEqual(ev_con["stance"], "refute")
        self.assertTrue(ev_con["is_decision_eligible"])

    def test_case_b_figurative_idiom_dictionary_entry_not_decisive_support(self):
        """
        Case B: Contextual dictionary entry explaining an idiom must NOT count as decisive support.
        Claim: Lightning never strikes the same place twice.
        Idiom definition is marked neutral context and ineligible for decisive support.
        Scientific empirical observation is marked refuting and decisive.
        """
        claim = "Lightning never strikes the same place twice."
        passage_idiom = "Lightning never strikes twice: idiom. Used to say that the same unusual misfortune is not likely to occur again to the same person."
        passage_fact = "Cloud-to-ground lightning flashes frequently strike the same location repeatedly; the Empire State Building is struck an average of 25 times per year."

        self.assertTrue(VerificationEngine.is_figurative_or_lexical_definition(passage_idiom))
        self.assertFalse(VerificationEngine.is_figurative_or_lexical_definition(passage_fact))

        # Idiom entry is NOT decision-eligible for literal truth
        self.assertFalse(VerificationEngine.is_decision_eligible(claim, passage_idiom))
        self.assertTrue(VerificationEngine.is_decision_eligible(claim, passage_fact))

        res = self.adapter.verify(claim=claim, user_evidence=[passage_idiom, passage_fact])
        self.assertEqual(res["verdict"], "VERIFIED_FALSE")

        ev_idiom = next(e for e in res["evidence"] if "idiom" in e["text"])
        ev_fact = next(e for e in res["evidence"] if "Empire State Building" in e["text"])

        self.assertEqual(ev_idiom["stance"], "neutral")
        self.assertFalse(ev_idiom["is_decision_eligible"])
        self.assertFalse(ev_idiom["contributed_to_decision"])

        self.assertEqual(ev_fact["stance"], "refute")
        self.assertTrue(ev_fact["is_decision_eligible"])

    def test_case_c_compound_claim_distinguishes_partial_and_full_coverage(self):
        """
        Case C: Incomplete support for a compound claim must not be labelled decisive full proof.
        Claim: The first modern Olympic Games were held in Athens in 1896.
        A passage covering only the venue is partial; a passage covering all components is full.
        """
        claim = "The first modern Olympic Games were held in Athens in 1896."
        comps = VerificationEngine.decompose_claim_components(claim)
        self.assertGreaterEqual(len(comps), 2, "Must decompose into at least 2 components")

        part_passage = "Athens hosted the first modern Olympic Games."
        cov_level, cov_labels = VerificationEngine.assess_claim_coverage(comps, part_passage)
        self.assertEqual(cov_level, "partial")
        self.assertIn("Venue / Location", cov_labels)
        self.assertNotIn("Time / Date", cov_labels)

        full_passage = "The first modern Olympic Games were held in Athens, Greece, in 1896."
        cov_full, cov_full_labels = VerificationEngine.assess_claim_coverage(comps, full_passage)
        self.assertEqual(cov_full, "full")

        res = self.adapter.verify(claim=claim, user_evidence=[part_passage, full_passage])
        ev_part = next(e for e in res["evidence"] if e["text"] == part_passage)
        ev_full = next(e for e in res["evidence"] if e["text"] == full_passage)

        self.assertEqual(ev_part["claim_coverage"], "partial")
        self.assertFalse(ev_part["contributed_to_decision"], "Partial passage cannot be standalone decisive")

        self.assertEqual(ev_full["claim_coverage"], "full")
        self.assertTrue(ev_full["contributed_to_decision"], "Full passage can be decisive")

    def test_case_d_fragment_does_not_override_complete_passage_hierarchy(self):
        """
        Case D: Referential grounding in hierarchy prevents isolated fragments from overriding passage entailment.
        Claim: NASA's James Webb Space Telescope launched in December 2021.
        """
        claim = "NASA's James Webb Space Telescope launched in December 2021."
        passage = (
            "Targeted for launch on Dec. 25, 2021, the James Webb Space Telescope is NASA's premier space science observatory. "
            "Webb was launched on an Ariane 5 rocket from Europe's Spaceport in French Guiana."
        )
        res = self.adapter.verify(claim=claim, user_evidence=[passage])
        self.assertEqual(res["verdict"], "VERIFIED_TRUE")
        self.assertEqual(res["evidence"][0]["stance"], "support")

    def test_case_e_unrelated_evidence_excluded_from_decisive_counts(self):
        """
        Case E: Unrelated/off-topic passages receive score 0.0, stance neutral, and are excluded from decisive counts.
        """
        claim = "NASA's James Webb Space Telescope launched in December 2021."
        passages = [
            {
                "doc_id": "webb_real",
                "text": "The James Webb Space Telescope launched on December 25, 2021.",
                "source_type": "web",
            },
            {
                "doc_id": "local_irrel",
                "text": "The municipal water authority operates a gravity-fed water purification facility serving 40,000 residents.",
                "source_type": "local",
            },
        ]
        res = self.adapter.verify(claim=claim, user_evidence=passages, mode="hybrid")
        self.assertEqual(res["verdict"], "VERIFIED_TRUE")

        ev_real = next(e for e in res["evidence"] if e["doc_id"] == "webb_real")
        ev_irrel = next(e for e in res["evidence"] if e["doc_id"] == "local_irrel")

        self.assertEqual(ev_real["stance"], "support")
        self.assertTrue(ev_real["is_decision_eligible"])

        self.assertFalse(ev_irrel["is_relevant"])
        self.assertEqual(ev_irrel["score"], 0.0)
        self.assertEqual(ev_irrel["stance"], "neutral")
        self.assertFalse(ev_irrel["is_decision_eligible"])
        self.assertFalse(ev_irrel["contributed_to_decision"])

    # -------------------------------------------------------------------------
    # Category 17: Multi-Hop Reasoning with Interlocking Evidence
    # -------------------------------------------------------------------------
    def test_multihop_reasoning_combines_interlocking_evidence(self):
        """Separate passages interlocking on bridge entities jointly substantiate a proposition."""
        claim = "Dr. Alvarez oversaw the clinical trial."
        passages = [
            "Dr. Alvarez was elected director of the institute in September 2021.",
            "The director of the institute authorized and oversaw the clinical trial.",
        ]
        res = self.adapter.verify(claim=claim, user_evidence=passages)
        self.assertEqual(res["verdict"], "VERIFIED_TRUE")
        self.assertGreater(res["confidence"], 0.60)

    # -------------------------------------------------------------------------
    # Category 18: Near-Duplicate / Copied Evidence Discounting
    # -------------------------------------------------------------------------
    def test_duplicate_and_copied_evidence_does_not_inflate_consensus(self):
        """Multiple near-duplicate copies of the same wire excerpt do not artificially inflate decisive corroboration."""
        claim = "Gold has the chemical symbol Au and atomic number 79."
        base_text = "Gold is a chemical element with the symbol Au and atomic number 79, making it one of the higher atomic number elements that occur naturally."
        dup_text_1 = "Gold is a chemical element with the symbol Au and atomic number 79, making it one of the higher atomic number elements occurring naturally."
        dup_text_2 = "Gold is a chemical element with the symbol Au and atomic number 79, making it one of the higher atomic number elements that occurs naturally."

        res = self.adapter.verify(claim=claim, user_evidence=[base_text, dup_text_1, dup_text_2])
        self.assertEqual(res["verdict"], "VERIFIED_TRUE")
        counts = res.get("evidence_counts", {})
        self.assertGreaterEqual(counts.get("duplicates", 0), 1, "Near-duplicates must be detected and discounted")
        # Duplicates must not be counted as separate decisive independent contributors
        decisive_count = len([e for e in res["evidence"] if e.get("contributed_to_decision")])
        self.assertEqual(decisive_count, 1, "Only 1 unique passage should be badged decisive")

    # -------------------------------------------------------------------------
    # Category 19: Empty Retrieval and Honest Abstention
    # -------------------------------------------------------------------------
    def test_empty_retrieval_and_abstention(self):
        """When zero evidence is available, the engine produces honest abstention."""
        claim = "Zebras evolved on the surface of Mars."
        res = self.adapter.verify(claim=claim, user_evidence=[])
        self.assertEqual(res["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertTrue(res["abstained"])
        self.assertIn("No relevant evidence", res["reason"])

    # -------------------------------------------------------------------------
    # Category 20: Confidence Calibration & Simplex Consistency
    # -------------------------------------------------------------------------
    def test_probability_simplex_and_confidence_consistency(self):
        """All 4 class probabilities must sum to 1.0 within numerical precision."""
        claim = "Water freezes at 0 degrees Celsius."
        res = self.adapter.verify(claim=claim, user_evidence=["Under standard pressure, water freezes at 0 degrees Celsius."])
        probs = res["probabilities"]
        self.assertEqual(len(probs), 4)
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)

    # -------------------------------------------------------------------------
    # Metamorphic Property Tests
    # -------------------------------------------------------------------------
    def test_metamorphic_property_number_variation_preserves_relevance(self):
        """
        Metamorphic property: Varying the asserted numerical value (three vs seven vs twelve)
        while preserving the subject and attribute must NOT alter topical relevance or relation alignment.
        """
        passage = "An adult human respiratory system consists of two biological lungs."
        for asserted_num in ["three", "seven", "twelve", "five"]:
            claim = f"An adult human normally has {asserted_num} lungs."
            self.assertTrue(
                VerificationEngine.is_topically_relevant(claim, passage),
                f"Relevance failed for asserted number '{asserted_num}'"
            )
            self.assertTrue(
                VerificationEngine.has_claim_relation_alignment(claim, passage),
                f"Relation alignment failed for asserted number '{asserted_num}'"
            )

    def test_metamorphic_property_reordering_evidence_preserves_verdict(self):
        """
        Metamorphic property: Permuting the order of independent candidate evidence
        must preserve the consensus verdict and disparity state.
        """
        claim = "Drug Beta is effective against bacterial pneumonia."
        p_sup = "A randomized controlled trial demonstrated Drug Beta cleared bacterial pneumonia in 92% of patients."
        p_ref = "A nationwide observational study demonstrated Drug Beta failed to show efficacy against bacterial pneumonia compared to standard care."

        res_forward = self.adapter.verify(claim=claim, user_evidence=[p_sup, p_ref])
        res_reverse = self.adapter.verify(claim=claim, user_evidence=[p_ref, p_sup])

        self.assertEqual(res_forward["verdict"], res_reverse["verdict"])
        self.assertEqual(res_forward["has_cross_passage_disparity"], res_reverse["has_cross_passage_disparity"])
        self.assertEqual(res_forward["verdict"], "CONFLICTING_OR_CHERRYPICKED")


if __name__ == "__main__":
    unittest.main()
