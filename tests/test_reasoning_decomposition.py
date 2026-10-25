import os
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from unittest import TestCase
from veritas.reasoning.decomposer import ClaimDecomposer
from veritas.reasoning.types import SubClaimStatus


class TestClaimDecomposer(TestCase):

    def setUp(self):
        self.decomposer = ClaimDecomposer()

    def test_decompose_bridge_compound_claim(self):
        claim = (
            "The bridge was closed for repairs on Monday, and the transport authority "
            "announced that it reopened on Tuesday."
        )
        subclaims = self.decomposer.decompose(claim)

        # Must extract multiple atomic propositions
        self.assertGreaterEqual(len(subclaims), 2)
        subclaim_texts = [sc.text.lower() for sc in subclaims]

        # Verify key atomic components
        self.assertTrue(any("bridge was closed" in t for t in subclaim_texts))
        self.assertTrue(any("repairs" in t or "closure" in t for t in subclaim_texts))
        self.assertTrue(any("reopened on tuesday" in t for t in subclaim_texts))

        # Verify dependencies and relations
        self.assertEqual(subclaims[0].subclaim_id, "SC-01")
        self.assertIsInstance(subclaims[0].span, list if isinstance(subclaims[0].span, list) else tuple)

    def test_decompose_adversative_compound_claim(self):
        claim = (
            "The candidate won the popular vote in the election, but the supreme court "
            "halted the mandatory recount."
        )
        subclaims = self.decomposer.decompose(claim)

        self.assertEqual(len(subclaims), 2)
        self.assertEqual(subclaims[0].subclaim_id, "SC-01")
        self.assertEqual(subclaims[1].subclaim_id, "SC-02")
        self.assertIn("candidate won the popular vote", subclaims[0].text.lower())
        self.assertIn("supreme court halted the mandatory recount", subclaims[1].text.lower())
        self.assertEqual(subclaims[1].relation_to_parent, "adversative")

    def test_decompose_reporting_clause(self):
        claim = "NASA announced that liquid water was discovered on Mars."
        subclaims = self.decomposer.decompose(claim)

        self.assertEqual(len(subclaims), 2)
        self.assertIn("nasa announced", subclaims[0].text.lower())
        self.assertIn("liquid water was discovered on mars", subclaims[1].text.lower())
        self.assertIn("SC-01", subclaims[1].dependencies)

    def test_atomic_claim_preserved_single_subclaim(self):
        claim = "Water boils at 100 degrees Celsius under standard atmospheric pressure."
        subclaims = self.decomposer.decompose(claim)

        self.assertEqual(len(subclaims), 1)
        self.assertEqual(subclaims[0].subclaim_id, "SC-01")
        self.assertEqual(subclaims[0].text, claim)
        self.assertEqual(subclaims[0].relation_to_parent, "root")
        self.assertEqual(subclaims[0].dependencies, [])

    def test_empty_claim_returns_empty_list(self):
        self.assertEqual(self.decomposer.decompose(""), [])
        self.assertEqual(self.decomposer.decompose("   \n\t "), [])
