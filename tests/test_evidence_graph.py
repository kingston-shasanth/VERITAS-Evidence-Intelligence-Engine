import os
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from unittest import TestCase
from veritas.graph.graph_builder import ReasoningGraphBuilder
from veritas.reasoning.types import SubClaim, SubClaimStatus


class TestReasoningGraphBuilder(TestCase):

    def setUp(self):
        self.builder = ReasoningGraphBuilder()

    def test_build_graph_structure(self):
        subclaims = [
            SubClaim(
                subclaim_id="SC-01",
                text="The bridge was closed on Monday",
                span=(0, 31),
                dependencies=[],
                status=SubClaimStatus.SUPPORTED,
                verdict="VERIFIED_TRUE",
                confidence=0.85,
                evidence_ids=["E-01"],
            ),
            SubClaim(
                subclaim_id="SC-02",
                text="The bridge reopened on Tuesday",
                span=(36, 68),
                dependencies=["SC-01"],
                status=SubClaimStatus.SUPPORTED,
                verdict="VERIFIED_TRUE",
                confidence=0.90,
                evidence_ids=["E-02"],
            ),
        ]

        evidence = [
            {
                "id": "E-01",
                "text": "Authorities closed the bridge Monday morning.",
                "source": "https://transport.gov/closure",
                "stance": "support",
                "score": 0.88,
                "entailment_prob": 0.88,
                "contradiction_prob": 0.02,
                "neutral_prob": 0.10,
            },
            {
                "id": "E-02",
                "text": "The bridge reopened to light traffic Tuesday.",
                "source": "https://news.org/reopen",
                "stance": "support",
                "score": 0.92,
                "entailment_prob": 0.92,
                "contradiction_prob": 0.01,
                "neutral_prob": 0.07,
            },
        ]

        graph = self.builder.build_graph(
            claim_id="VX-BRIDGE",
            claim_text="The bridge closed on Monday and reopened on Tuesday.",
            verdict="VERIFIED_TRUE",
            confidence=0.87,
            subclaims=subclaims,
            evidence_items=evidence,
        )

        nodes = graph.nodes
        edges = graph.edges
        summary = graph.summary

        # Check nodes
        node_types = {n.type for n in nodes}
        self.assertIn("claim", node_types)
        self.assertIn("subclaim", node_types)
        self.assertIn("passage", node_types)
        self.assertIn("source", node_types)

        # Check edge relations
        edge_relations = {e.relation for e in edges}
        self.assertIn("decomposes_to", edge_relations)
        self.assertIn("depends_on", edge_relations)
        self.assertIn("supports", edge_relations)
        self.assertIn("published_by", edge_relations)

        # Check summary stats
        self.assertEqual(summary["subclaims_count"], 2)
        self.assertEqual(summary["evidence_count"], 2)
        self.assertEqual(summary["support_edges"], 2)
        self.assertEqual(summary["refute_edges"], 0)
        self.assertFalse(summary["has_contradictory_sources"])

    def test_graph_detects_contradictory_sources(self):
        subclaims = [
            SubClaim(
                subclaim_id="SC-01",
                text="The proposal was accepted",
                status=SubClaimStatus.CONFLICTING,
                verdict="CONFLICTING_OR_CHERRYPICKED",
                confidence=0.60,
                evidence_ids=["E-01", "E-02"],
            )
        ]
        evidence = [
            {"id": "E-01", "text": "The council approved the proposal.", "source": "Source A", "stance": "support", "score": 0.85},
            {"id": "E-02", "text": "The council voted down the proposal.", "source": "Source B", "stance": "refute", "score": 0.90},
        ]

        graph = self.builder.build_graph(
            claim_id="VX-COUNCIL",
            claim_text="The proposal was accepted.",
            verdict="CONFLICTING_OR_CHERRYPICKED",
            confidence=0.60,
            subclaims=subclaims,
            evidence_items=evidence,
        )

        self.assertTrue(graph.summary["has_contradictory_sources"])
        self.assertEqual(graph.summary["support_edges"], 1)
        self.assertEqual(graph.summary["refute_edges"], 1)
