"""
Reasoning Graph Builder for VERITAS-X 2.0.
Constructs an auditable, interactive directed graph representing the
exact topology between root claims, decomposed subclaims, retrieved passages,
and primary documentary sources.
Every node and edge is grounded in real verification pipeline outputs.
"""
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
from veritas.reasoning.types import (
    SubClaim,
    SubClaimStatus,
    GraphNode,
    GraphEdge,
    ReasoningGraph,
)


class ReasoningGraphBuilder:
    """
    Builds the multi-tiered evidence relationship graph for UI visualization
    and structured topological analysis.
    """

    def build_graph(
        self,
        claim_id: str,
        claim_text: str,
        verdict: str,
        confidence: float,
        subclaims: List[SubClaim],
        evidence_items: List[Dict[str, Any]],
        temporal_events: Optional[List[Dict[str, Any]]] = None,
    ) -> ReasoningGraph:
        """
        Assembles graph nodes and edges from pipeline artifacts.
        """
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []
        node_id_set = set()

        # 1. Root Claim Node
        claim_node_id = "node-claim-root"
        nodes.append(
            GraphNode(
                id=claim_node_id,
                type="claim",
                label=claim_id,
                text=claim_text,
                verdict=verdict,
                confidence=confidence,
                metadata={"level": 0, "role": "root"},
            )
        )
        node_id_set.add(claim_node_id)

        # 2. Subclaim Nodes
        sc_node_map: Dict[str, str] = {}
        for sc in subclaims:
            sc_node_id = f"node-subclaim-{sc.subclaim_id}"
            sc_node_map[sc.subclaim_id] = sc_node_id
            nodes.append(
                GraphNode(
                    id=sc_node_id,
                    type="subclaim",
                    label=sc.subclaim_id,
                    text=sc.text,
                    verdict=sc.verdict or "UNRESOLVED",
                    confidence=sc.confidence,
                    stance=sc.status.value.lower(),
                    metadata={
                        "level": 1,
                        "span": sc.span,
                        "relation": sc.relation_to_parent,
                        "status": sc.status.value,
                    },
                )
            )
            node_id_set.add(sc_node_id)

            # Edge: Claim -> Subclaim
            edges.append(
                GraphEdge(
                    source=claim_node_id,
                    target=sc_node_id,
                    relation="decomposes_to",
                    weight=1.0,
                    label=f"Decomposed into {sc.subclaim_id}",
                    metadata={"relation_type": sc.relation_to_parent},
                )
            )

        # Edges between dependent subclaims
        for sc in subclaims:
            for dep_id in sc.dependencies:
                if dep_id in sc_node_map:
                    edges.append(
                        GraphEdge(
                            source=sc_node_map[dep_id],
                            target=sc_node_map[sc.subclaim_id],
                            relation="depends_on",
                            weight=0.8,
                            label="Reasoning dependency",
                        )
                    )

        # 3. Source Domain Nodes & Evidence Passage Nodes
        source_domain_map: Dict[str, str] = {}
        src_counter = 1

        for ev in evidence_items:
            ev_id = ev.get("id", f"E-{len(nodes)}")
            ev_node_id = f"node-passage-{ev_id}"
            stance = ev.get("stance", "neutral")
            score = float(ev.get("score", 0.0))

            nodes.append(
                GraphNode(
                    id=ev_node_id,
                    type="passage",
                    label=ev_id,
                    text=ev.get("text", ""),
                    stance=stance,
                    score=score,
                    metadata={
                        "level": 2,
                        "entailment_prob": ev.get("entailment_prob", 0.0),
                        "contradiction_prob": ev.get("contradiction_prob", 0.0),
                        "neutral_prob": ev.get("neutral_prob", 0.0),
                        "doc_id": ev.get("doc_id"),
                        "source": ev.get("source"),
                    },
                )
            )
            node_id_set.add(ev_node_id)

            # Link Passage -> Relevant Subclaim (or Claim if single subclaim)
            target_sc_id = claim_node_id
            if subclaims:
                # Find matching subclaim by evidence association if present, or link to first relevant
                matched_sc = next((sc for sc in subclaims if ev_id in sc.evidence_ids), subclaims[0])
                target_sc_id = sc_node_map.get(matched_sc.subclaim_id, claim_node_id)

            edge_relation = "supports" if stance in ("support", "entailment") else ("refutes" if stance in ("refute", "contradiction") else "neutral")
            edges.append(
                GraphEdge(
                    source=ev_node_id,
                    target=target_sc_id,
                    relation=edge_relation,
                    weight=score,
                    label=f"{stance.upper()} ({score:.2f})",
                    metadata={"probabilities": {
                        "entailment": ev.get("entailment_prob", 0.0),
                        "contradiction": ev.get("contradiction_prob", 0.0),
                        "neutral": ev.get("neutral_prob", 0.0),
                    }},
                )
            )

            # Extract distinct source domain for provenance tier
            source_raw = str(ev.get("source", "Evidence Corpus"))
            domain_name = self._extract_clean_domain(source_raw)

            if domain_name not in source_domain_map:
                src_id = f"node-source-{src_counter:02d}"
                source_domain_map[domain_name] = src_id
                src_counter += 1
                nodes.append(
                    GraphNode(
                        id=src_id,
                        type="source",
                        label=domain_name,
                        text=f"Primary publisher: {domain_name}",
                        metadata={"level": 3, "raw_source": source_raw},
                    )
                )
                node_id_set.add(src_id)

            # Edge: Passage -> Source
            edges.append(
                GraphEdge(
                    source=ev_node_id,
                    target=source_domain_map[domain_name],
                    relation="published_by",
                    weight=1.0,
                    label="Published by",
                )
            )

        # 4. Temporal Sequencing Edges if dated events exist
        if temporal_events and len(temporal_events) >= 2:
            sorted_dated = sorted(temporal_events, key=lambda x: x["date"])
            for i in range(len(sorted_dated) - 1):
                p1_id = f"node-passage-{sorted_dated[i].get('evidence_id')}"
                p2_id = f"node-passage-{sorted_dated[i+1].get('evidence_id')}"
                if p1_id in node_id_set and p2_id in node_id_set:
                    edges.append(
                        GraphEdge(
                            source=p1_id,
                            target=p2_id,
                            relation="precedes",
                            weight=0.5,
                            label=f"Precedes ({sorted_dated[i]['date']} -> {sorted_dated[i+1]['date']})",
                            metadata={"temporal_transition": True},
                        )
                    )

        # Summary statistics
        support_count = sum(1 for e in edges if e.relation == "supports")
        refute_count = sum(1 for e in edges if e.relation == "refutes")
        neutral_count = sum(1 for e in edges if e.relation == "neutral")
        unsupported_sc = sum(1 for sc in subclaims if sc.status == SubClaimStatus.UNRESOLVED)

        summary = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "subclaims_count": len(subclaims),
            "evidence_count": len(evidence_items),
            "sources_count": len(source_domain_map),
            "support_edges": support_count,
            "refute_edges": refute_count,
            "neutral_edges": neutral_count,
            "unsupported_subclaims": unsupported_sc,
            "has_contradictory_sources": support_count > 0 and refute_count > 0,
        }

        return ReasoningGraph(nodes=nodes, edges=edges, summary=summary)

    def _extract_clean_domain(self, source_str: str) -> str:
        """Extracts clean hostname or corpus name from url or string."""
        if source_str.startswith("http://") or source_str.startswith("https://"):
            try:
                # Handle web.archive.org URLs
                if "web.archive.org/web/" in source_str:
                    parts = source_str.split("http")
                    if len(parts) >= 3:
                        inner_url = "http" + parts[-1]
                        return urlparse(inner_url).netloc.replace("www.", "")
                domain = urlparse(source_str).netloc.replace("www.", "")
                return domain if domain else source_str
            except Exception:
                return source_str
        return source_str
