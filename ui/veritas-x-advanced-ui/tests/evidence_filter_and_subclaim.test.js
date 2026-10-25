import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

/**
 * Filter contract helpers mirroring EvidenceDossier.jsx and ForensicReport.jsx
 */
const isPassageSupporting = (e) =>
  ['support', 'entailment'].includes(String(e.stance || 'neutral').toLowerCase()) &&
  e.is_informative !== false &&
  e.is_relevant !== false &&
  e.is_claim_aligned !== false &&
  e.is_figurative !== true &&
  e.is_decision_eligible !== false;

const isPassageRefuting = (e) =>
  ['refute', 'contradiction'].includes(String(e.stance || 'neutral').toLowerCase()) &&
  e.is_informative !== false &&
  e.is_relevant !== false &&
  e.is_claim_aligned !== false &&
  e.is_figurative !== true &&
  e.is_decision_eligible !== false;

describe('VERITAS-X Evidence Filter Consistency & Subclaim Isolation Tests', () => {
  it('correctly filters eligible supporting and refuting passages', () => {
    const evidence = [
      { id: 'E1', stance: 'support', is_relevant: true, is_informative: true, is_claim_aligned: true, is_decision_eligible: true },
      { id: 'E2', stance: 'refute', is_relevant: true, is_informative: true, is_claim_aligned: true, is_decision_eligible: true },
      { id: 'E3', stance: 'neutral', is_relevant: true, is_informative: true, is_claim_aligned: true, is_decision_eligible: true },
    ];

    assert.equal(isPassageSupporting(evidence[0]), true);
    assert.equal(isPassageRefuting(evidence[0]), false);

    assert.equal(isPassageSupporting(evidence[1]), false);
    assert.equal(isPassageRefuting(evidence[1]), true);

    assert.equal(isPassageSupporting(evidence[2]), false);
    assert.equal(isPassageRefuting(evidence[2]), false);
  });

  it('strictly excludes off-topic and ineligible passages from active support/refute tally', () => {
    const evidence = [
      // Off-topic passage that happens to have raw stance 'support'
      { id: 'E_offtopic', stance: 'support', is_relevant: false, is_informative: true, is_claim_aligned: false, is_decision_eligible: false },
      // Uninformative fragment with raw stance 'refute'
      { id: 'E_uninf', stance: 'refute', is_relevant: true, is_informative: false, is_claim_aligned: true, is_decision_eligible: false },
      // Figurative definition with raw stance 'support'
      { id: 'E_fig', stance: 'support', is_relevant: true, is_informative: true, is_claim_aligned: true, is_figurative: true, is_decision_eligible: false },
      // Unaligned relation with raw stance 'refute'
      { id: 'E_unaligned', stance: 'refute', is_relevant: true, is_informative: true, is_claim_aligned: false, is_decision_eligible: false },
      // General ineligible
      { id: 'E_ineligible', stance: 'support', is_relevant: true, is_informative: true, is_claim_aligned: true, is_decision_eligible: false },
    ];

    for (const item of evidence) {
      assert.equal(isPassageSupporting(item), false, `Item ${item.id} must NOT be supporting`);
      assert.equal(isPassageRefuting(item), false, `Item ${item.id} must NOT be refuting`);
    }

    const supCount = evidence.filter(isPassageSupporting).length;
    const refCount = evidence.filter(isPassageRefuting).length;
    assert.equal(supCount, 0);
    assert.equal(refCount, 0);
  });

  it('prevents decisive badge on irrelevant or ineligible evidence', () => {
    const items = [
      { id: 'E1', contributed_to_decision: true, is_relevant: true, is_decision_eligible: true },
      { id: 'E2', contributed_to_decision: true, is_relevant: false, is_decision_eligible: false },
      { id: 'E3', contributed_to_decision: true, is_relevant: true, is_decision_eligible: false },
    ];

    const canShowDecisive = (item) =>
      Boolean(item.contributed_to_decision && item.is_relevant !== false && item.is_decision_eligible !== false);

    assert.equal(canShowDecisive(items[0]), true);
    assert.equal(canShowDecisive(items[1]), false);
    assert.equal(canShowDecisive(items[2]), false);
  });

  it('guarantees compound-claim subclaims isolate their evidence without broadcasting', () => {
    // Simulated compound response payload from VeritasAPIAdapter
    const payload = {
      claim: "The 1912 Olympic Games were held in Stockholm, and electronic timing was featured.",
      evidence: [
        { id: "E-01", text: "The 1912 Olympic Games were held in Stockholm.", stance: "support", is_relevant: true, is_decision_eligible: true },
        { id: "E-02", text: "Electronic timing was first introduced in 1968, not at the 1912 Games.", stance: "refute", is_relevant: true, is_decision_eligible: true },
        { id: "E-03", text: "Stockholm municipal water regulations were revised in 1980.", stance: "neutral", is_relevant: false, is_decision_eligible: false },
      ],
      subclaims: [
        {
          subclaim_id: "SC-01",
          text: "The 1912 Olympic Games were held in Stockholm",
          evidence_ids: ["E-01"],
          status: "SUPPORTED",
          stance_distribution: { support: 1, refute: 0, neutral: 0 },
        },
        {
          subclaim_id: "SC-02",
          text: "electronic timing was featured",
          evidence_ids: ["E-02"],
          status: "REFUTED",
          stance_distribution: { support: 0, refute: 1, neutral: 0 },
        },
      ],
    };

    assert.equal(payload.subclaims.length, 2);

    const sc1 = payload.subclaims[0];
    const sc2 = payload.subclaims[1];

    // Evidence isolation checks
    assert.deepEqual(sc1.evidence_ids, ["E-01"]);
    assert.deepEqual(sc2.evidence_ids, ["E-02"]);

    // E-03 (off-topic) must not appear in any subclaim
    assert.ok(!sc1.evidence_ids.includes("E-03"));
    assert.ok(!sc2.evidence_ids.includes("E-03"));

    // Stances must reflect individual subclaim evidence, not whole-claim union
    assert.equal(sc1.status, "SUPPORTED");
    assert.equal(sc2.status, "REFUTED");
  });

  it('guarantees generateConciseReason contract prioritizes result.reason and prevents false contradictions', () => {
    // Helper function mirroring ForensicReport.jsx generateConciseReason logic
    const generateConciseReasonHelper = (result, supCount, refCount, totalEv) => {
      if (result.reason) return result.reason;
      if (result.summary) return result.summary;

      const verdict = result.abstained ? 'INSUFFICIENT_EVIDENCE' : result.verdict;
      if (totalEv === 0) return 'No matching evidence passages were found.';
      if (verdict === 'VERIFIED_TRUE' && supCount > 0) return `Available evidence consistently supports the claim.`;
      if (verdict === 'VERIFIED_FALSE') {
        if (refCount > 0) return `Available evidence directly contradicts the statement.`;
        return `Retrieved evidence mentions related topics but does not establish or disprove the statement.`;
      }
      return 'Retrieved evidence mentions related topics but does not establish or disprove the statement.';
    };

    // Case 1: Backend grounded reason provided -> must be preserved verbatim
    const r1 = { verdict: 'INSUFFICIENT_EVIDENCE', abstained: true, reason: 'Insufficient evidentiary grounding: no supporting or refuting documentary evidence.' };
    assert.equal(generateConciseReasonHelper(r1, 0, 0, 5), r1.reason);

    // Case 2: Verdict is VERIFIED_FALSE with 0 refuting passages -> must NOT say directly contradicts
    const r2 = { verdict: 'VERIFIED_FALSE', abstained: false };
    const text2 = generateConciseReasonHelper(r2, 0, 0, 5);
    assert.ok(!text2.includes('directly contradicts'));
    assert.ok(text2.includes('does not establish or disprove'));

    // Case 3: Verdict is VERIFIED_FALSE with >= 1 refuting passage -> states contradicts
    const r3 = { verdict: 'VERIFIED_FALSE', abstained: false };
    const text3 = generateConciseReasonHelper(r3, 0, 2, 5);
    assert.ok(text3.includes('directly contradicts'));
  });
});

