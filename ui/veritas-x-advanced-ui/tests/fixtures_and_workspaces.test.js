import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeResponse } from '../src/lib/api.js';

describe('VERITAS-X Fixture Workflow & Workspace Integrity Tests', () => {
  const FIXTURE_CASES = [
    {
      id: 'VX-DEMO-014',
      claim: 'Pure water freezes at 0 degrees Celsius at sea level.',
      verdict: 'VERIFIED_TRUE',
      confidence: 0.91,
      evidence: [
        { id: 'E-01', text: 'A water molecule consists of two hydrogen atoms bonded to one oxygen atom (H₂O). Under standard pressure at sea level, it freezes at 0°C.', stance: 'support', is_informative: true },
        { id: 'E-02', text: 'The Celsius temperature scale defines zero degrees as the melting and freezing point of water at 1 atm.', stance: 'support', is_informative: true }
      ]
    },
    {
      id: 'VX-DEMO-027',
      claim: 'The Moon is a star that produces its own light through nuclear fusion.',
      verdict: 'VERIFIED_FALSE',
      confidence: 0.86,
      evidence: [
        { id: 'E-03', text: 'The Moon is a natural rocky satellite that reflects solar illumination; it lacks sufficient mass for nuclear fusion.', stance: 'refute', is_informative: true },
        { id: 'E-04', text: 'Stars generate light through sustained thermonuclear reactions in their cores, whereas the Moon is cold rock.', stance: 'refute', is_informative: true }
      ]
    },
    {
      id: 'VX-DEMO-038',
      claim: 'The municipal bridge was completely closed to vehicle traffic on Monday.',
      verdict: 'CONFLICTING_OR_CHERRYPICKED',
      confidence: 0.46,
      evidence: [
        { id: 'E-05', text: 'The municipal public works bulletin stated the bridge was closed to all traffic on Monday for emergency span repairs.', stance: 'support', is_informative: true },
        { id: 'E-06', text: 'The metropolitan transport authority stated that the bridge remained fully open to commercial and private vehicles on Monday.', stance: 'refute', is_informative: true }
      ]
    },
    {
      id: 'VX-DEMO-049',
      claim: 'Extraterrestrial spacecraft landed in the central city plaza yesterday evening.',
      verdict: 'INSUFFICIENT_EVIDENCE',
      confidence: 0.39,
      evidence: [
        { id: 'E-07', text: 'No municipal records, aviation telemetry, or verified journalistic reports document unconfirmed aerial landings in the city plaza.', stance: 'neutral', is_informative: false }
      ]
    }
  ];

  test('All four built-in reference fixtures are complete with non-empty claims and premises', () => {
    assert.equal(FIXTURE_CASES.length, 4, 'Must have exactly 4 reference fixtures');
    for (const f of FIXTURE_CASES) {
      assert.ok(f.claim && f.claim.length > 10, `Fixture ${f.id} must have a descriptive claim`);
      assert.ok(Array.isArray(f.evidence) && f.evidence.length >= 1, `Fixture ${f.id} must have premise passages`);
      for (const e of f.evidence) {
        assert.ok(e.text && e.text.length > 20, `Evidence ${e.id} in ${f.id} must have text content`);
      }
    }
  });

  test('CASE 03 Conflicting Evidence extracts distinct opposing premise passages', () => {
    const case03 = FIXTURE_CASES[2];
    assert.equal(case03.id, 'VX-DEMO-038');
    assert.equal(case03.verdict, 'CONFLICTING_OR_CHERRYPICKED');
    const premiseTexts = case03.evidence.map(e => e.text).filter(Boolean);
    assert.equal(premiseTexts.length, 2);
    assert.match(premiseTexts[0], /closed to all traffic/i);
    assert.match(premiseTexts[1], /remained fully open/i);
  });

  test('normalizeResponse preserves user_supplied mode and candidate premise flags', () => {
    const rawUserSupplied = {
      claim: 'The municipal bridge was completely closed to vehicle traffic on Monday.',
      verdict: 'CONFLICTING_OR_CHERRYPICKED',
      confidence: 0.73,
      mode: 'user_supplied',
      disparity: true,
      has_cross_passage_disparity: true,
      evidence: [
        {
          id: 'E-01',
          text: 'The municipal public works bulletin stated the bridge was closed to all traffic on Monday.',
          stance: 'support',
          source_type: 'user_supplied',
          is_informative: true,
          is_claim_aligned: true
        },
        {
          id: 'E-02',
          text: 'The metropolitan transport authority stated that the bridge remained fully open.',
          stance: 'refute',
          source_type: 'user_supplied',
          is_informative: true,
          is_claim_aligned: true
        }
      ]
    };

    const norm = normalizeResponse(rawUserSupplied, rawUserSupplied.claim);
    assert.equal(norm.mode, 'user_supplied');
    assert.equal(norm.disparity, true);
    assert.equal(norm.evidence.length, 2);
    assert.equal(norm.evidence[0].source_type, 'user_supplied');
    assert.equal(norm.evidence[1].source_type, 'user_supplied');
  });

  test('Uninformative or misaligned passages are excluded from active support/refute tally', () => {
    const raw = {
      claim: 'Sample claim',
      verdict: 'INSUFFICIENT_EVIDENCE',
      confidence: 0.35,
      mode: 'local',
      evidence: [
        { id: 'E-01', text: 'Unrelated text', stance: 'refute', is_informative: false, is_claim_aligned: false },
        { id: 'E-02', text: 'Another unrelated text', stance: 'support', is_informative: false, is_claim_aligned: true },
        { id: 'E-03', text: 'Valid informative supporting text', stance: 'support', is_informative: true, is_claim_aligned: true }
      ]
    };

    const norm = normalizeResponse(raw, raw.claim);
    const activeSupporters = norm.evidence.filter(e => e.stance === 'support' && e.is_informative !== false && e.is_claim_aligned !== false);
    const activeRefuters = norm.evidence.filter(e => e.stance === 'refute' && e.is_informative !== false && e.is_claim_aligned !== false);

    assert.equal(activeSupporters.length, 1, 'Only E-03 should count as active supporter');
    assert.equal(activeRefuters.length, 0, 'E-01 should not count as active refuter because is_informative=false and is_claim_aligned=false');
  });

  test('All eight workspace navigation routes are defined and distinct', () => {
    const WORKSPACES = [
      { id: 'overview', label: 'Command Center' },
      { id: 'investigate', label: 'Claim Investigator' },
      { id: 'evidence', label: 'Evidence Dossier' },
      { id: 'graph', label: 'Evidence Graph' },
      { id: 'timeline', label: 'Temporal & Provenance Timeline' },
      { id: 'multihop', label: 'Multi-Hop Reasoning' },
      { id: 'evaluation', label: 'Model Performance Lab' },
      { id: 'system', label: 'System Telemetry' }
    ];

    assert.equal(WORKSPACES.length, 8, 'There must be exactly 8 workspaces');
    const ids = new Set(WORKSPACES.map(w => w.id));
    assert.equal(ids.size, 8, 'All 8 workspace IDs must be unique');
    assert.ok(ids.has('investigate'), 'Primary workspace must be investigate');
  });
});
