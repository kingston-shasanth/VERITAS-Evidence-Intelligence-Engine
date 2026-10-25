import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeResponse } from '../src/lib/api.js';

describe('VERITAS-X Evidence Mode & Provenance Normalization Tests', () => {
  test('Local mode strictly marks evidence as local even when archived source_url exists', () => {
    const rawLocalResult = {
      claim_id: 'VX-LOC-01',
      claim: 'Water expands when freezing.',
      verdict: 'VERIFIED_TRUE',
      confidence: 0.95,
      mode: 'local',
      retrieval_status: {
        mode: 'local',
        provider: 'local_corpus',
        status: 'success',
        results_count: 2,
        local_count: 2,
        web_count: 0,
        time_ms: 15.2,
      },
      evidence: [
        {
          id: 'E-01',
          doc_id: 'averitec_doc_101',
          source: 'https://archived.independent.co.uk/science/water-ice',
          source_title: 'Independent Science Archive',
          source_url: 'https://archived.independent.co.uk/science/water-ice',
          text: 'Liquid water expands by approximately 9% upon freezing into ice.',
          stance: 'support',
          score: 0.94,
        },
        {
          id: 'E-02',
          doc_id: 'averitec_doc_102',
          source: 'Local Evidence Corpus',
          source_title: 'AVeriTeC Physics Compendium',
          text: 'Due to hydrogen bonding, ice is less dense than liquid water.',
          stance: 'support',
          score: 0.91,
        },
      ],
    };

    const norm = normalizeResponse(rawLocalResult, rawLocalResult.claim);

    assert.equal(norm.mode, 'local');
    assert.equal(norm.retrieval_status.mode, 'local');
    assert.equal(norm.retrieval_status.local_count, 2);
    assert.equal(norm.retrieval_status.web_count, 0);

    // Both evidence items must be strictly 'local'
    assert.equal(norm.evidence.length, 2);
    assert.equal(norm.evidence[0].source_type, 'local', 'Local mode passage must have source_type="local" even with archived URL');
    assert.equal(norm.evidence[1].source_type, 'local', 'Local mode passage must have source_type="local"');

    // URLs may exist for archived citation without changing source_type to web
    assert.equal(norm.evidence[0].source_url, 'https://archived.independent.co.uk/science/water-ice');
  });

  test('Web mode strictly marks evidence as web with search snippet labels', () => {
    const rawWebResult = {
      claim_id: 'VX-WEB-02',
      claim: 'James Webb Space Telescope discovered distant galaxy.',
      verdict: 'VERIFIED_TRUE',
      confidence: 0.89,
      mode: 'web',
      retrieval_status: {
        mode: 'web',
        provider: 'tavily',
        status: 'success',
        results_count: 1,
        local_count: 0,
        web_count: 1,
        time_ms: 480.0,
      },
      evidence: [
        {
          id: 'E-01',
          doc_id: 'web_99887766',
          source: 'https://nasa.gov/news/jwst-galaxy',
          source_title: 'NASA News Dispatch',
          source_url: 'https://nasa.gov/news/jwst-galaxy',
          text: 'JWST detected infrared emission from high-redshift galaxy.',
          stance: 'support',
          score: 0.89,
        },
      ],
    };

    const norm = normalizeResponse(rawWebResult, rawWebResult.claim);

    assert.equal(norm.mode, 'web');
    assert.equal(norm.retrieval_status.web_count, 1);
    assert.equal(norm.evidence[0].source_type, 'web');
    assert.equal(norm.evidence[0].source_url, 'https://nasa.gov/news/jwst-galaxy');
  });

  test('Hybrid mode preserves distinct local and web provenance individually', () => {
    const rawHybridResult = {
      claim_id: 'VX-HYB-03',
      claim: 'The global average temperature was verified.',
      verdict: 'VERIFIED_TRUE',
      confidence: 0.92,
      mode: 'hybrid',
      retrieval_status: {
        mode: 'hybrid',
        provider: 'hybrid_local_and_web',
        status: 'success',
        results_count: 2,
        local_count: 1,
        web_count: 1,
        time_ms: 320.0,
      },
      evidence: [
        {
          id: 'E-01',
          doc_id: 'loc_doc_climate',
          source_type: 'local',
          source_title: 'AVeriTeC Climate Records',
          text: 'Historical meteorological measurements show warming trends.',
          stance: 'support',
          score: 0.91,
        },
        {
          id: 'E-02',
          doc_id: 'web_e1f2a3b4',
          source_type: 'web',
          source_title: 'WMO 2026 Climate Update',
          source_url: 'https://wmo.int/climate/2026',
          text: 'WMO confirmed warmest decade on observational record.',
          stance: 'support',
          score: 0.93,
        },
      ],
    };

    const norm = normalizeResponse(rawHybridResult, rawHybridResult.claim);

    assert.equal(norm.mode, 'hybrid');
    assert.equal(norm.retrieval_status.local_count, 1);
    assert.equal(norm.retrieval_status.web_count, 1);

    assert.equal(norm.evidence[0].source_type, 'local');
    assert.equal(norm.evidence[1].source_type, 'web');
  });

  test('User-supplied mode preserves user_supplied provenance', () => {
    const rawUserResult = {
      claim_id: 'VX-USR-04',
      claim: 'The meeting will take place at 10 AM.',
      verdict: 'VERIFIED_TRUE',
      confidence: 0.98,
      mode: 'user_supplied',
      retrieval_status: {
        mode: 'user_supplied',
        provider: 'user',
        status: 'success',
        results_count: 1,
        local_count: 0,
        web_count: 0,
      },
      evidence: [
        {
          id: 'E-01',
          doc_id: 'user_doc_01',
          source_title: 'User Supplied Passage #1',
          text: 'Official notice confirms the meeting begins at 10 AM sharp.',
          stance: 'support',
          score: 1.0,
        },
      ],
    };

    const norm = normalizeResponse(rawUserResult, rawUserResult.claim);
    assert.equal(norm.mode, 'user_supplied');
    assert.equal(norm.evidence[0].source_type, 'user_supplied');
  });
});

describe('VERITAS-X Provenance Badge & Banner Presentation Helper Tests', () => {
  // Helper simulating the ForensicReport banner logic
  function getBannerPresentation(result) {
    const mode = result.mode;
    const rStat = result.retrieval_status || {};
    const isSuccess = ['ok', 'success'].includes(rStat.status);
    const localCount = rStat.local_count ?? (mode === 'local' ? (rStat.results_count ?? 0) : 0);
    const webCount = rStat.web_count ?? (mode === 'web' ? (rStat.results_count ?? 0) : 0);
    const timeMs = rStat.time_ms ?? 0;

    let title = '';
    let description = '';

    if (mode === 'local') {
      title = isSuccess ? 'Local Corpus Mode: Retrieval Complete' : 'Local Corpus Mode: Retrieval Limitation';
      description = `Retrieved ${localCount} verified document passage(s) from the local offline corpus (1,047 articles). Offline evaluation with zero external API calls.`;
    } else if (mode === 'web') {
      title = isSuccess ? 'Live Web Retrieval Mode: Online Search Complete' : 'Live Web Retrieval Mode: Search Limitation';
      description = `Retrieved ${webCount} web citation snippet(s) from Tavily search (${timeMs}ms).`;
    } else if (mode === 'hybrid') {
      if (rStat.is_local_only) {
        title = 'Hybrid Retrieval Mode: Local Corpus Fallback';
        description = rStat.notice || `Adjudicated strictly using ${localCount} local corpus passage(s).`;
      } else {
        title = isSuccess ? 'Hybrid Retrieval Mode: Multi-Source Search Complete' : 'Hybrid Retrieval Mode: Retrieval Limitation';
        description = `Retrieved ${localCount} local corpus passage(s) and ${webCount} web citation snippet(s) from Tavily (${timeMs}ms). Merged and deduplicated across sources.`;
      }
    }

    return { title, description };
  }

  // Helper simulating evidence card badges
  function getEvidenceBadges(item) {
    const isWeb = item.source_type === 'web';
    const isUser = item.source_type === 'user_supplied' || item.source_type === 'user';
    const provenanceLabel = isWeb ? 'LIVE WEB' : isUser ? 'USER SUPPLIED' : 'LOCAL CORPUS';
    const hasSearchSnippet = isWeb;

    return { provenanceLabel, hasSearchSnippet };
  }

  test('Local mode banner never reports Tavily or Online Search', () => {
    const result = {
      mode: 'local',
      retrieval_status: { status: 'success', local_count: 5, web_count: 0, results_count: 5 },
    };
    const { title, description } = getBannerPresentation(result);

    assert.equal(title, 'Local Corpus Mode: Retrieval Complete');
    assert.ok(description.includes('local offline corpus'));
    assert.ok(description.includes('zero external API calls'));
    assert.ok(!description.includes('Tavily'));
    assert.ok(!title.includes('Online Search Complete'));
  });

  test('Local evidence cards never display LIVE WEB or SEARCH SNIPPET badges', () => {
    const localItem = {
      id: 'E-01',
      source_type: 'local',
      source_url: 'https://archived.org/ref', // even with URL
    };
    const { provenanceLabel, hasSearchSnippet } = getEvidenceBadges(localItem);

    assert.equal(provenanceLabel, 'LOCAL CORPUS');
    assert.equal(hasSearchSnippet, false, 'SEARCH SNIPPET must not appear for local corpus evidence');
  });

  test('Web evidence cards display LIVE WEB and SEARCH SNIPPET badges', () => {
    const webItem = {
      id: 'E-01',
      source_type: 'web',
      source_url: 'https://live-domain.com/article',
    };
    const { provenanceLabel, hasSearchSnippet } = getEvidenceBadges(webItem);

    assert.equal(provenanceLabel, 'LIVE WEB');
    assert.equal(hasSearchSnippet, true, 'SEARCH SNIPPET badge must appear for live web evidence');
  });

  test('Hybrid mode banner reflects combined source counts and individual item badges', () => {
    const hybridResult = {
      mode: 'hybrid',
      retrieval_status: { status: 'success', local_count: 3, web_count: 2, time_ms: 250 },
    };
    const { title, description } = getBannerPresentation(hybridResult);

    assert.equal(title, 'Hybrid Retrieval Mode: Multi-Source Search Complete');
    assert.ok(description.includes('3 local corpus passage(s)'));
    assert.ok(description.includes('2 web citation snippet(s) from Tavily'));

    const localItem = { source_type: 'local' };
    const webItem = { source_type: 'web' };

    assert.equal(getEvidenceBadges(localItem).provenanceLabel, 'LOCAL CORPUS');
    assert.equal(getEvidenceBadges(localItem).hasSearchSnippet, false);

    assert.equal(getEvidenceBadges(webItem).provenanceLabel, 'LIVE WEB');
    assert.equal(getEvidenceBadges(webItem).hasSearchSnippet, true);
  });
});
