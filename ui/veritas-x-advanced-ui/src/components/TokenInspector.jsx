import React, { useState } from 'react';
import { Icon } from './Icons.jsx';

/**
 * Universal POS category color tokens & descriptive labels.
 */
const UPOS_MAP = {
  NOUN: { label: 'Noun', color: 'upos-noun', desc: 'Concrete or abstract entity' },
  PROPN: { label: 'Proper Noun', color: 'upos-propn', desc: 'Named individual, place, or organization' },
  VERB: { label: 'Verb', color: 'upos-verb', desc: 'Action, event, or state' },
  AUX: { label: 'Auxiliary Verb', color: 'upos-aux', desc: 'Modal or auxiliary tense marker' },
  ADJ: { label: 'Adjective', color: 'upos-adj', desc: 'Property or modifier of a noun' },
  ADV: { label: 'Adverb', color: 'upos-adv', desc: 'Modifier of a verb, adjective, or clause' },
  PRON: { label: 'Pronoun', color: 'upos-pron', desc: 'Substitute for a noun phrase' },
  DET: { label: 'Determiner', color: 'upos-det', desc: 'Definite/indefinite reference marker' },
  ADP: { label: 'Adposition', color: 'upos-adp', desc: 'Preposition or postposition' },
  CCONJ: { label: 'Coord Conjunction', color: 'upos-cconj', desc: 'Linking coordinating conjunction' },
  SCONJ: { label: 'Subord Conjunction', color: 'upos-sconj', desc: 'Subordinating clause marker' },
  PART: { label: 'Particle / Negation', color: 'upos-part', desc: 'Grammatical particle or negation operator' },
  NUM: { label: 'Numeral', color: 'upos-num', desc: 'Quantitative number or cardinal' },
  PUNCT: { label: 'Punctuation', color: 'upos-punct', desc: 'Sentence or clause delimiter' },
  SYM: { label: 'Symbol', color: 'upos-sym', desc: 'Mathematical or currency symbol' },
  X: { label: 'Other', color: 'upos-other', desc: 'Unclassified foreign or novel token' },
};

/**
 * TokenInspector Component
 * Stage 5: Displays authentic token boundaries, lemmas, Universal POS & Penn Treebank tags.
 */
export function TokenInspector({
  tokens = [],
  tokenCount = 0,
  sanitizedClaim = '',
}) {
  const [selectedTokenIndex, setSelectedTokenIndex] = useState(0);
  const [filterCategory, setFilterCategory] = useState('ALL');

  const safeTokens = Array.isArray(tokens) ? tokens : [];
  const selectedToken = safeTokens[selectedTokenIndex] || safeTokens[0] || null;

  // Filter categories
  const filteredTokens = safeTokens.filter((tok) => {
    if (filterCategory === 'ALL') return true;
    if (filterCategory === 'NOUNS') return tok.upos === 'NOUN' || tok.upos === 'PROPN';
    if (filterCategory === 'VERBS') return tok.upos === 'VERB' || tok.upos === 'AUX';
    if (filterCategory === 'MODIFIERS') return tok.upos === 'ADJ' || tok.upos === 'ADV';
    if (filterCategory === 'OPERATORS') return tok.upos === 'PART' || tok.upos === 'CCONJ' || tok.upos === 'SCONJ';
    return true;
  });

  return (
    <div className="token-inspector-workspace" role="region" aria-label="Tokenization and Morphological Inspector">
      {/* Stage Header */}
      <div className="token-inspector-header">
        <div className="token-header-meta">
          <span className="cinematic-eyebrow">
            <span className="eyebrow-glow-dot" />
            STAGE 05 &bull; MORPHOLOGICAL TOKENIZATION
          </span>
          <h3 className="cinematic-scene-title">
            Token & POS <span>Boundary Decomposition.</span>
          </h3>
          <p className="cinematic-scene-desc">
            Authentic rule-based morphological tokens, lemmatization roots, character offset ranges, and Penn Treebank part-of-speech annotations.
          </p>
        </div>

        <div className="token-metrics-strip">
          <div className="token-metric-pill">
            <span className="metric-label">TOTAL TOKENS</span>
            <strong className="metric-val">{tokenCount || safeTokens.length}</strong>
          </div>
          <div className="token-metric-pill">
            <span className="metric-label">CHAR LENGTH</span>
            <strong className="metric-val">{sanitizedClaim.length}</strong>
          </div>
        </div>
      </div>

      {/* Filter categories */}
      <div className="token-filters-row">
        <span className="filter-title">FILTER BY SYNTACTIC CLASS:</span>
        {[
          { id: 'ALL', label: 'All Tokens' },
          { id: 'NOUNS', label: 'Nouns & Entities' },
          { id: 'VERBS', label: 'Verbs & Actions' },
          { id: 'MODIFIERS', label: 'Modifiers (Adj/Adv)' },
          { id: 'OPERATORS', label: 'Particles & Negation' },
        ].map((f) => (
          <button
            key={f.id}
            type="button"
            className={`token-filter-chip ${filterCategory === f.id ? 'active' : ''}`}
            onClick={() => setFilterCategory(f.id)}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Interactive Token Stream */}
      <div className="tokens-stream-container" role="list" aria-label="Tokenized Stream">
        {filteredTokens.length === 0 ? (
          <div className="no-tokens-placeholder">No tokens match the active filter.</div>
        ) : (
          filteredTokens.map((tok, idx) => {
            const originalIndex = safeTokens.indexOf(tok);
            const isSelected = originalIndex === selectedTokenIndex;
            const uposMeta = UPOS_MAP[tok.upos] || UPOS_MAP.X;

            return (
              <button
                key={`${tok.text}-${tok.start_char}-${idx}`}
                type="button"
                role="listitem"
                className={`token-chip-button ${isSelected ? 'token-selected' : ''} ${uposMeta.color}`}
                onClick={() => setSelectedTokenIndex(originalIndex)}
                title={`Click to inspect '${tok.text}' [${tok.start_char}:${tok.end_char}]`}
              >
                <span className="token-chip-text">{tok.text}</span>
                <span className="token-chip-pos">{tok.upos || tok.pos}</span>
              </button>
            );
          })
        )}
      </div>

      {/* Detail Inspector Card for Active Token */}
      {selectedToken && (
        <div className="token-detail-card" role="region" aria-label="Selected Token Metadata">
          <div className="token-detail-header">
            <div className="token-title-group">
              <span className="token-sublabel">SELECTED TOKEN</span>
              <h4 className="token-main-title">
                "{selectedToken.text}"
                {selectedToken.lemma && selectedToken.lemma.toLowerCase() !== selectedToken.text.toLowerCase() && (
                  <span className="token-lemma-tag">
                    &rarr; Lemma: <code>{selectedToken.lemma}</code>
                  </span>
                )}
              </h4>
            </div>

            <div className="token-pos-badge-group">
              <span className={`token-badge ${(UPOS_MAP[selectedToken.upos] || UPOS_MAP.X).color}`}>
                UPOS: {selectedToken.upos}
              </span>
              <span className="token-badge badge-penn">
                Penn: {selectedToken.pos}
              </span>
            </div>
          </div>

          <div className="token-attributes-grid">
            <div className="token-attr-item">
              <span className="attr-label">Character Offsets</span>
              <strong className="attr-value">
                [{selectedToken.start_char} &ndash; {selectedToken.end_char}]
              </strong>
              <span className="attr-sub">Zero-indexed boundary</span>
            </div>

            <div className="token-attr-item">
              <span className="attr-label">Root Lemma</span>
              <strong className="attr-value">
                {selectedToken.lemma || selectedToken.text}
              </strong>
              <span className="attr-sub">Base uninflected dictionary form</span>
            </div>

            <div className="token-attr-item">
              <span className="attr-label">Syntactic Function</span>
              <strong className="attr-value">
                {(UPOS_MAP[selectedToken.upos] || UPOS_MAP.X).label}
              </strong>
              <span className="attr-sub">{(UPOS_MAP[selectedToken.upos] || UPOS_MAP.X).desc}</span>
            </div>

            <div className="token-attr-item">
              <span className="attr-label">In-Claim Context</span>
              <div className="attr-context-preview">
                {sanitizedClaim.slice(0, selectedToken.start_char)}
                <mark className="context-highlight">
                  {sanitizedClaim.slice(selectedToken.start_char, selectedToken.end_char)}
                </mark>
                {sanitizedClaim.slice(selectedToken.end_char)}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default TokenInspector;
