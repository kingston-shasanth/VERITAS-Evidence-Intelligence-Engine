import React from 'react';
import { Icon } from './Icons.jsx';

/**
 * SyntaxRelationshipView Component
 * Stage 6: Linguistic analysis visualizing Subject-Predicate-Object relations,
 * active vs passive voice, clause boundaries, and sentence intent constraints.
 */
export function SyntaxRelationshipView({
  linguisticDetails = {},
  sanitizedClaim = '',
}) {
  const sentenceType = linguisticDetails.sentence_type || 'assertion';
  const isQuestion = Boolean(linguisticDetails.is_question);
  const isPassive = Boolean(linguisticDetails.is_passive);
  const subject = linguisticDetails.subject_candidate;
  const predicate = linguisticDetails.predicate_candidate;
  const object = linguisticDetails.object_candidate;
  const clauses = Array.isArray(linguisticDetails.clauses) ? linguisticDetails.clauses : [];
  const parserEngine = linguisticDetails.parser_engine || 'shallow_syntactic_rule_parser';

  const hasCompleteSVO = Boolean(subject && predicate);
  const isIncomplete = !subject || !predicate;

  return (
    <div className="syntax-relationship-workspace" role="region" aria-label="Linguistic Analysis and Syntactic Construction">
      {/* Stage Header */}
      <div className="syntax-header">
        <div className="syntax-header-meta">
          <span className="cinematic-eyebrow">
            <span className="eyebrow-glow-dot" />
            STAGE 06 &bull; LINGUISTIC ANALYSIS
          </span>
          <h3 className="cinematic-scene-title">
            Syntactic Relation <span>& Clause Parsing.</span>
          </h3>
          <p className="cinematic-scene-desc">
            Rule-based grammatical decomposition identifying core predicate structures, voice alternation, and clause boundaries.
          </p>
        </div>

        <div className="syntax-engine-badge">
          <span className="engine-status-indicator" />
          <span className="engine-label">PARSER:</span>
          <code>{parserEngine}</code>
        </div>
      </div>

      {/* Grammatical Constraints & Intent Banners */}
      <div className="syntax-constraints-grid">
        <div className={`syntax-constraint-card ${isQuestion ? 'constraint-alert' : 'constraint-normal'}`}>
          <div className="constraint-icon-wrap">
            <Icon name={isQuestion ? 'alertCircle' : 'check'} size={18} />
          </div>
          <div className="constraint-text-wrap">
            <span className="constraint-title">SENTENCE INTENT</span>
            <strong className="constraint-value">
              {isQuestion ? 'Inquiry / Question' : sentenceType.toUpperCase()}
            </strong>
            <p className="constraint-desc">
              {isQuestion
                ? 'Question identified: Verification evaluates the embedded proposition without asserting truth.'
                : 'Assertive statement declaring an empirical factual proposition.'}
            </p>
          </div>
        </div>

        <div className={`syntax-constraint-card ${isPassive ? 'constraint-passive' : 'constraint-active'}`}>
          <div className="constraint-icon-wrap">
            <Icon name={isPassive ? 'rotate' : 'sparkles'} size={18} />
          </div>
          <div className="constraint-text-wrap">
            <span className="constraint-title">VOICE ORIENTATION</span>
            <strong className="constraint-value">
              {isPassive ? 'Passive Voice' : 'Active Voice'}
            </strong>
            <p className="constraint-desc">
              {isPassive
                ? 'Subject receives the action from an explicit or implied agent.'
                : 'Subject actively performs the action denoted by the predicate.'}
            </p>
          </div>
        </div>

        <div className="syntax-constraint-card constraint-clauses">
          <div className="constraint-icon-wrap">
            <Icon name="layers" size={18} />
          </div>
          <div className="constraint-text-wrap">
            <span className="constraint-title">CLAUSE COUNT</span>
            <strong className="constraint-value">
              {clauses.length || 1} {clauses.length === 1 ? 'Clause' : 'Clauses'}
            </strong>
            <p className="constraint-desc">
              {clauses.length > 1
                ? 'Compound or complex statement with multiple coordinate/subordinate clauses.'
                : 'Single atomic proposition clause.'}
            </p>
          </div>
        </div>
      </div>

      {/* S-P-O Pipeline Construction */}
      <div className="svo-pipeline-section">
        <div className="svo-section-header">
          <h4 className="svo-title">SYNTACTIC RELATION PIPELINE</h4>
          <span className="svo-confidence-tag">
            {hasCompleteSVO ? 'Relation Extracted' : 'Incomplete SVO Argument Structure'}
          </span>
        </div>

        <div className="svo-diagram-flow">
          {/* Subject Node */}
          <div className={`svo-node svo-node-subject ${!subject ? 'svo-node-missing' : ''}`}>
            <div className="svo-node-header">
              <span className="svo-badge badge-subject">SUBJECT (ARG0)</span>
              <span className="svo-role-type">Nominal Entity</span>
            </div>
            <div className="svo-node-content">
              {subject ? (
                <strong className="svo-node-text">"{subject}"</strong>
              ) : (
                <span className="svo-unresolved-text">Not confidently identified</span>
              )}
            </div>
            <div className="svo-node-footer">Entity initiating or undergoing predicate</div>
          </div>

          {/* Connection Vector 1 */}
          <div className="svo-vector-connector">
            <div className="svo-vector-line" />
            <div className="svo-vector-arrow">&rarr;</div>
            <span className="svo-vector-label">governs</span>
          </div>

          {/* Predicate Node */}
          <div className={`svo-node svo-node-predicate ${!predicate ? 'svo-node-missing' : ''}`}>
            <div className="svo-node-header">
              <span className="svo-badge badge-predicate">PREDICATE (REL)</span>
              <span className="svo-role-type">Verbal Root</span>
            </div>
            <div className="svo-node-content">
              {predicate ? (
                <strong className="svo-node-text">"{predicate}"</strong>
              ) : (
                <span className="svo-unresolved-text">Not confidently identified</span>
              )}
            </div>
            <div className="svo-node-footer">Action, state, or relation asserted</div>
          </div>

          {/* Connection Vector 2 */}
          <div className="svo-vector-connector">
            <div className="svo-vector-line" />
            <div className="svo-vector-arrow">&rarr;</div>
            <span className="svo-vector-label">targets</span>
          </div>

          {/* Object Node */}
          <div className={`svo-node svo-node-object ${!object ? 'svo-node-empty' : ''}`}>
            <div className="svo-node-header">
              <span className="svo-badge badge-object">OBJECT (ARG1)</span>
              <span className="svo-role-type">Complement / Theme</span>
            </div>
            <div className="svo-node-content">
              {object ? (
                <strong className="svo-node-text">"{object}"</strong>
              ) : (
                <span className="svo-intransitive-text">Not confidently identified</span>
              )}
            </div>
            <div className="svo-node-footer">Entity affected or predicated upon</div>
          </div>
        </div>

        {isIncomplete && (
          <div className="svo-incomplete-callout">
            <Icon name="info" size={15} />
            <span>
              <strong>Syntactic Note:</strong> Sentence does not conform to standard Subject-Verb-Object argument structure. The system treats it with conservative linguistic bounds rather than forcing an artificial relation.
            </span>
          </div>
        )}
      </div>

      {/* Clause Boundaries Breakdown */}
      {clauses.length > 0 && (
        <div className="clauses-breakdown-section">
          <h4 className="clauses-title">IDENTIFIED CLAUSE SEGMENTS ({clauses.length})</h4>
          <div className="clauses-list">
            {clauses.map((cl, i) => (
              <div key={i} className="clause-item-card">
                <div className="clause-item-top">
                  <span className="clause-index">CLAUSE #{i + 1}</span>
                  <span className="clause-type-pill">{cl.clause_type || 'independent'}</span>
                  {cl.is_negated && (
                    <span className="clause-negation-pill">
                      <Icon name="xCircle" size={12} /> Negated ({cl.negation_words?.join(', ') || 'not'})
                    </span>
                  )}
                  {cl.modality && cl.modality !== 'certain' && (
                    <span className="clause-modality-pill">Modality: {cl.modality}</span>
                  )}
                </div>
                <div className="clause-text">"{cl.text}"</div>
                <div className="clause-svo-summary">
                  <span><strong>S:</strong> {cl.subject || '—'}</span> &bull;{' '}
                  <span><strong>P:</strong> {cl.predicate || '—'}</span> &bull;{' '}
                  <span><strong>O:</strong> {cl.object || '—'}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default SyntaxRelationshipView;
