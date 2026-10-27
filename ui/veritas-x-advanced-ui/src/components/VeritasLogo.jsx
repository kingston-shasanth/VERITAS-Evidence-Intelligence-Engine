import React from 'react';

/**
 * VERITAS-X — Official Project Identity & Branding Component
 *
 * Employs the official veritas-x-logo.png visual mark paired with
 * precision typography for evidence-grounded claim verification.
 */
export function VeritasSymbol({
  size = 32,
  className = '',
  ariaLabel = 'VERITAS-X Official Logo',
}) {
  return (
    <img
      src="/veritas-x-logo.png"
      width={size}
      height={size}
      alt={ariaLabel}
      className={`veritas-symbol ${className}`}
      style={{
        width: size,
        height: size,
        borderRadius: Math.round(size * 0.22),
        objectFit: 'cover',
        display: 'inline-block',
        verticalAlign: 'middle',
        flexShrink: 0,
        boxShadow: '0 1px 3px rgba(15, 23, 42, 0.12)',
      }}
    />
  );
}

/**
 * Wordmark Lockup: Clean, modern typography paired with the official project logo.
 */
export function VeritasWordmark({
  size = 'md', // 'sm' | 'md' | 'lg'
  tagline = 'Evidence-Grounded Claim Verification',
  showTagline = true,
  className = '',
}) {
  const primaryText = '#0F172A';
  const indigoAccent = '#4F46E5';
  const tagColor = '#64748B';

  const scaleMap = {
    sm: { symbol: 28, titleSize: '1.05rem', tagSize: '0.65rem', gap: '0.65rem' },
    md: { symbol: 34, titleSize: '1.25rem', tagSize: '0.70rem', gap: '0.75rem' },
    lg: { symbol: 44, titleSize: '1.65rem', tagSize: '0.78rem', gap: '0.90rem' },
  };

  const currentScale = scaleMap[size] || scaleMap.md;

  return (
    <div
      className={`veritas-wordmark-lockup ${className}`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: currentScale.gap,
        userSelect: 'none',
      }}
    >
      <VeritasSymbol size={currentScale.symbol} />
      <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.15 }}>
        <div
          style={{
            fontFamily: 'var(--font-sans, "Plus Jakarta Sans", sans-serif)',
            fontWeight: 800,
            fontSize: currentScale.titleSize,
            letterSpacing: '0.02em',
            color: primaryText,
            display: 'flex',
            alignItems: 'center',
          }}
        >
          <span>VERITAS</span>
          <span style={{ color: indigoAccent, marginLeft: '0.08em', fontWeight: 900 }}>-X</span>
        </div>
        {showTagline && (
          <div
            style={{
              fontFamily: 'var(--font-body, "Inter", sans-serif)',
              fontWeight: 600,
              fontSize: currentScale.tagSize,
              letterSpacing: '0.04em',
              color: tagColor,
              marginTop: '0.2rem',
              textTransform: 'uppercase',
            }}
          >
            {tagline}
          </div>
        )}
      </div>
    </div>
  );
}

export default VeritasWordmark;
