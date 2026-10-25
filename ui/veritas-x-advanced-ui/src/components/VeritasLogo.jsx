import React from 'react';

/**
 * VERITAS-X — Brand Identity & Minimal Geometric Mark
 *
 * Distinctive geometric emblem:
 * Interlocking precision verification chevron (V) and cross-examination vector (X)
 * rendered in refined indigo (#4F46E5) with crisp slate (#0F172A).
 */
export function VeritasSymbol({
  size = 32,
  className = '',
  theme = 'indigo', // 'indigo' | 'monochrome'
  ariaLabel = 'VERITAS-X Logo Mark',
}) {
  const isMono = theme === 'monochrome';
  const primaryIndigo = isMono ? 'currentColor' : '#4F46E5';
  const lightIndigo = isMono ? 'currentColor' : '#EEF2FF';
  const accentIndigo = isMono ? 'currentColor' : '#A5B4FC';

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 36 36"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={`veritas-symbol ${className}`}
      role="img"
      aria-label={ariaLabel}
    >
      {/* Refined rounded base in signature indigo */}
      <rect width="36" height="36" rx="9" fill={primaryIndigo} />

      {/* Downward verification chevron (V) */}
      <path
        d="M9.5 11.5L18 24.5L26.5 11.5"
        stroke={lightIndigo}
        strokeWidth="2.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Crossing analytical vector creating the iconic V-X intersection */}
      <path
        d="M25 24.5L18 14.5L11 24.5"
        stroke={accentIndigo}
        strokeWidth="2.25"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Center verification nexus focal node */}
      <circle cx="18" cy="19.5" r="1.8" fill="#FFFFFF" />
    </svg>
  );
}

/**
 * Wordmark Lockup: Clean, modern typography paired with the minimal geometric symbol.
 */
export function VeritasWordmark({
  size = 'md', // 'sm' | 'md' | 'lg'
  tagline = 'Explainable AI Fact Verification',
  showTagline = true,
  theme = 'indigo',
  className = '',
}) {
  const primaryText = '#0F172A';
  const indigoAccent = '#4F46E5';
  const tagColor = '#64748B';

  const scaleMap = {
    sm: { symbol: 26, titleSize: '1.05rem', tagSize: '0.65rem', gap: '0.6rem' },
    md: { symbol: 32, titleSize: '1.25rem', tagSize: '0.70rem', gap: '0.7rem' },
    lg: { symbol: 42, titleSize: '1.65rem', tagSize: '0.78rem', gap: '0.85rem' },
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
      <VeritasSymbol size={currentScale.symbol} theme={theme} />
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
              fontWeight: 500,
              fontSize: currentScale.tagSize,
              letterSpacing: '0.01em',
              color: tagColor,
              marginTop: '0.2rem',
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
