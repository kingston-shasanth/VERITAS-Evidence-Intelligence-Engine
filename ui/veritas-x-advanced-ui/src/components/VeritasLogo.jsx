import React from 'react';

/**
 * VERITAS-X — Brand Identity & Minimal Geometric Mark
 *
 * Distinctive geometric emblem:
 * Interlocking precision verification chevron (V) and analytical cross-examination vector (X)
 * anchored on documentary evidence planes, rendered in signature indigo (#4F46E5) with cyan accents (#38BDF8).
 */
export function VeritasSymbol({
  size = 32,
  className = '',
  theme = 'indigo', // 'indigo' | 'monochrome'
  ariaLabel = 'VERITAS-X Logo Mark',
}) {
  const isMono = theme === 'monochrome';
  const primaryIndigo = isMono ? 'currentColor' : '#4F46E5';
  const darkIndigo = isMono ? 'currentColor' : '#312E81';
  const borderIndigo = isMono ? 'currentColor' : '#6366F1';
  const docLineColor = isMono ? 'currentColor' : '#C7D2FE';
  const cyanAccent = isMono ? 'currentColor' : '#38BDF8';
  const whiteStroke = isMono ? 'currentColor' : '#FFFFFF';
  const nexusDot = isMono ? 'currentColor' : '#0284C7';

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={`veritas-symbol ${className}`}
      role="img"
      aria-label={ariaLabel}
    >
      <defs>
        {!isMono && (
          <linearGradient id="symBgGrad" x1="4" y1="4" x2="60" y2="60" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor={primaryIndigo} />
            <stop offset="100%" stopColor={darkIndigo} />
          </linearGradient>
        )}
      </defs>

      {/* Signature Squircle Base */}
      <rect
        x="2"
        y="2"
        width="60"
        height="60"
        rx="15"
        fill={isMono ? 'none' : 'url(#symBgGrad)'}
        stroke={borderIndigo}
        strokeWidth={isMono ? 3 : 1.2}
        strokeOpacity={isMono ? 1 : 0.5}
      />

      {/* Documentary Evidence Guidelines */}
      <line x1="16" y1="16" x2="34" y2="16" stroke={docLineColor} strokeWidth="2.2" strokeLinecap="round" opacity="0.55" />
      <line x1="39" y1="16" x2="48" y2="16" stroke={docLineColor} strokeWidth="2.2" strokeLinecap="round" opacity="0.35" />
      <line x1="16" y1="22" x2="26" y2="22" stroke={docLineColor} strokeWidth="2" strokeLinecap="round" opacity="0.4" />

      {/* Analytical Cross-Examination Vector (X) */}
      <path
        d="M19 44 L45 22"
        stroke={cyanAccent}
        strokeWidth="3.2"
        strokeLinecap="round"
      />

      {/* Primary Verification Chevron (V) */}
      <path
        d="M17 28 L29.5 45 L47 18"
        stroke={whiteStroke}
        strokeWidth="4.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Precision Evidentiary Nexus Node */}
      <circle cx="32" cy="33.5" r="3.2" fill={whiteStroke} />
      <circle cx="32" cy="33.5" r="1.6" fill={nexusDot} />
    </svg>
  );
}

/**
 * Wordmark Lockup: Clean, modern typography paired with the minimal geometric symbol.
 */
export function VeritasWordmark({
  size = 'md', // 'sm' | 'md' | 'lg'
  tagline = 'Evidence-Grounded Claim Verification',
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
