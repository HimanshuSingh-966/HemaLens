import { useEffect, useState } from 'react';

export default function ConfidenceRing({ confidence }) {
  const [offset, setOffset] = useState(263.9);
  
  useEffect(() => {
    // Animate ring fill after mount
    setTimeout(() => {
      setOffset(263.9 - (263.9 * (confidence || 0.5)));
    }, 100);
  }, [confidence]);

  const pct = Math.round((confidence || 0.5) * 100);

  return (
    <div style={{ width: '120px', height: '120px', position: 'relative', flexShrink: 0 }}>
      <svg width="120" height="120" viewBox="0 0 100 100" style={{ transform: 'rotate(-90deg)' }}>
        <circle cx="50" cy="50" r="42" fill="none" stroke="var(--border)" strokeWidth="6" />
        <circle 
          cx="50" cy="50" r="42" 
          fill="none" 
          stroke="url(#ringGrad)" 
          strokeWidth="6" 
          strokeLinecap="round" 
          strokeDasharray="263.9" 
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 1.5s cubic-bezier(0.4, 0, 0.2, 1)' }}
        />
        <defs>
          <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#0048f9" />
            <stop offset="100%" stopColor="#6b9fff" />
          </linearGradient>
        </defs>
      </svg>
      <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <div className="display-font" style={{ fontSize: '26px', fontWeight: 800, color: '#000', letterSpacing: '-0.02em', lineHeight: 1 }}>{pct}%</div>
        <div className="mono" style={{ fontSize: '9px', color: 'var(--text-muted)', letterSpacing: '1px', marginTop: '4px' }}>CONFIDENCE</div>
      </div>
    </div>
  );
}
