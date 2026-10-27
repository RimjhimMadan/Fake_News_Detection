import React from 'react';
import { Eye, HelpCircle } from 'lucide-react';

export default function ExplainabilityView({ words = [] }) {
  if (!words || words.length === 0) return null;

  const realWords = words.filter((w) => w.supports === 'Real');
  const fakeWords = words.filter((w) => w.supports === 'Fake');

  return (
    <div className="explain-section">
      <div className="explain-title">
        <Eye size={16} color="#818cf8" />
        <span>Top Influential Words (LIME Explanation)</span>
      </div>

      <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
        Token-level attribution showing words that pushed the prediction toward{' '}
        <strong style={{ color: 'var(--real-color)' }}>Real</strong> or{' '}
        <strong style={{ color: 'var(--fake-color)' }}>Fake</strong>:
      </p>

      {/* Badge Cloud */}
      <div className="word-badges-cloud">
        {words.map((item, idx) => {
          const isReal = item.supports === 'Real';
          return (
            <div
              key={idx}
              className={`word-badge ${isReal ? 'word-badge-real' : 'word-badge-fake'}`}
              title={`Pushes toward ${item.supports} (Weight: ${item.weight})`}
            >
              <span>{item.word}</span>
              <span className="word-weight">
                {item.weight > 0 ? `+${item.weight}` : item.weight}
              </span>
            </div>
          );
        })}
      </div>

      {/* Feature Breakdown Table */}
      <div className="word-breakdown-list">
        {words.slice(0, 6).map((item, idx) => {
          const isReal = item.supports === 'Real';
          const pct = Math.min(100, Math.round(item.importance * 300));
          return (
            <div key={idx} className="word-item-row">
              <span className="word-item-name">"{item.word}"</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', width: '55%' }}>
                <div style={{ flex: 1, height: '6px', background: 'rgba(255,255,255,0.06)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${pct}%`,
                      background: isReal ? 'var(--real-color)' : 'var(--fake-color)',
                      borderRadius: '4px'
                    }}
                  />
                </div>
                <span
                  style={{
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    color: isReal ? 'var(--real-color)' : 'var(--fake-color)',
                    minWidth: '55px',
                    textAlign: 'right'
                  }}
                >
                  {item.supports}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
