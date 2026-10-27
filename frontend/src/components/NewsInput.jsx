import React from 'react';
import { Newspaper, Sparkles, Trash2, ArrowRight } from 'lucide-react';

const SAMPLES = [
  {
    label: '📰 Real News Sample',
    text: 'WASHINGTON (Reuters) - The U.S. Federal Reserve decided on Wednesday to keep benchmark interest rates steady, citing solid economic expansion and historically low unemployment levels nationwide, according to statements released following the FOMC meeting.'
  },
  {
    label: '🚨 Fake News Sample',
    text: 'SHOCKING LEAK: Secret whistleblower documents prove global cabal fabricated weather events to manipulate financial markets! Mainstream corporate media caught in massive coverup of explosive hidden files.'
  },
  {
    label: '🏛️ Political News Sample',
    text: 'LONDON (Reuters) - British lawmakers voted on Tuesday to approve new environmental guidelines for commercial aviation, aiming to curb emissions and accelerate adoption of sustainable aviation fuels by 2030.'
  }
];

export default function NewsInput({ text, setText, sourceUrl, setSourceUrl, onAnalyze, loading }) {
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const charCount = text.length;
  const isValid = wordCount >= 3 && charCount >= 10;

  return (
    <div className="glass-card">
      <div className="input-header">
        <div className="input-title">
          <Newspaper size={18} color="#6366f1" />
          <span>Analyze News Article or Headline</span>
        </div>
        {text && (
          <button
            className="btn-secondary"
            onClick={() => setText('')}
            style={{ padding: '0.3rem 0.6rem', fontSize: '0.78rem' }}
            title="Clear Text"
          >
            <Trash2 size={13} />
            <span>Clear</span>
          </button>
        )}
      </div>

      <div className="samples-row">
        <span style={{ fontSize: '0.8rem', color: 'var(--text-subtle)', alignSelf: 'center' }}>
          Try samples:
        </span>
        {SAMPLES.map((sample, idx) => (
          <button
            key={idx}
            type="button"
            className="sample-chip"
            onClick={() => setText(sample.text)}
          >
            {sample.label}
          </button>
        ))}
      </div>

      <textarea
        id="news-input-textarea"
        className="news-textarea"
        placeholder="Paste a news headline, statement, or full article text here to evaluate its authenticity..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={loading}
      />

      <input
        id="source-url-input"
        className="source-url-input"
        type="url"
        placeholder="Optional source URL for a credibility signal (e.g. https://reuters.com/article)"
        value={sourceUrl}
        onChange={(e) => setSourceUrl(e.target.value)}
        disabled={loading}
        aria-label="Optional source URL"
      />

      <div className="input-footer">
        <div className="counter-text">
          <span>{wordCount} words</span> • <span>{charCount} characters</span>
          {text && !isValid && (
            <span style={{ color: '#f87171', marginLeft: '0.5rem' }}>
              (Minimum 3 words & 10 characters required)
            </span>
          )}
        </div>

        <button
          id="check-news-btn"
          className="btn-primary"
          onClick={onAnalyze}
          disabled={!isValid || loading}
        >
          {loading ? (
            <>
              <div className="spinner" />
              <span>Analyzing authenticity...</span>
            </>
          ) : (
            <>
              <Sparkles size={17} />
              <span>Check News</span>
              <ArrowRight size={15} />
            </>
          )}
        </button>
      </div>
    </div>
  );
}
