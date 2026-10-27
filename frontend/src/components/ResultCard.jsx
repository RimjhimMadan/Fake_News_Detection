import React from 'react';
import { CheckCircle2, AlertTriangle, HelpCircle, Activity } from 'lucide-react';
import ExplainabilityView from './ExplainabilityView';
import ShareResultButton from './ShareResultButton';

export default function ResultCard({ result, loading, error }) {
  if (loading) {
    return (
      <div className="glass-card" style={{ textAlign: 'center', padding: '3.5rem 1.5rem' }}>
        <div className="spinner" style={{ width: '36px', height: '36px', margin: '0 auto 1.25rem', borderWidth: '3px' }} />
        <h3 style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>Evaluating Authenticity...</h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          Extracting TF-IDF features and generating LIME local surrogate explanations.
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-card" style={{ borderColor: 'var(--fake-border)', background: 'var(--fake-bg)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: 'var(--fake-color)', marginBottom: '0.5rem' }}>
          <AlertTriangle size={20} />
          <h3 style={{ fontSize: '1.1rem' }}>Analysis Error</h3>
        </div>
        <p style={{ color: 'var(--text-main)', fontSize: '0.9rem' }}>{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="glass-card empty-result-placeholder">
        <Activity size={40} className="empty-icon" />
        <h3 style={{ fontSize: '1.15rem', color: 'var(--text-main)', marginBottom: '0.4rem' }}>
          No Article Analyzed Yet
        </h3>
        <p style={{ fontSize: '0.85rem', maxWidth: '320px' }}>
          Paste a news headline or article in the box on the left and click <strong>"Check News"</strong> to inspect its authenticity.
        </p>
      </div>
    );
  }

  const isReal = result.label === 'Real';
  const confidence = result.confidence_score;
  const cardClass = isReal ? 'result-card-real' : 'result-card-fake';
  const badgeClass = isReal ? 'badge-real' : 'badge-fake';

  return (
    <div className={`glass-card ${cardClass} result-card-container`}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Authenticity Prediction
        </span>
        <ShareResultButton result={result} />
      </div>

      {result.credibility && <div className={`credibility-badge credibility-${result.credibility.rating.toLowerCase()}`}>
        <span>Source credibility: <strong>{result.credibility.rating}</strong></span>
        <span>{result.credibility.source_name || result.credibility.domain}</span>
      </div>}

      {/* Main Badge */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '0.25rem' }}>
        <div className={`badge-status ${badgeClass}`} id="result-label-badge">
          {isReal ? <CheckCircle2 size={22} /> : <AlertTriangle size={22} />}
          <span>{result.label} News</span>
        </div>
        <div style={{ textAlign: 'right' }}>
          <span style={{ fontSize: '1.75rem', fontWeight: 800, color: isReal ? 'var(--real-color)' : 'var(--fake-color)' }}>
            {confidence}%
          </span>
          <span style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Confidence
          </span>
        </div>
      </div>

      {/* Confidence Gauge */}
      <div className="confidence-gauge-container">
        <div className="gauge-header">
          <span>Real Probability ({Math.round(result.probabilities?.Real * 100 || 0)}%)</span>
          <span>Fake Probability ({Math.round(result.probabilities?.Fake * 100 || 0)}%)</span>
        </div>
        <div className="gauge-bar-bg">
          <div
            className={isReal ? 'gauge-bar-fill-real' : 'gauge-bar-fill-fake'}
            style={{ width: `${confidence}%` }}
          />
        </div>
      </div>

      {/* Explainability Breakdown */}
      <ExplainabilityView words={result.top_influential_words} />
    </div>
  );
}
