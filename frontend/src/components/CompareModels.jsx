import React, { useState } from 'react';
import { GitCompareArrows, LoaderCircle } from 'lucide-react';
import { readJsonResponse } from '../api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

export default function CompareModels({ text, setText }) {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const compare = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/predict-compare`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text })
      });
      const data = await readJsonResponse(response);
      setPredictions(data.predictions || []);
    } catch (err) {
      setError(err.message || 'Model comparison failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="compare-panel">
      <div className="glass-card compare-input-card">
        <div className="input-title"><GitCompareArrows size={18} color="#6366f1" /><span>Compare Models</span></div>
        <p className="panel-description">Run the same article through Logistic Regression, Linear SVM, and DistilBERT.</p>
        <textarea className="news-textarea compare-textarea" value={text} onChange={(event) => setText(event.target.value)} placeholder="Paste at least three words of news text..." />
        <button className="btn-primary" onClick={compare} disabled={loading || text.trim().split(/\s+/).length < 3}>
          {loading ? <><LoaderCircle className="spin-icon" size={17} /><span>Comparing...</span></> : <><GitCompareArrows size={17} /><span>Compare Predictions</span></>}
        </button>
        {error && <p className="form-error">{error}</p>}
      </div>
      {predictions.length > 0 && <div className="compare-grid">{predictions.map((prediction) => {
        const isReal = prediction.label === 'Real';
        return <article className={`glass-card model-result ${isReal ? 'result-card-real' : 'result-card-fake'}`} key={prediction.model_name}>
          <span className="model-name">{prediction.model_name}</span>
          <strong className={isReal ? 'real-text' : 'fake-text'}>{prediction.label}</strong>
          <span className="model-confidence">{prediction.confidence_score}% confidence</span>
          <div className="mini-probabilities"><span>Real {Math.round(prediction.probabilities.Real * 100)}%</span><span>Fake {Math.round(prediction.probabilities.Fake * 100)}%</span></div>
        </article>;
      })}</div>}
    </section>
  );
}