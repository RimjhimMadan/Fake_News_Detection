import React, { useEffect, useState } from 'react';
import Navbar from './components/Navbar';
import NewsInput from './components/NewsInput';
import ResultCard from './components/ResultCard';
import MetricsModal from './components/MetricsModal';
import BatchUpload from './components/BatchUpload';
import CompareModels from './components/CompareModels';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import { readJsonResponse } from './api';
import { ShieldCheck, Zap, Sparkles } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

export default function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showMetrics, setShowMetrics] = useState(false);
  const [sourceUrl, setSourceUrl] = useState('');
  const [activeView, setActiveView] = useState('analyze');
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const savedTheme = window.localStorage.getItem('fakefound-theme');
    return savedTheme ? savedTheme === 'dark' : true;
  });

  useEffect(() => {
    document.documentElement.dataset.theme = isDarkMode ? 'dark' : 'light';
    window.localStorage.setItem('fakefound-theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  const handleAnalyze = async () => {
    if (!text.trim() || text.trim().length < 10) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source_url: sourceUrl || null, num_explanation_words: 8 })
      });

      const data = await readJsonResponse(response);
      setResult(data);
    } catch (err) {
      console.error('Analysis failed:', err);
      setError(err.message || 'Failed to connect to FakeFound prediction backend.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar
        onOpenMetrics={() => setShowMetrics(true)}
        isDarkMode={isDarkMode}
        onToggleTheme={() => setIsDarkMode((current) => !current)}
      />

      <main className="app-container">
        <nav className="view-tabs" aria-label="Analysis views">
          {[
            ['analyze', 'Analyze Article'],
            ['batch', 'Batch Upload'],
            ['compare', 'Compare Models'],
            ['analytics', 'Analytics'],
          ].map(([view, label]) => (
            <button
              key={view}
              className={activeView === view ? 'view-tab active' : 'view-tab'}
              onClick={() => setActiveView(view)}
              role="tab"
              aria-selected={activeView === view}
              tabIndex={activeView === view ? 0 : -1}
            >
              {label}
            </button>
          ))}
        </nav>

        {activeView === 'batch' && <BatchUpload />}
        {activeView === 'compare' && <CompareModels text={text} setText={setText} />}
        {activeView === 'analytics' && <AnalyticsDashboard />}

        {activeView === 'analyze' && <>
        {/* Hero Section */}
        <section className="hero-section">
          <h1 className="hero-title">
            Intelligent Fake News Detection with Explainable AI
          </h1>
          <p className="hero-subtitle">
            Analyze headlines and news stories in real-time. Uncover algorithmic confidence scores and token-level LIME attributions to distinguish factual journalism from fabricated misinformation.
          </p>
        </section>

        {/* Main Grid */}
        <div className="main-grid">
          <NewsInput
            text={text}
            setText={setText}
            sourceUrl={sourceUrl}
            setSourceUrl={setSourceUrl}
            onAnalyze={handleAnalyze}
            loading={loading}
          />
          <ResultCard
            result={result}
            loading={loading}
            error={error}
          />
        </div>
        </>}

        {/* Feature Badges Footer */}
        <footer style={{ marginTop: '2rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', color: 'var(--text-subtle)', fontSize: '0.82rem' }}>
          <div style={{ display: 'flex', gap: '1.5rem' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <ShieldCheck size={14} color="#10b981" /> 99.47% Benchmark Accuracy
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Zap size={14} color="#6366f1" /> Sub-50ms Inference Latency
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Sparkles size={14} color="#a855f7" /> LIME Token Attribution
            </span>
          </div>
          <div>
            <span>FakeFound NLP System • v1.0.0</span>
          </div>
        </footer>
      </main>

      {/* Metrics Modal */}
      <MetricsModal
        isOpen={showMetrics}
        onClose={() => setShowMetrics(false)}
      />
    </div>
  );
}
