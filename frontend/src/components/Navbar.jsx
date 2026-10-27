import React from 'react';
import { ShieldCheck, BarChart2, Github, Moon, Sun } from 'lucide-react';

export default function Navbar({ onOpenMetrics, isDarkMode, onToggleTheme }) {
  return (
    <header className="navbar">
      <div className="nav-brand">
        <div className="brand-icon">
          <ShieldCheck size={22} color="#ffffff" />
        </div>
        <div>
          <span className="brand-title">TruthLens</span>
        </div>
        <span className="nav-tag">AI News Verification</span>
      </div>

      <div className="nav-actions">
        <button className="btn-secondary" onClick={onOpenMetrics} id="view-metrics-btn" aria-label="Open model benchmark">
          <BarChart2 size={16} />
          <span>Model Benchmark</span>
        </button>
        <button
          className="btn-secondary theme-toggle"
          onClick={onToggleTheme}
          aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          title={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {isDarkMode ? <Sun size={16} /> : <Moon size={16} />}
        </button>
        <a
          href="https://github.com"
          target="_blank"
          rel="noreferrer"
          className="btn-secondary"
          title="View Source on GitHub"
          aria-label="View source on GitHub"
        >
          <Github size={16} />
        </a>
      </div>
    </header>
  );
}
