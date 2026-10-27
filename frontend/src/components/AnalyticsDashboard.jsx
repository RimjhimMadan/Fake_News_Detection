import React, { useEffect, useState } from 'react';
import { Activity, BarChart3, Database, LoaderCircle } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { readJsonResponse } from '../api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

export default function AnalyticsDashboard() {
  const [analytics, setAnalytics] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE_URL}/analytics`)
      .then(readJsonResponse)
      .then((data) => { if (active) setAnalytics(data); })
      .catch((err) => { if (active) setError(err.message || 'Analytics could not be loaded.'); });
    return () => { active = false; };
  }, []);

  if (error) return <div className="glass-card form-error">{error}</div>;
  if (!analytics) return <div className="glass-card dashboard-loading"><LoaderCircle className="spin-icon" size={28} /><span>Loading analytics...</span></div>;

  return <section className="analytics-dashboard">
    <div className="dashboard-heading">
      <div><h2>Prediction Analytics</h2><p className="panel-description">Aggregate activity from this deployment. Individual article text is stored locally for keyword analysis.</p></div>
      <Activity size={28} color="var(--accent-primary)" />
    </div>
    <div className="stat-grid">
      <div className="glass-card stat-card"><Database size={20} color="var(--accent-primary)" /><strong>{analytics.total_checks}</strong><span>Total checks</span></div>
      <div className="glass-card stat-card"><BarChart3 size={20} color="var(--fake-color)" /><strong>{analytics.fake_count}</strong><span>Fake predictions</span></div>
      <div className="glass-card stat-card"><Activity size={20} color="var(--real-color)" /><strong>{analytics.real_count}</strong><span>Real predictions</span></div>
    </div>
    <div className="analytics-chart-grid">
      <div className="glass-card chart-card"><h3>Checks over time</h3><ResponsiveContainer width="100%" height={260}><LineChart data={analytics.over_time}><CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" /><XAxis dataKey="date" stroke="#94a3b8" /><YAxis allowDecimals={false} stroke="#94a3b8" /><Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.12)' }} /><Line type="monotone" dataKey="real" stroke="#10b981" strokeWidth={2} /><Line type="monotone" dataKey="fake" stroke="#f43f5e" strokeWidth={2} /></LineChart></ResponsiveContainer></div>
      <div className="glass-card chart-card"><h3>Most flagged keywords</h3><ResponsiveContainer width="100%" height={260}><BarChart data={analytics.top_keywords} layout="vertical" margin={{ left: 18, right: 12 }}><CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" /><XAxis type="number" allowDecimals={false} stroke="#94a3b8" /><YAxis dataKey="keyword" type="category" width={80} stroke="#94a3b8" /><Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.12)' }} /><Bar dataKey="count" fill="#818cf8" radius={[0, 4, 4, 0]} /></BarChart></ResponsiveContainer></div>
    </div>
    <div className="glass-card chart-card"><h3>Model usage</h3><div className="model-usage-list">{analytics.model_usage.length === 0 ? <span className="panel-description">No model activity yet.</span> : analytics.model_usage.map((item) => <div className="model-usage-row" key={item.model}><span>{item.model}</span><strong>{item.count}</strong></div>)}</div></div>
  </section>;
}
