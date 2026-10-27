import React, { useRef, useState } from 'react';
import { Download, FileUp, LoaderCircle } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

export default function BatchUpload() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [downloadUrl, setDownloadUrl] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) return;
    setLoading(true);
    setError(null);
    setDownloadUrl(null);
    try {
      const body = new FormData();
      body.append('file', file);
      const response = await fetch(`${API_BASE_URL}/predict-batch`, { method: 'POST', body });
      if (!response.ok) {
        const details = await response.json().catch(() => ({}));
        throw new Error(details.detail || `Server returned error ${response.status}`);
      }
      const blob = await response.blob();
      setDownloadUrl(URL.createObjectURL(blob));
    } catch (err) {
      setError(err.message || 'Batch prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="glass-card batch-panel">
      <div className="input-title"><FileUp size={18} color="#6366f1" /><span>Batch Upload</span></div>
      <p className="panel-description">Upload a UTF-8 CSV with a <strong>text</strong> column. Results include the original rows plus labels and confidence scores.</p>
      <form onSubmit={handleSubmit}>
        <input ref={inputRef} type="file" accept=".csv,text/csv" onChange={(event) => setFile(event.target.files?.[0] || null)} />
        <div className="batch-actions">
          <button type="submit" className="btn-primary" disabled={!file || loading}>
            {loading ? <><LoaderCircle className="spin-icon" size={17} /><span>Processing CSV...</span></> : <><FileUp size={17} /><span>Analyze CSV</span></>}
          </button>
          {downloadUrl && <a className="btn-secondary" href={downloadUrl} download="fakefound-predictions.csv"><Download size={16} /><span>Download Results</span></a>}
        </div>
      </form>
      {file && <p className="file-name">Selected: {file.name}</p>}
      {error && <p className="form-error">{error}</p>}
    </section>
  );
}