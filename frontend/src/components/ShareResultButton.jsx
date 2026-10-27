import React, { useState } from 'react';
import { Download, Share2 } from 'lucide-react';

export default function ShareResultButton({ result }) {
  const [sharing, setSharing] = useState(false);

  const downloadCard = async () => {
    setSharing(true);
    try {
      const canvas = document.createElement('canvas');
      canvas.width = 1200;
      canvas.height = 700;
      const context = canvas.getContext('2d');
      context.fillStyle = '#0f172a';
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = result.label === 'Real' ? '#10b981' : '#f43f5e';
      context.fillRect(0, 0, 18, canvas.height);
      context.fillStyle = '#f8fafc';
      context.font = '700 56px sans-serif';
      context.fillText('FakeFound', 80, 110);
      context.font = '400 28px sans-serif';
      context.fillStyle = '#94a3b8';
      context.fillText('Explainable news authenticity signal', 82, 155);
      context.font = '700 72px sans-serif';
      context.fillStyle = result.label === 'Real' ? '#10b981' : '#f43f5e';
      context.fillText(`${result.label} News`, 82, 290);
      context.font = '500 36px sans-serif';
      context.fillStyle = '#f8fafc';
      context.fillText(`${result.confidence_score}% confidence`, 84, 360);
      if (result.credibility) {
        context.fillStyle = '#cbd5e1';
        context.fillText(`Source: ${result.credibility.rating}`, 84, 430);
        context.font = '400 26px sans-serif';
        context.fillStyle = '#94a3b8';
        context.fillText(result.credibility.domain, 84, 475);
      }
      context.font = '400 22px sans-serif';
      context.fillStyle = '#64748b';
      context.fillText('Assistive signal, not a substitute for independent verification.', 84, 620);
      const link = document.createElement('a');
      link.download = 'fakefound-result.png';
      link.href = canvas.toDataURL('image/png');
      link.click();
    } finally {
      setSharing(false);
    }
  };

  return <button className="btn-secondary share-button" onClick={downloadCard} disabled={sharing} title="Download a shareable result card">
    {sharing ? <Download size={16} /> : <Share2 size={16} />}
    <span>{sharing ? 'Preparing...' : 'Share Result'}</span>
  </button>;
}
