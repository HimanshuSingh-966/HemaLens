import { useState } from 'react';
import { UploadCloud, User, ActivitySquare } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function AnalysisInput({ onAnalyzeFile, onAnalyzeText, onAnalyzeManual, isLoading, error }) {
  const [gender, setGender] = useState('male');
  const [tab, setTab] = useState('upload');
  
  const [file, setFile] = useState(null);
  const [text, setText] = useState('');
  
  const PARAM_LIST = [
    'Hemoglobin', 'RBC', 'WBC', 'Platelets', 'Hematocrit', 'MCV', 'MCH', 
    'MCHC', 'Neutrophils', 'Lymphocytes', 'Glucose', 'Creatinine'
  ];
  const [manualParams, setManualParams] = useState({});

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleManualChange = (p, val) => {
    if (val === '') {
      const newParams = { ...manualParams };
      delete newParams[p];
      setManualParams(newParams);
    } else {
      setManualParams(prev => ({ ...prev, [p]: parseFloat(val) }));
    }
  };

  const contentVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.3 } },
    exit: { opacity: 0, y: -10, transition: { duration: 0.2 } }
  };

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      
      {/* Header */}
      <div className="flex items-center gap-4" style={{ padding: '24px 32px', borderBottom: '1px solid var(--border)', background: 'var(--bg-card-soft)' }}>
        <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(0, 72, 249, 0.08)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <ActivitySquare size={24} color="var(--accent)" />
        </div>
        <div>
          <div className="display-font" style={{ fontWeight: 700, fontSize: '20px', color: '#000', letterSpacing: '-0.02em' }}>Data Ingestion</div>
          <div className="mono" style={{ fontSize: '11px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginTop: '4px' }}>Input target metrics</div>
        </div>
      </div>

      <div style={{ padding: '32px', flex: 1, display: 'flex', flexDirection: 'column' }}>
        
        {/* Gender Toggle using Pill Switch */}
        <div style={{ marginBottom: '32px' }}>
          <label className="mono" style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '12px', letterSpacing: '0.05em' }}>PATIENT DEMOGRAPHIC</label>
          <div className="flex" style={{ background: 'rgba(0,0,0,0.03)', borderRadius: '12px', padding: '6px', border: '1px solid var(--border)' }}>
            <button 
              onClick={() => setGender('male')}
              style={{ flex: 1, padding: '12px', borderRadius: '8px', background: gender === 'male' ? 'var(--bg-card)' : 'transparent', border: `1px solid ${gender === 'male' ? 'var(--border)' : 'transparent'}`, color: gender === 'male' ? '#000' : 'var(--text-secondary)', transition: 'all 0.3s ease', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', fontSize: '14px', fontWeight: 600, boxShadow: gender === 'male' ? '0 2px 10px rgba(0,0,0,0.05)' : 'none' }}
            >
              <User size={16} color={gender === 'male' ? 'var(--accent)' : 'currentColor'} /> Male
            </button>
            <button 
              onClick={() => setGender('female')}
              style={{ flex: 1, padding: '12px', borderRadius: '8px', background: gender === 'female' ? 'var(--bg-card)' : 'transparent', border: `1px solid ${gender === 'female' ? 'var(--border)' : 'transparent'}`, color: gender === 'female' ? '#000' : 'var(--text-secondary)', transition: 'all 0.3s ease', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', fontSize: '14px', fontWeight: 600, boxShadow: gender === 'female' ? '0 2px 10px rgba(0,0,0,0.05)' : 'none' }}
            >
              <User size={16} color={gender === 'female' ? 'var(--accent)' : 'currentColor'} /> Female
            </button>
          </div>
        </div>

        {/* Input Method Tabs */}
        <div style={{ marginBottom: '32px' }}>
          <div className="flex gap-4" style={{ borderBottom: '1px solid var(--border)', paddingBottom: '2px' }}>
            {['upload', 'paste', 'manual'].map(t => (
              <button
                key={t}
                onClick={() => setTab(t)}
                style={{ padding: '12px 0', border: 'none', background: 'transparent', color: tab === t ? '#000' : 'var(--text-muted)', fontSize: '14px', fontWeight: tab === t ? 600 : 500, position: 'relative', transition: 'color 0.3s', textTransform: 'capitalize' }}
              >
                {t}
                {tab === t && (
                  <motion.div 
                    layoutId="tab-indicator"
                    style={{ position: 'absolute', bottom: -2, left: 0, width: '100%', height: '2px', background: 'var(--accent)' }} 
                  />
                )}
              </button>
            ))}
          </div>
        </div>

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative' }}>
          <AnimatePresence mode="wait">
            {/* Upload Tab */}
            {tab === 'upload' && (
              <motion.div key="upload" variants={contentVariants} initial="hidden" animate="visible" exit="exit" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                <div 
                  onDragOver={e => e.preventDefault()} 
                  onDrop={handleDrop}
                  style={{ flex: 1, border: '1px dashed var(--border)', borderRadius: '16px', padding: '48px 24px', textAlign: 'center', position: 'relative', cursor: 'pointer', transition: 'all 0.3s ease', background: 'var(--bg-card-soft)' }}
                  onMouseOver={e => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.background = 'rgba(0, 72, 249, 0.02)'; }}
                  onMouseOut={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.background = 'var(--bg-card-soft)'; }}
                >
                  <input type="file" onChange={e => setFile(e.target.files[0])} style={{ position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer', width: '100%' }} />
                  <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: '#fff', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 24px', boxShadow: '0 4px 14px rgba(0,0,0,0.05)' }}>
                    <UploadCloud size={28} color="var(--accent)" />
                  </div>
                  <div className="display-font" style={{ fontWeight: 600, fontSize: '18px', color: '#000', marginBottom: '8px' }}>
                    {file ? file.name : 'Drag & Drop PDF or Image'}
                  </div>
                  <div style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                    {file ? 'Click to change file selection' : 'NLP will auto-parse the document pipeline'}
                  </div>
                </div>

                <button 
                  className="btn-premium"
                  onClick={() => onAnalyzeFile(file, gender)} 
                  disabled={isLoading || !file}
                  style={{ width: '100%', marginTop: '32px' }}
                >
                  {isLoading ? 'Processing Pipeline...' : 'Initialize Analysis'}
                </button>
              </motion.div>
            )}

            {/* Paste Tab */}
            {tab === 'paste' && (
              <motion.div key="paste" variants={contentVariants} initial="hidden" animate="visible" exit="exit" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                <textarea 
                  className="premium-input"
                  value={text}
                  onChange={e => setText(e.target.value)}
                  placeholder="Paste raw OCR or standard text format..."
                  style={{ flex: 1, minHeight: '280px', resize: 'vertical', background: 'var(--bg-card-soft)' }}
                />
                <button 
                  className="btn-premium"
                  onClick={() => onAnalyzeText(text, gender)} 
                  disabled={isLoading || !text}
                  style={{ width: '100%', marginTop: '32px' }}
                >
                  {isLoading ? 'Processing Pipeline...' : 'Execute Text Parse'}
                </button>
              </motion.div>
            )}

            {/* Manual Tab */}
            {tab === 'manual' && (
              <motion.div key="manual" variants={contentVariants} initial="hidden" animate="visible" exit="exit" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                <div style={{ flex: 1, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '16px' }}>
                  {PARAM_LIST.map(p => (
                    <div key={p}>
                      <label className="mono" style={{ display: 'block', fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '8px', letterSpacing: '0.05em' }}>{p.toUpperCase()}</label>
                      <input 
                        type="number" 
                        className="premium-input"
                        placeholder="0.0"
                        onChange={e => handleManualChange(p, e.target.value)}
                        style={{ background: 'var(--bg-card-soft)' }}
                      />
                    </div>
                  ))}
                </div>
                <button 
                  className="btn-premium"
                  onClick={() => onAnalyzeManual(manualParams, gender)} 
                  disabled={isLoading || Object.keys(manualParams).length === 0}
                  style={{ width: '100%', marginTop: '32px' }}
                >
                  {isLoading ? 'Processing Pipeline...' : 'Run Diagnostics'}
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Error Toast */}
        <AnimatePresence>
          {error && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              style={{ background: 'rgba(230, 0, 0, 0.05)', border: '1px solid rgba(230, 0, 0, 0.2)', borderRadius: '12px', padding: '16px 20px', marginTop: '24px', fontSize: '14px', color: 'var(--danger)', display: 'flex', alignItems: 'flex-start', gap: '12px' }}
            >
              <div style={{ marginTop: '2px' }}>⚠️</div>
              <div style={{ lineHeight: 1.5 }}>{error}</div>
            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </div>
  );
}
