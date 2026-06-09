import { useState, useRef } from 'react';
import { Volume2, FileText, CheckCircle, AlertTriangle, AlertCircle, Cpu, ShieldAlert, Sparkles, PlusSquare, ActivitySquare } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ConfidenceRing from './ConfidenceRing';
import { fetchTTS, narrate } from '../api';

export default function ResultsPanel({ data }) {
  const [voiceMode, setVoiceMode] = useState('patient');
  const [language, setLanguage] = useState('english');
  const [showTranscript, setShowTranscript] = useState(false);
  const [ttsState, setTtsState] = useState({ loading: false, url: null, error: null, transcript: '' });
  const audioRef = useRef(null);

  if (!data) return null;

  const conditions = data.detected_conditions || [];
  const hasAbnormalFlags = (data.abnormal_flags || []).length > 0;
  const riskLevel = data.risk_level || 'NORMAL';
  
  // Build headline intelligently: avoid contradicting risk level
  let diagText = "Normal Blood Profile";
  if (conditions.length && !conditions.every(c => c.toLowerCase().includes('normal'))) {
    diagText = conditions.join(" + ");
  } else if (hasAbnormalFlags || riskLevel !== 'NORMAL') {
    diagText = `Abnormal Profile — ${riskLevel} Risk`;
  }
  
  const activeResults = (data.specialist_results || []).filter(s => s.active && s.confidence);
  const topConf = activeResults.length ? Math.max(...activeResults.map(s => s.confidence)) : 0.5;

  const getRiskIcon = (level) => {
    switch(level) {
      case 'NORMAL': return <CheckCircle size={16} />;
      case 'LOW': return <AlertCircle size={16} />;
      case 'MODERATE': return <AlertTriangle size={16} />;
      case 'HIGH': return <ShieldAlert size={16} />;
      default: return null;
    }
  };

  const handlePlayTTS = async () => {
    setTtsState(prev => ({ ...prev, loading: true, error: null }));
    if (audioRef.current) {
      audioRef.current.pause();
    }
    
    try {
      const textRes = await narrate(data, language, voiceMode);
      setTtsState(prev => ({ ...prev, transcript: textRes.spoken_text || textRes.english_text }));
      
      const audioBlob = await fetchTTS(data, language, voiceMode);
      const url = URL.createObjectURL(audioBlob);
      setTtsState(prev => ({ ...prev, url, loading: false }));
      
      if (audioRef.current) {
        audioRef.current.src = url;
        audioRef.current.play();
      }
    } catch (err) {
      setTtsState(prev => ({ ...prev, loading: false, error: 'TTS unavailable or error occurred.' }));
    }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.1 } }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" } }
  };

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="show">
      <motion.div variants={itemVariants} className="flex items-center gap-4" style={{ marginBottom: '32px' }}>
        <div style={{ width: '40px', height: '40px', borderRadius: '12px', background: 'var(--bg-card)', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Sparkles size={20} color="var(--accent)" />
        </div>
        <h3 className="display-font" style={{ fontSize: '24px', color: '#000', letterSpacing: '-0.02em', fontWeight: 700 }}>Diagnostic Results</h3>
        <div style={{ flex: 1, height: '1px', background: 'linear-gradient(90deg, var(--border), transparent)' }} />
      </motion.div>
      
      {/* ── ALARM / HERO CARD ── */}
      <motion.div variants={itemVariants} className="glass-card flex justify-between" style={{ padding: '48px', flexWrap: 'wrap', gap: '40px', marginBottom: '32px', position: 'relative', overflow: 'hidden' }}>
        {/* Subtle background glow based on risk level */}
        <div style={{ 
          position: 'absolute', top: '-50%', right: '-10%', width: '400px', height: '400px', 
          background: data.risk_level === 'HIGH' ? 'radial-gradient(circle, rgba(230,0,0,0.1) 0%, transparent 70%)' :
                      data.risk_level === 'MODERATE' ? 'radial-gradient(circle, rgba(230,115,0,0.1) 0%, transparent 70%)' :
                      'radial-gradient(circle, rgba(0,72,249,0.05) 0%, transparent 70%)',
          filter: 'blur(50px)', pointerEvents: 'none', zIndex: 0
        }} />

        <div style={{ flex: 1, zIndex: 1, position: 'relative' }}>
          <div className="mono" style={{ fontSize: '12px', color: 'var(--accent)', letterSpacing: '0.1em', marginBottom: '16px', textTransform: 'uppercase', fontWeight: 600 }}>
            Primary Discovery
          </div>
          <div className="display-font" style={{ 
            fontSize: 'clamp(32px, 5vw, 48px)', fontWeight: 800, lineHeight: 1.1, letterSpacing: '-0.03em',
            color: '#000'
          }}>
            {diagText}
          </div>
          
          <div className="flex items-center gap-4" style={{ marginTop: '32px', flexWrap: 'wrap' }}>
            <div className={`badge badge-${data.risk_level.toLowerCase()}`} style={{ fontSize: '13px', padding: '10px 20px', borderRadius: '10px', border: '1px solid transparent', fontWeight: 600 }}>
              {getRiskIcon(data.risk_level)} {data.risk_level} RISK
            </div>
            <div className="mono" style={{ fontSize: '13px', color: 'var(--text-secondary)', background: 'var(--bg-card-soft)', padding: '10px 20px', borderRadius: '10px', border: '1px solid var(--border)' }}>
              {data.total_abnormal} ABNORMAL METRICS
            </div>
          </div>
          
          <div style={{ fontSize: '16px', color: 'var(--text-secondary)', marginTop: '28px', lineHeight: 1.7, maxWidth: '700px' }}>
            {data.summary}
          </div>
        </div>
        
        <div style={{ zIndex: 1, position: 'relative', display: 'flex', alignItems: 'center' }}>
          <ConfidenceRing confidence={topConf} />
        </div>
      </motion.div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '32px', marginBottom: '32px' }}>
        
        {/* ── SPECIALIST GRID ── */}
        <motion.div variants={itemVariants} className="glass-card" style={{ padding: '32px' }}>
          <div className="flex items-center gap-4" style={{ marginBottom: '28px' }}>
            <Cpu size={20} color="var(--accent)" />
            <span className="display-font" style={{ fontWeight: 700, fontSize: '20px', color: '#000', letterSpacing: '-0.02em' }}>Neural Sub-Engines</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {[...data.specialist_results, ...(data.skipped_specialists || [])].map((s, i) => {
              const isSkipped = !s.active;
              const conf = s.confidence != null ? Math.round(s.confidence * 100) : null;
              
              return (
                <div key={i} style={{ padding: '20px', borderRadius: '16px', background: isSkipped ? 'var(--bg)' : 'var(--bg-card-soft)', border: '1px solid var(--border)', opacity: isSkipped ? 0.6 : 1, transition: 'all 0.3s ease' }}
                onMouseOver={e => { if(!isSkipped) e.currentTarget.style.borderColor = 'var(--border-hover)' }}
                onMouseOut={e => { if(!isSkipped) e.currentTarget.style.borderColor = 'var(--border)' }}
                >
                  <div className="flex justify-between items-center" style={{ marginBottom: '12px' }}>
                    <div className="mono" style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      {s.specialist}
                    </div>
                    {s.model_f1 && (
                      <div className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>F1: {(s.model_f1*100).toFixed(1)}%</div>
                    )}
                  </div>
                  <div className="display-font" style={{ fontSize: '16px', fontWeight: 600, color: '#000', marginBottom: conf != null ? '16px' : '0' }}>
                    {isSkipped ? 'INACTIVE (MISSING DATA)' : s.diagnosis || s.reason || '—'}
                  </div>
                  {conf != null && (
                    <div className="flex items-center gap-4">
                      <div style={{ flex: 1, height: '4px', background: 'var(--border)', borderRadius: '2px', overflow: 'hidden' }}>
                        <motion.div 
                          initial={{ width: 0 }}
                          animate={{ width: `${conf}%` }}
                          transition={{ duration: 1, delay: 0.5 + (i * 0.1) }}
                          style={{ height: '100%', background: 'var(--accent)', borderRadius: '2px' }} 
                        />
                      </div>
                      <div className="mono" style={{ fontSize: '11px', color: 'var(--text-secondary)', fontWeight: 600 }}>{conf}%</div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </motion.div>

        {/* ── FLAGS & RECOMMENDATIONS ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
          
          {data.abnormal_flags?.length > 0 && (
            <motion.div variants={itemVariants} className="glass-card" style={{ padding: '32px' }}>
              <div className="flex items-center gap-4" style={{ marginBottom: '28px' }}>
                <ActivitySquare size={20} color="var(--accent)" />
                <span className="display-font" style={{ fontWeight: 700, fontSize: '20px', color: '#000', letterSpacing: '-0.02em' }}>Critical Flags</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {data.abnormal_flags.map((f, i) => (
                  <div key={i} className="flex gap-4 items-center" style={{ padding: '16px 20px', borderRadius: '16px', background: 'var(--bg-card-soft)', border: '1px solid var(--border)' }}>
                    <div style={{ width: '4px', height: '40px', borderRadius: '2px', background: f.status === 'HIGH' ? 'var(--danger)' : 'var(--warning)' }} />
                    <div style={{ flex: 1 }}>
                      <div className="display-font" style={{ fontSize: '15px', fontWeight: 600, color: '#000' }}>{f.parameter.replace(/_/g, ' ')}</div>
                      <div className="mono" style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '6px', letterSpacing: '0.05em' }}>{f.status} FLAG</div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div className="mono" style={{ fontSize: '20px', fontWeight: 600, color: '#000', lineHeight: 1 }}>{f.value}</div>
                      <div className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px' }}>NORM: {f.normal_range}</div>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {data.recommendations?.length > 0 && (
            <motion.div variants={itemVariants} className="glass-card" style={{ padding: '32px' }}>
               <div className="flex items-center gap-4" style={{ marginBottom: '28px' }}>
                <PlusSquare size={20} color="var(--accent)" />
                <span className="display-font" style={{ fontWeight: 700, fontSize: '20px', color: '#000', letterSpacing: '-0.02em' }}>Clinical Recommendations</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                {data.recommendations.map((rec, i) => (
                  <div key={i} className="flex gap-4" style={{ fontSize: '15px', lineHeight: 1.6, color: 'var(--text-secondary)' }}>
                    <div style={{ marginTop: '6px', width: '6px', height: '6px', borderRadius: '50%', background: 'var(--accent)', flexShrink: 0 }} />
                    <span>{rec}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </div>
      </div>

      {/* ── AUDIO PANEL ── */}
      <motion.div variants={itemVariants} className="glass-card" style={{ padding: '32px', marginBottom: '40px' }}>
        <div style={{ fontSize: '18px', fontWeight: 700, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '12px', color: '#000', letterSpacing: '-0.02em' }} className="display-font">
          <Volume2 size={20} color="var(--accent)" /> Sarvam AI Voice Synthesizer
        </div>
        
        <div className="flex items-center gap-4" style={{ flexWrap: 'wrap' }}>
          <select 
            className="premium-input"
            value={language} onChange={e => setLanguage(e.target.value)}
            style={{ width: 'auto', minWidth: '180px', padding: '12px 16px', background: '#fff' }}
          >
            <option value="english">🇬🇧 English</option>
            <option value="hindi">🇮🇳 Hindi</option>
            <option value="tamil">Tamil</option>
            <option value="telugu">Telugu</option>
            <option value="kannada">Kannada</option>
            <option value="malayalam">Malayalam</option>
            <option value="bengali">Bengali</option>
            <option value="marathi">Marathi</option>
            <option value="gujarati">Gujarati</option>
            <option value="punjabi">Punjabi</option>
          </select>

          <div className="flex" style={{ background: 'var(--bg-card-soft)', borderRadius: '12px', padding: '6px', border: '1px solid var(--border)' }}>
            <button 
              onClick={() => setVoiceMode('patient')}
              style={{ padding: '10px 20px', borderRadius: '8px', background: voiceMode === 'patient' ? '#fff' : 'transparent', color: voiceMode === 'patient' ? '#000' : 'var(--text-secondary)', border: 'none', fontSize: '14px', transition: 'all 0.3s ease', fontWeight: voiceMode === 'patient' ? 600 : 500, boxShadow: voiceMode === 'patient' ? '0 2px 10px rgba(0,0,0,0.05)' : 'none' }}
            >
              Patient Context
            </button>
            <button 
              onClick={() => setVoiceMode('clinical')}
              style={{ padding: '10px 20px', borderRadius: '8px', background: voiceMode === 'clinical' ? '#fff' : 'transparent', color: voiceMode === 'clinical' ? '#000' : 'var(--text-secondary)', border: 'none', fontSize: '14px', transition: 'all 0.3s ease', fontWeight: voiceMode === 'clinical' ? 600 : 500, boxShadow: voiceMode === 'clinical' ? '0 2px 10px rgba(0,0,0,0.05)' : 'none' }}
            >
              Clinical Context
            </button>
          </div>

          <button 
            className="btn-premium"
            onClick={handlePlayTTS} disabled={ttsState.loading}
            style={{ padding: '12px 28px', borderRadius: '12px', width: 'auto' }}
          >
            {ttsState.loading ? 'Synthesizing...' : '▶ Render Audio'}
          </button>
          
          <button 
            onClick={() => setShowTranscript(!showTranscript)}
            style={{ padding: '12px 20px', borderRadius: '12px', background: '#fff', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: '10px', fontSize: '14px', color: 'var(--text-primary)', transition: 'all 0.3s ease', fontWeight: 500 }}
            onMouseOver={e => e.currentTarget.style.borderColor = 'var(--border-hover)'}
            onMouseOut={e => e.currentTarget.style.borderColor = 'var(--border)'}
          >
            <FileText size={18} /> Read Logs
          </button>
        </div>
        
        <audio ref={audioRef} controls style={{ width: '100%', marginTop: '24px', display: ttsState.url ? 'block' : 'none', borderRadius: '12px' }} />
        
        {ttsState.error && <div className="mono" style={{ fontSize: '12px', color: 'var(--danger)', marginTop: '16px' }}>[ERROR] {ttsState.error}</div>}
        
        <AnimatePresence>
          {showTranscript && ttsState.transcript && (
            <motion.div 
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mono" style={{ background: '#f9f9f9', border: '1px solid var(--border)', borderRadius: '16px', padding: '24px', marginTop: '24px', fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.8, maxHeight: '250px', overflowY: 'auto' }}
            >
              <div style={{ color: 'var(--accent)', marginBottom: '12px', letterSpacing: '0.05em', fontWeight: 600 }}>// GENERATED TRANSCRIPT [{language.toUpperCase()}]</div>
              <div style={{ color: '#000' }}>{ttsState.transcript}</div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </motion.div>
  );
}
