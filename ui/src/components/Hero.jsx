import { ArrowRight, Database, FileText, Activity } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Hero() {
  const containerVariants = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.15 } }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 40 },
    show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] } }
  };

  return (
    <motion.section 
      variants={containerVariants}
      initial="hidden"
      animate="show"
      style={{
        paddingTop: '60px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        textAlign: 'center',
        position: 'relative'
      }}
    >
      <motion.div variants={itemVariants} className="badge badge-normal" style={{ marginBottom: '24px', background: 'rgba(0, 72, 249, 0.08)', color: 'var(--accent)', borderColor: 'rgba(0, 72, 249, 0.15)', padding: '6px 16px', fontSize: '12px' }}>
        <span style={{ display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', background: 'var(--accent)' }} />
        Neural Engine v2 is Online
      </motion.div>
      
      <motion.h1 variants={itemVariants} className="display-font" style={{
        fontSize: 'clamp(48px, 8vw, 88px)',
        fontWeight: 900,
        lineHeight: 1.05,
        letterSpacing: '-0.04em',
        maxWidth: '1000px',
        color: '#000'
      }}>
        Blood Reports, <br/>
        <span className="text-gradient">Decoded by AI.</span>
      </motion.h1>

      <motion.p variants={itemVariants} style={{
        marginTop: '24px',
        fontSize: 'clamp(16px, 2vw, 20px)',
        color: 'var(--text-secondary)',
        maxWidth: '650px',
        lineHeight: 1.6
      }}>
        Upload any blood report and let our ensemble of specialized machine learning models instantly identify risks, extract biomarkers, and generate clinical insights.
      </motion.p>

      <motion.div variants={itemVariants} className="flex items-center gap-4" style={{ marginTop: '40px' }}>
        <a href="#how-it-works" className="btn-secondary" style={{ padding: '16px 32px' }}>
          See how it works
        </a>
      </motion.div>
      
      {/* Bento Grid */}
      <motion.div variants={itemVariants} style={{ marginTop: '80px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px', width: '100%', maxWidth: '1000px' }}>
        <div className="glass-card" style={{ padding: '32px', textAlign: 'left', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(0, 72, 249, 0.08)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Database size={24} color="var(--accent)" />
          </div>
          <div className="display-font" style={{ fontSize: '20px', fontWeight: 700, color: '#000', letterSpacing: '-0.02em' }}>Multi-Modal Ingestion</div>
          <div style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>Native support for PDF lab reports, raw text extraction, and manual entry.</div>
        </div>
        <div className="glass-card" style={{ padding: '32px', textAlign: 'left', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(0, 72, 249, 0.08)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Activity size={24} color="var(--accent)" />
          </div>
          <div className="display-font" style={{ fontSize: '20px', fontWeight: 700, color: '#000', letterSpacing: '-0.02em' }}>5 Specialist Models</div>
          <div style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>Ensemble of models trained for diabetes, liver, kidney, thyroid, and anemia.</div>
        </div>
        <div className="glass-card" style={{ padding: '32px', textAlign: 'left', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(0, 72, 249, 0.08)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <FileText size={24} color="var(--accent)" />
          </div>
          <div className="display-font" style={{ fontSize: '20px', fontWeight: 700, color: '#000', letterSpacing: '-0.02em' }}>Sarvam Audio Logs</div>
          <div style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>Generate high-quality multi-lingual audio summaries using Sarvam TTS.</div>
        </div>
      </motion.div>
    </motion.section>
  );
}
