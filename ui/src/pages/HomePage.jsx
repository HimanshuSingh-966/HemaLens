import Hero from '../components/Hero';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';

export default function HomePage() {
  const fadeUpVariant = {
    hidden: { opacity: 0, y: 30 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" } }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
      <Hero />

      <motion.div initial="hidden" whileInView="show" viewport={{ once: true, margin: "-100px" }} style={{ display: 'flex', flexDirection: 'column', gap: '80px', marginBottom: '80px' }}>
        <motion.section variants={fadeUpVariant} id="how-it-works" className="section" style={{ marginTop: '0' }}>
          <div style={{ marginBottom: '64px', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
            <div className="display-font" style={{ fontSize: '48px', fontWeight: 800, letterSpacing: '-0.03em', color: '#000' }}>
              How It Works
            </div>
            <p style={{ marginTop: '20px', color: 'var(--text-secondary)', fontSize: '18px', maxWidth: '600px', lineHeight: 1.6 }}>
              Three simple steps to transform raw blood reports into AI-assisted diagnostic clarity.
            </p>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '32px' }}>
            <div className="glass-card" style={{ padding: '48px' }}>
              <div className="mono" style={{ color: 'var(--accent)', fontSize: '13px', letterSpacing: '0.1em', marginBottom: '24px', fontWeight: 600 }}>01 / UPLOAD</div>
              <div className="display-font" style={{ fontSize: '28px', fontWeight: 700, color: '#000', marginBottom: '16px', letterSpacing: '-0.02em' }}>Input Data</div>
              <p style={{ color: 'var(--text-secondary)', lineHeight: 1.7, fontSize: '16px' }}>
                Upload report files, paste extracted text, or enter biomarker values manually into the secure system.
              </p>
            </div>
            <div className="glass-card" style={{ padding: '48px' }}>
              <div className="mono" style={{ color: 'var(--accent)', fontSize: '13px', letterSpacing: '0.1em', marginBottom: '24px', fontWeight: 600 }}>02 / INGESTION</div>
              <div className="display-font" style={{ fontSize: '28px', fontWeight: 700, color: '#000', marginBottom: '16px', letterSpacing: '-0.02em' }}>AI Processing</div>
              <p style={{ color: 'var(--text-secondary)', lineHeight: 1.7, fontSize: '16px' }}>
                NLP extraction and 5 specialist ML models process the metrics in parallel for structured diagnosis output.
              </p>
            </div>
            <div className="glass-card" style={{ padding: '48px' }}>
              <div className="mono" style={{ color: 'var(--accent)', fontSize: '13px', letterSpacing: '0.1em', marginBottom: '24px', fontWeight: 600 }}>03 / OUTPUT</div>
              <div className="display-font" style={{ fontSize: '28px', fontWeight: 700, color: '#000', marginBottom: '16px', letterSpacing: '-0.02em' }}>Refined Insights</div>
              <p style={{ color: 'var(--text-secondary)', lineHeight: 1.7, fontSize: '16px' }}>
                Review confidence scores, risk flags, and actionable recommendations in a pristine clinical presentation.
              </p>
            </div>
          </div>
        </motion.section>

        <motion.section variants={fadeUpVariant} className="glass-card section" style={{ padding: '80px 40px', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', background: '#fff' }}>
          <div className="display-font" style={{ fontSize: '48px', fontWeight: 800, color: '#000', letterSpacing: '-0.03em' }}>
            Ready to Analyze?
          </div>
          <p style={{ marginTop: '24px', color: 'var(--text-secondary)', fontSize: '18px', maxWidth: '700px', lineHeight: 1.6 }}>
            Experience the precision of HemaLens in our interactive diagnostic workspace.
          </p>
          <Link to="/workspace" className="btn-premium" style={{ marginTop: '40px', display: 'inline-flex' }}>
            Open Workspace
          </Link>
        </motion.section>
      </motion.div>
    </motion.div>
  );
}
