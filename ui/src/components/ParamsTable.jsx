import { Microscope, HelpCircle } from 'lucide-react';
import { motion } from 'framer-motion';

export default function ParamsTable({ params }) {
  const paramEntries = Object.entries(params || {});
  
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card" 
      style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}
    >
      
      {/* Header */}
      <div className="flex items-center justify-between" style={{ padding: '24px 32px', borderBottom: '1px solid var(--border)', background: 'var(--bg-card-soft)' }}>
        <div className="flex items-center gap-4">
          <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(0, 72, 249, 0.08)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Microscope size={24} color="var(--accent)" />
          </div>
          <div>
            <div className="display-font" style={{ fontWeight: 700, fontSize: '20px', color: '#000', letterSpacing: '-0.02em' }}>Extracted Telemetry</div>
            <div className="mono" style={{ fontSize: '11px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginTop: '4px' }}>
              {paramEntries.length} vectors identified
            </div>
          </div>
        </div>
      </div>

      <div style={{ flex: 1, padding: '0', display: 'flex', flexDirection: 'column' }}>
        {paramEntries.length === 0 ? (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '80px 20px', color: 'var(--text-muted)' }}>
            <div style={{ width: '80px', height: '80px', borderRadius: '50%', background: 'var(--bg-card-soft)', border: '1px dashed var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '24px' }}>
              <Microscope size={32} style={{ opacity: 0.4, color: 'var(--text-secondary)' }} />
            </div>
            <div className="display-font" style={{ fontSize: '18px', fontWeight: 600, color: '#000', opacity: 0.8 }}>Awaiting Data Ingestion</div>
            <div className="mono" style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '12px', letterSpacing: '0.02em' }}>Input parameters to populate this panel</div>
          </div>
        ) : (
          <div style={{ overflowX: 'auto', flex: 1 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ background: 'var(--bg-card-soft)' }}>
                  <th className="mono" style={{ padding: '16px 32px', fontSize: '11px', color: 'var(--text-secondary)', fontWeight: 600, letterSpacing: '0.05em', borderBottom: '1px solid var(--border)' }}>METRIC</th>
                  <th className="mono" style={{ padding: '16px 32px', fontSize: '11px', color: 'var(--text-secondary)', fontWeight: 600, letterSpacing: '0.05em', borderBottom: '1px solid var(--border)', textAlign: 'right' }}>VALUE</th>
                  <th className="mono" style={{ padding: '16px 32px', fontSize: '11px', color: 'var(--text-secondary)', fontWeight: 600, letterSpacing: '0.05em', borderBottom: '1px solid var(--border)', width: '80px', textAlign: 'center' }}><HelpCircle size={16} /></th>
                </tr>
              </thead>
              <tbody>
                {paramEntries.map(([k, v], i) => (
                  <motion.tr 
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05 }}
                    key={k} 
                    style={{ 
                      borderBottom: i === paramEntries.length - 1 ? 'none' : '1px solid var(--border)',
                      transition: 'background 0.3s ease',
                    }}
                    onMouseOver={e => e.currentTarget.style.background = 'rgba(0, 72, 249, 0.02)'}
                    onMouseOut={e => e.currentTarget.style.background = 'transparent'}
                  >
                    <td style={{ padding: '18px 32px', color: '#000', fontSize: '15px', fontWeight: 600 }}>
                      {k.replace(/_/g, ' ')}
                    </td>
                    <td className="mono" style={{ padding: '18px 32px', color: 'var(--accent)', fontSize: '15px', textAlign: 'right', fontWeight: 600 }}>
                      {v}
                    </td>
                    <td style={{ padding: '18px 32px', textAlign: 'center' }}>
                      <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--text-muted)', display: 'inline-block', opacity: 0.3 }} />
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </motion.div>
  );
}
