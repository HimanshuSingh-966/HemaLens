import { Outlet, Link, useLocation } from 'react-router-dom';
import { Activity } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Layout() {
  const location = useLocation();

  const getNavStyle = (path) => ({
    fontSize: '14px',
    color: location.pathname === path ? '#000' : 'var(--text-secondary)',
    background: location.pathname === path ? '#fff' : 'transparent',
    padding: '8px 18px',
    borderRadius: '999px',
    transition: 'all 0.3s ease',
    fontWeight: location.pathname === path ? 600 : 500,
    boxShadow: location.pathname === path ? '0 2px 10px rgba(0,0,0,0.05)' : 'none'
  });

  return (
    <div style={{ minHeight: '100vh', width: '100%', display: 'flex', flexDirection: 'column' }}>
      <motion.header 
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 100, damping: 20, delay: 0.1 }}
        style={{ 
          position: 'sticky', 
          top: '24px', 
          zIndex: 40, 
          margin: '0 auto',
          width: 'calc(100% - 48px)',
          maxWidth: '1200px'
        }}
      >
        <div className="glass-card" style={{ 
          padding: '12px 24px', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          borderRadius: '999px'
        }}>
          <Link to="/" className="flex items-center gap-3" style={{ textDecoration: 'none' }}>
            <div style={{ 
              width: '40px', height: '40px', 
              borderRadius: '50%', 
              display: 'flex', alignItems: 'center', justifyContent: 'center', 
              background: 'var(--accent)',
              color: '#ffffff',
              boxShadow: '0 4px 14px rgba(0, 72, 249, 0.2)'
            }}>
              <Activity size={20} strokeWidth={2.5} />
            </div>
            <div>
              <div className="display-font" style={{ fontWeight: 800, fontSize: '20px', lineHeight: 1, letterSpacing: '-0.03em', color: '#000' }}>HemaLens</div>
            </div>
          </Link>

          <nav className="flex items-center gap-1" style={{ background: 'rgba(0,0,0,0.04)', padding: '4px', borderRadius: '999px' }}>
            <Link 
              to="/" 
              style={getNavStyle('/')} 
              onMouseOver={e => {if(location.pathname !== '/') e.target.style.color = '#000'}} 
              onMouseOut={e => {if(location.pathname !== '/') e.target.style.color = 'var(--text-secondary)'}}
            >
              Home
            </Link>
            <Link 
              to="/workspace" 
              style={getNavStyle('/workspace')} 
              onMouseOver={e => {if(location.pathname !== '/workspace') e.target.style.color = '#000'}} 
              onMouseOut={e => {if(location.pathname !== '/workspace') e.target.style.color = 'var(--text-secondary)'}}
            >
              Workspace
            </Link>
          </nav>

          <div className="badge badge-low" style={{ background: 'rgba(0, 72, 249, 0.08)', color: 'var(--accent)', borderColor: 'rgba(0, 72, 249, 0.15)', fontWeight: 600 }}>
            v2.0 Beta
          </div>
        </div>
      </motion.header>

      <main className="app-main" style={{ flex: 1, maxWidth: '1200px', width: '100%', margin: '0 auto', padding: '60px 24px 80px' }}>
        <Outlet />
      </main>

      <footer style={{ 
        padding: '60px 24px 40px', 
        borderTop: '1px solid var(--border)', 
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '16px',
        marginTop: 'auto'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#000' }}>
          <Activity size={18} />
          <span className="display-font" style={{ fontWeight: 700, fontSize: '16px' }}>HemaLens</span>
        </div>
        <p style={{ color: 'var(--text-muted)', fontSize: '14px', letterSpacing: '0.01em' }}>
          © 2026 AI Diagnostic Systems. All rights reserved.
        </p>
      </footer>
    </div>
  );
}
