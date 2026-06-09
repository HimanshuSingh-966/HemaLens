import { useState } from 'react';
import AnalysisInput from '../components/AnalysisInput';
import ParamsTable from '../components/ParamsTable';
import ResultsPanel from '../components/ResultsPanel';
import { analyzeFile, extractText, analyzeParams } from '../api';
import { motion, AnimatePresence } from 'framer-motion';

export default function WorkspacePage() {
  const [extractedParams, setExtractedParams] = useState({});
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAnalyzeFile = async (file, gender) => {
    setIsLoading(true);
    setError('');
    try {
      const data = await analyzeFile(file, gender);
      setExtractedParams(data.params);
      setResults(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnalyzeText = async (text, gender) => {
    setIsLoading(true);
    setError('');
    try {
      const extData = await extractText(text);
      if (!extData.count) throw new Error("No parameters could be extracted.");
      setExtractedParams(extData.params);
      
      const data = await analyzeParams(extData.params, gender);
      setResults(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnalyzeManual = async (params, gender) => {
    setIsLoading(true);
    setError('');
    try {
      const data = await analyzeParams(params, gender);
      setExtractedParams(params);
      setResults(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const fadeUpVariant = {
    hidden: { opacity: 0, y: 30 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" } }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
      <div style={{ marginBottom: '48px', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
        <div className="display-font" style={{ fontSize: '48px', fontWeight: 800, letterSpacing: '-0.03em', color: '#000' }}>
          Diagnostic Workspace
        </div>
        <p style={{ marginTop: '16px', color: 'var(--text-secondary)', fontSize: '18px', maxWidth: '600px' }}>
          Upload a report and explore extracted parameters, risk scores, and specialist recommendations in real time.
        </p>
      </div>
      
      <motion.div variants={fadeUpVariant} initial="hidden" animate="show" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '32px', alignItems: 'start' }}>
        <AnalysisInput 
          onAnalyzeFile={handleAnalyzeFile} 
          onAnalyzeText={handleAnalyzeText} 
          onAnalyzeManual={handleAnalyzeManual}
          isLoading={isLoading}
          error={error}
        />
        <div style={{ display: 'flex', flexDirection: 'column', gap: '32px', height: '100%' }}>
          <ParamsTable params={extractedParams} />
        </div>
      </motion.div>
      
      <AnimatePresence>
        {results && (
          <motion.div 
            id="results" 
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -50 }}
            transition={{ duration: 0.5 }}
            style={{ marginTop: '40px' }}
          >
            <ResultsPanel data={results} />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
