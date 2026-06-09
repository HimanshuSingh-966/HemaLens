import axios from 'axios';

// Vite proxies /api to the backend in dev mode
const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json'
  }
});

export const analyzeFile = async (file, gender) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post(`/analyze/file?gender=${gender}`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
  return response.data;
};

export const extractText = async (text) => {
  const response = await api.post('/extract/text', { text });
  return response.data;
};

export const analyzeParams = async (params, gender) => {
  const response = await api.post('/analyze/params', { gender, params });
  return response.data;
};

export const narrate = async (result, language, mode) => {
  const response = await api.post('/narrate', { result, language, mode });
  return response.data;
};

export const fetchTTS = async (result, language, mode) => {
  const response = await api.post('/tts', { result, language, mode }, {
    responseType: 'blob'
  });
  return response.data;
};

export const fetchHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const fetchReferenceRanges = async (gender = 'male') => {
  const response = await api.get(`/reference-ranges?gender=${gender}`);
  return response.data;
};
