import axios from 'axios';

// Create an axios instance for API calls
const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const uploadFile = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const processData = async () => {
  const response = await api.post('/process');
  return response.data;
};

export const trainModel = async (minSupport: number, minConfidence: number) => {
  const response = await api.post('/train', null, {
    params: {
      min_support: minSupport,
      min_confidence: minConfidence,
    },
  });
  return response.data;
};

export const getVisualization = async () => {
  const response = await api.get('/visualization');
  return response.data;
};

export const getRecommendations = async (items: string[], count: number = 3) => {
  const response = await api.post('/recommend', {
    items,
    recommendations_count: count,
  });
  return response.data;
};

export const resetModel = async () => {
  const response = await api.post('/reset');
  return response.data;
};

export default api;