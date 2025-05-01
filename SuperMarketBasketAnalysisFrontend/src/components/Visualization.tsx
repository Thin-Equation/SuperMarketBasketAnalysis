import React, { useState, useEffect } from 'react';
import { getVisualization } from '../utils/api';

interface VisualizationProps {
  isModelTrained: boolean;
  onError: (error: string) => void;
}

const Visualization: React.FC<VisualizationProps> = ({ isModelTrained, onError }) => {
  const [loading, setLoading] = useState(false);
  const [visualizationData, setVisualizationData] = useState<string | null>(null);

  const loadVisualization = async () => {
    if (!isModelTrained) {
      onError('Please train the model first');
      return;
    }

    try {
      setLoading(true);
      const result = await getVisualization();
      if (result.visualization) {
        setVisualizationData(result.visualization);
      }
    } catch (error: any) {
      onError(error.response?.data?.detail || 'Failed to load visualization');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isModelTrained) {
      loadVisualization();
    }
  }, [isModelTrained]);

  return (
    <div className="p-6 bg-white rounded-lg shadow-md mt-4">
      <h2 className="text-xl font-semibold mb-4">Visualization</h2>
      <p className="text-sm text-gray-600 mb-4">
        Top frequent itemsets visualization.
      </p>

      {loading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-700"></div>
        </div>
      ) : visualizationData ? (
        <div className="mt-2 flex justify-center">
          <img 
            src={visualizationData} 
            alt="Frequent Itemsets Visualization"
            className="max-w-full h-auto rounded-lg shadow-sm"
          />
        </div>
      ) : (
        <div className="text-center py-10 bg-gray-50 rounded-lg">
          <p className="text-gray-500">No visualization available</p>
          <button
            onClick={loadVisualization}
            disabled={!isModelTrained || loading}
            className={`mt-4 px-4 py-2 text-white rounded-md ${
              !isModelTrained ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            Refresh Visualization
          </button>
        </div>
      )}
    </div>
  );
};

export default Visualization;