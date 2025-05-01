import React, { useState } from 'react';
import { trainModel } from '../utils/api';

interface ModelTrainingProps {
  isDataProcessed: boolean;
  onTrainSuccess: (data: any) => void;
  onTrainError: (error: string) => void;
}

const ModelTraining: React.FC<ModelTrainingProps> = ({
  isDataProcessed,
  onTrainSuccess,
  onTrainError,
}) => {
  const [minSupport, setMinSupport] = useState<number>(0.02);
  const [minConfidence, setMinConfidence] = useState<number>(0.2);
  const [training, setTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState<any>(null);

  const handleTrain = async () => {
    if (!isDataProcessed) {
      onTrainError('Please process data first');
      return;
    }

    try {
      setTraining(true);
      const result = await trainModel(minSupport, minConfidence);
      setTrainingResult(result);
      onTrainSuccess(result);
    } catch (error: any) {
      onTrainError(error.response?.data?.detail || 'Training failed');
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-md mt-4">
      <h2 className="text-xl font-semibold mb-4">Train Model</h2>
      <p className="text-sm text-gray-600 mb-4">
        Configure and train the FP-Growth algorithm to find frequent itemsets and association rules.
      </p>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label htmlFor="min-support" className="block text-sm font-medium text-gray-700">
            Minimum Support: {minSupport}
          </label>
          <input
            type="range"
            id="min-support"
            min="0.001"
            max="0.1"
            step="0.001"
            value={minSupport}
            onChange={(e) => setMinSupport(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <p className="text-xs text-gray-500 mt-1">
            Support determines how frequently an itemset appears in the dataset
          </p>
        </div>
        
        <div>
          <label htmlFor="min-confidence" className="block text-sm font-medium text-gray-700">
            Minimum Confidence: {minConfidence}
          </label>
          <input
            type="range"
            id="min-confidence"
            min="0.01"
            max="0.9"
            step="0.01"
            value={minConfidence}
            onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <p className="text-xs text-gray-500 mt-1">
            Confidence indicates how likely item Y is purchased when item X is purchased
          </p>
        </div>
      </div>
      
      <button
        onClick={handleTrain}
        disabled={!isDataProcessed || training}
        className={`px-4 py-2 text-white rounded-md mb-4 ${
          !isDataProcessed || training ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
        }`}
      >
        {training ? 'Training...' : 'Train Model'}
      </button>

      {trainingResult && (
        <div className="mt-4">
          <h3 className="font-medium text-lg mb-2">Training Results:</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-50 p-4 rounded-md">
              <h4 className="font-medium mb-2">Frequent Itemsets:</h4>
              <p>Count: {trainingResult.frequent_itemsets_count}</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-md">
              <h4 className="font-medium mb-2">Association Rules:</h4>
              <p>Count: {trainingResult.rules_count}</p>
            </div>
          </div>

          {trainingResult.sample_rules && trainingResult.sample_rules.length > 0 && (
            <div className="mt-4">
              <h4 className="font-medium mb-2">Sample Rules:</h4>
              <div className="overflow-auto max-h-48 bg-gray-50 p-2 rounded-md">
                <table className="min-w-full">
                  <thead>
                    <tr>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Antecedent</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Consequent</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trainingResult.sample_rules.slice(0, 5).map((rule: any, index: number) => (
                      <tr key={index} className="border-t">
                        <td className="px-4 py-2 text-sm">
                          {rule.antecedent.map((item: string) => (
                            <span key={item} className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded mr-1 mb-1">
                              {item.length > 20 ? item.substring(0, 20) + '...' : item}
                            </span>
                          ))}
                        </td>
                        <td className="px-4 py-2 text-sm">
                          {rule.consequent.map((item: string) => (
                            <span key={item} className="inline-block bg-green-100 text-green-800 text-xs px-2 py-1 rounded mr-1 mb-1">
                              {item.length > 20 ? item.substring(0, 20) + '...' : item}
                            </span>
                          ))}
                        </td>
                        <td className="px-4 py-2 text-sm">{(rule.confidence * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ModelTraining;