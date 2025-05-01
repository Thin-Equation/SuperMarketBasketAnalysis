import React, { useState } from 'react';
import { getRecommendations } from '../utils/api';

interface RecommendationsProps {
  isModelTrained: boolean;
  onError: (error: string) => void;
  availableItems?: string[];
}

const Recommendations: React.FC<RecommendationsProps> = ({ 
  isModelTrained, 
  onError,
  availableItems = []
}) => {
  const [selectedItems, setSelectedItems] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [recommendationsCount, setRecommendationsCount] = useState(3);
  const [loading, setLoading] = useState(false);
  const [recommendationResults, setRecommendationResults] = useState<any>(null);

  const filteredItems = availableItems.filter(item => 
    item.toLowerCase().includes(searchTerm.toLowerCase())
  ).slice(0, 10);

  const addItem = (item: string) => {
    if (!selectedItems.includes(item)) {
      setSelectedItems([...selectedItems, item]);
    }
    setSearchTerm('');
  };

  const removeItem = (item: string) => {
    setSelectedItems(selectedItems.filter(i => i !== item));
  };

  const getRecommendationsHandler = async () => {
    if (!isModelTrained) {
      onError('Please train the model first');
      return;
    }

    if (selectedItems.length === 0) {
      onError('Please select at least one item');
      return;
    }

    try {
      setLoading(true);
      const result = await getRecommendations(selectedItems, recommendationsCount);
      setRecommendationResults(result);
    } catch (error: any) {
      onError(error.response?.data?.detail || 'Failed to get recommendations');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-md mt-4">
      <h2 className="text-xl font-semibold mb-4">Product Recommendations</h2>
      <p className="text-sm text-gray-600 mb-4">
        Enter items in a shopping cart to get product recommendations.
      </p>

      <div className="mb-4">
        <label htmlFor="item-search" className="block text-sm font-medium text-gray-700 mb-1">
          Search Products
        </label>
        <div className="flex">
          <input
            id="item-search"
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Type to search products..."
            className="flex-1 p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          />
          {availableItems.length === 0 && (
            <button
              onClick={() => addItem(searchTerm)}
              disabled={!searchTerm.trim()}
              className="ml-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              Add
            </button>
          )}
        </div>

        {searchTerm && availableItems.length > 0 && (
          <div className="mt-2 p-2 max-h-40 overflow-auto border border-gray-200 rounded-md bg-white">
            {filteredItems.length > 0 ? (
              filteredItems.map((item, index) => (
                <div
                  key={index}
                  className="p-1 hover:bg-blue-50 cursor-pointer rounded"
                  onClick={() => addItem(item)}
                >
                  {item}
                </div>
              ))
            ) : (
              <div className="p-1 text-gray-500">No matching items</div>
            )}
          </div>
        )}
      </div>

      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Selected Items
        </label>
        <div className="min-h-16 p-2 border border-gray-200 rounded-md bg-gray-50">
          {selectedItems.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {selectedItems.map((item, index) => (
                <div
                  key={index}
                  className="inline-flex items-center bg-blue-100 text-blue-800 text-sm px-2 py-1 rounded"
                >
                  <span className="max-w-xs truncate">{item}</span>
                  <button
                    type="button"
                    className="ml-1 text-blue-500 hover:text-blue-700 focus:outline-none"
                    onClick={() => removeItem(item)}
                  >
                    &times;
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-gray-500 text-sm">No items selected</div>
          )}
        </div>
      </div>

      <div className="mb-4">
        <label htmlFor="recommendations-count" className="block text-sm font-medium text-gray-700">
          Number of Recommendations: {recommendationsCount}
        </label>
        <input
          type="range"
          id="recommendations-count"
          min="1"
          max="10"
          step="1"
          value={recommendationsCount}
          onChange={(e) => setRecommendationsCount(parseInt(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
      </div>

      <button
        onClick={getRecommendationsHandler}
        disabled={!isModelTrained || loading || selectedItems.length === 0}
        className={`px-4 py-2 text-white rounded-md mb-4 ${
          !isModelTrained || loading || selectedItems.length === 0
            ? 'bg-gray-400 cursor-not-allowed'
            : 'bg-green-600 hover:bg-green-700'
        }`}
      >
        {loading ? 'Getting Recommendations...' : 'Get Recommendations'}
      </button>

      {recommendationResults && (
        <div className="mt-4">
          <h3 className="font-medium text-lg mb-2">Recommendations:</h3>
          {recommendationResults.recommendations.length > 0 ? (
            <div className="bg-gray-50 p-4 rounded-md">
              <h4 className="font-medium mb-2">Based on your selection, we recommend:</h4>
              <div className="space-y-2">
                {recommendationResults.recommendations.map((rec: any, index: number) => (
                  <div key={index} className="border-l-4 border-green-500 pl-3 py-2">
                    <div className="flex flex-wrap gap-1 mb-1">
                      {rec.items.map((item: string, i: number) => (
                        <span key={i} className="inline-block bg-green-100 text-green-800 text-xs px-2 py-1 rounded">
                          {item}
                        </span>
                      ))}
                    </div>
                    <p className="text-sm text-gray-600">
                      Confidence: {(rec.confidence * 100).toFixed(2)}%
                    </p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="bg-gray-50 p-4 rounded-md text-gray-700">
              No recommendations found for these items. Try selecting different products.
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Recommendations;