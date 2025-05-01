import React, { useState } from 'react';
import { processData } from '../utils/api';

interface DataProcessingProps {
  isDataLoaded: boolean;
  onProcessSuccess: (data: any) => void;
  onProcessError: (error: string) => void;
}

const DataProcessing: React.FC<DataProcessingProps> = ({
  isDataLoaded,
  onProcessSuccess,
  onProcessError,
}) => {
  const [processing, setProcessing] = useState(false);
  const [processingResult, setProcessingResult] = useState<any>(null);

  const handleProcessData = async () => {
    if (!isDataLoaded) {
      onProcessError('Please upload data first');
      return;
    }

    try {
      setProcessing(true);
      const result = await processData();
      setProcessingResult(result);
      onProcessSuccess(result);
    } catch (error: any) {
      onProcessError(error.response?.data?.detail || 'Processing failed');
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-md mt-4">
      <h2 className="text-xl font-semibold mb-4">Process Data</h2>
      <p className="text-sm text-gray-600 mb-4">
        Clean data by removing nulls, duplicates, and preparing transactions for analysis.
      </p>
      
      <button
        onClick={handleProcessData}
        disabled={!isDataLoaded || processing}
        className={`px-4 py-2 text-white rounded-md mb-4 ${
          !isDataLoaded || processing ? 'bg-gray-400 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700'
        }`}
      >
        {processing ? 'Processing...' : 'Process Data'}
      </button>

      {processingResult && (
        <div className="mt-4">
          <h3 className="font-medium text-lg mb-2">Processing Results:</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-50 p-4 rounded-md">
              <h4 className="font-medium mb-2">Data Cleaning:</h4>
              <p>Initial count: {processingResult.cleaning.initial_count}</p>
              <p>Final count: {processingResult.cleaning.final_count}</p>
              <p>Removed: {processingResult.cleaning.removed_count}</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-md">
              <h4 className="font-medium mb-2">Transaction Preparation:</h4>
              <p>Transactions: {processingResult.preparation.transaction_count}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataProcessing;