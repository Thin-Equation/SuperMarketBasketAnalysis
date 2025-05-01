'use client';

import React, { useState, useEffect } from 'react';
import FileUpload from '../components/FileUpload';
import DataProcessing from '../components/DataProcessing';
import ModelTraining from '../components/ModelTraining';
import Visualization from '../components/Visualization';
import Recommendations from '../components/Recommendations';

export default function Home() {
  const [error, setError] = useState<string | null>(null);
  const [isDataLoaded, setIsDataLoaded] = useState(false);
  const [isDataProcessed, setIsDataProcessed] = useState(false);
  const [isModelTrained, setIsModelTrained] = useState(false);
  const [availableItems, setAvailableItems] = useState<string[]>([]);
  const [workflowStep, setWorkflowStep] = useState(1);

  useEffect(() => {
    const timer = setTimeout(() => {
      setError(null);
    }, 5000);

    return () => clearTimeout(timer);
  }, [error]);

  const handleError = (message: string) => {
    setError(message);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleUploadSuccess = (data: any) => {
    setIsDataLoaded(true);
    setWorkflowStep(2);
  };

  const handleProcessSuccess = (data: any) => {
    setIsDataProcessed(true);
    setWorkflowStep(3);
  };

  const handleTrainSuccess = (data: any) => {
    setIsModelTrained(true);
    
    // Extract unique items from frequent itemsets for recommendation search
    if (data.sample_frequent_itemsets) {
      const items = new Set<string>();
      data.sample_frequent_itemsets.forEach((itemset: any) => {
        if (itemset.items && Array.isArray(itemset.items)) {
          itemset.items.forEach((item: string) => items.add(item));
        }
      });
      setAvailableItems(Array.from(items));
    }
    
    setWorkflowStep(4);
  };

  return (
    <main className="min-h-screen bg-gray-100 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-2 text-center">
          SuperMarket Basket Analysis
        </h1>
        <p className="text-center mb-8 text-gray-600">
          Upload retail data, analyze patterns, and get product recommendations
        </p>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
            <strong className="font-bold">Error! </strong>
            <span className="block sm:inline">{error}</span>
          </div>
        )}

        <div className="mb-8">
          <div className="relative">
            <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-gray-200">
              <div 
                className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-blue-500"
                style={{ width: `${Math.min(100, workflowStep * 25)}%` }}
              ></div>
            </div>
            <div className="flex justify-between">
              <div className={`text-xs ${workflowStep >= 1 ? 'text-blue-600 font-semibold' : 'text-gray-500'}`}>Upload Data</div>
              <div className={`text-xs ${workflowStep >= 2 ? 'text-blue-600 font-semibold' : 'text-gray-500'}`}>Process Data</div>
              <div className={`text-xs ${workflowStep >= 3 ? 'text-blue-600 font-semibold' : 'text-gray-500'}`}>Train Model</div>
              <div className={`text-xs ${workflowStep >= 4 ? 'text-blue-600 font-semibold' : 'text-gray-500'}`}>Get Insights</div>
            </div>
          </div>
        </div>

        <FileUpload 
          onUploadSuccess={handleUploadSuccess} 
          onUploadError={handleError} 
        />

        {isDataLoaded && (
          <DataProcessing 
            isDataLoaded={isDataLoaded}
            onProcessSuccess={handleProcessSuccess}
            onProcessError={handleError}
          />
        )}

        {isDataProcessed && (
          <ModelTraining 
            isDataProcessed={isDataProcessed}
            onTrainSuccess={handleTrainSuccess}
            onTrainError={handleError}
          />
        )}

        {isModelTrained && (
          <>
            <Visualization 
              isModelTrained={isModelTrained} 
              onError={handleError} 
            />
            
            <Recommendations 
              isModelTrained={isModelTrained}
              onError={handleError}
              availableItems={availableItems}
            />
          </>
        )}
      </div>
    </main>
  );
}
