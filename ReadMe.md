# SuperMarket Basket Analysis Application

This project provides a web application for market basket analysis to find frequent itemsets and association rules in retail transaction data. The application consists of a Python FastAPI backend and a Next.js TypeScript frontend.

## Project Structure

- `SuperMarketBasketAnalysisBackend/`: FastAPI backend server
  - `models/`: Data models and analysis logic 
  - `routers/`: API endpoints
  - `requirements.txt`: Python dependencies
  - `main.py`: Server startup script

- `SuperMarketBasketAnalysisFrontend/`: Next.js TypeScript frontend
  - `src/`: Source code
    - `app/`: Next.js app directory
    - `components/`: React components
    - `utils/`: Utility functions

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd SuperMarketBasketAnalysisBackend
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:
   ```
   python main.py
   ```

   The API will be available at http://localhost:8000

   API documentation will be available at http://localhost:8000/docs

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd SuperMarketBasketAnalysisFrontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Run the development server:
   ```
   npm run dev
   ```

   The application will be available at http://localhost:3000

## Usage Guide

1. **Upload Data**
   - On the homepage, upload your retail transaction data (Excel file)
   - The Excel file should have columns like: InvoiceNo, CustomerID, Description, Quantity

2. **Process Data**
   - After successful upload, click "Process Data" to clean and prepare the data
   - This step removes nulls, duplicates, and prepares transaction groups

3. **Train Model**
   - Adjust the minimum support and confidence parameters
   - Click "Train Model" to run the FP-Growth algorithm
   - Review the frequent itemsets and association rules

4. **View Visualization**
   - The application will display a chart of the top frequent itemsets

5. **Get Recommendations**
   - Enter products in the recommendation panel
   - Click "Get Recommendations" to see product suggestions based on association rules

## Sample Data

You can use the provided "Online Retail.xlsx" file as sample data, which contains retail transaction records.

## Technologies Used

- **Backend**:
  - Python
  - FastAPI
  - Pandas
  - Matplotlib
  - NumPy

- **Frontend**:
  - Next.js
  - TypeScript
  - React
  - Tailwind CSS
  - Axios
  - Chart.js
  - React Chart.js 2
  - React Hook Form

## API Endpoints

- `POST /api/upload`: Upload retail data file
- `POST /api/process`: Process and prepare data
- `POST /api/train`: Train market basket analysis model
- `GET /api/visualization`: Get visualization of frequent itemsets
- `POST /api/recommend`: Get product recommendations
- `POST /api/reset`: Reset the model and start over
