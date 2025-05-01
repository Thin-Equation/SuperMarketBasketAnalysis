from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
import io
from models.market_basket import MarketBasketAnalysis

router = APIRouter(prefix="/api", tags=["recommendations"])

# Create a global instance of MarketBasketAnalysis
mba = MarketBasketAnalysis()

# Pydantic models for request/response validation
class RecommendRequest(BaseModel):
    items: List[str]
    recommendations_count: Optional[int] = 3

class FPGrowthParams(BaseModel):
    min_support: Optional[float] = 0.02
    min_confidence: Optional[float] = 0.2

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload retail data file (Excel format)"""
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are allowed")
    
    content = await file.read()
    result = mba.load_data(content)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@router.post("/process")
async def process_data():
    """Process the uploaded data: clean and prepare for analysis"""
    # Step 1: Clean data
    clean_result = mba.clean_data()
    if "error" in clean_result:
        raise HTTPException(status_code=400, detail=clean_result["error"])
    
    # Step 2: Prepare transactions
    prep_result = mba.prepare_transactions()
    if "error" in prep_result:
        raise HTTPException(status_code=400, detail=prep_result["error"])
    
    return {
        "cleaning": clean_result,
        "preparation": prep_result
    }

@router.post("/train")
async def train_model(params: FPGrowthParams = Depends()):
    """Train FP-Growth model with the specified parameters"""
    result = mba.apply_fp_growth(min_support=params.min_support, min_confidence=params.min_confidence)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@router.get("/visualization")
async def get_visualization():
    """Get visualization of frequent itemsets"""
    result = mba.visualize_frequent_itemsets()
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@router.post("/recommend")
async def recommend_products(request: RecommendRequest):
    """Recommend products based on purchased items"""
    result = mba.recommend_products(
        purchased_items=request.items,
        n_recommendations=request.recommendations_count
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@router.post("/reset")
async def reset_model():
    """Reset the model and start over"""
    global mba  # Declare global first, before using it
    
    # Stop the current Spark session
    mba.shutdown()
    
    # Create a new instance
    mba = MarketBasketAnalysis()
    
    return {"message": "Model reset successfully"}