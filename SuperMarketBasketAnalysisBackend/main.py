from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import recommendations
import uvicorn

app = FastAPI(title="SuperMarket Basket Analysis API", 
              description="API for market basket analysis and product recommendations",
              version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(recommendations.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to SuperMarket Basket Analysis API"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)