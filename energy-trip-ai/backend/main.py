"""
FastAPI application entry point.
Uses absolute imports — compatible with uvicorn running from backend/ directory.
"""

import sys
import os

# Ensure backend/ is on the path so absolute imports work
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Absolute imports (no leading dots)
from routes.upload import router as upload_router
from routes.predict import router as predict_router
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Energy Trip Prediction System",
    description="AI-powered early power trip and appliance fault prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api", tags=["Upload"])
app.include_router(predict_router, prefix="/api", tags=["Predict"])

# Serve frontend static files if folder exists
frontend_path = os.path.join(os.path.dirname(BACKEND_DIR), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Energy Trip Prediction API"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Energy Trip Prediction API...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)