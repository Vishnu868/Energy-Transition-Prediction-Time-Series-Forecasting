"""
File upload route — absolute imports only.
"""

import os
import sys
import uuid
import shutil

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi import APIRouter, UploadFile, File, HTTPException
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Accept a CSV upload, save it, return session_id."""

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    session_id = str(uuid.uuid4())[:8]
    save_path  = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with open(save_path, 'r', errors='ignore') as f:
            row_count = sum(1 for _ in f) - 1

        logger.info(f"Uploaded: {file.filename} -> {save_path} ({row_count} rows)")

        return {
            "success":        True,
            "session_id":     session_id,
            "filename":       file.filename,
            "saved_path":     save_path,
            "estimated_rows": row_count
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")