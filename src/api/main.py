"""
FastAPI main application for WSD AI Judge system.
"""
import os
import uuid
import asyncio
from datetime import datetime
from typing import List, Optional
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config.settings import settings
from src.api.models import (
    VideoUploadRequest, AnalysisResultResponse, AnalysisStatusResponse,
    ErrorResponse, HealthCheckResponse, SpeakerRoleEnum
)
from src.core.processor import DebateAnalysisProcessor
from src.core.storage import AnalysisStorage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="WSD AI Judge",
    description="AI-powered World Schools Debate judge and feedback system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
processor = DebateAnalysisProcessor()
storage = AnalysisStorage()

# In-memory storage for analysis status (use Redis in production)
analysis_status = {}


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting WSD AI Judge system...")
    
    # Create upload directory
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # Initialize storage
    await storage.initialize()
    
    logger.info("WSD AI Judge system started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down WSD AI Judge system...")
    await storage.close()


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        components={
            "api": "healthy",
            "storage": "healthy",
            "processor": "healthy"
        }
    )


@app.post("/upload", response_model=dict)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    speaker_name: str = Form(...),
    speaker_role: SpeakerRoleEnum = Form(...),
    debate_topic: str = Form(...),
    team_side: str = Form(...)
):
    """
    Upload video for analysis.
    
    Args:
        file: Video file
        speaker_name: Name of the speaker
        speaker_role: Speaker's role in the debate
        debate_topic: Topic of the debate
        team_side: Proposition or Opposition
        
    Returns:
        Analysis ID and status
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.0f}MB"
            )
        
        # Check file format
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in settings.ALLOWED_VIDEO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed: {', '.join(settings.ALLOWED_VIDEO_FORMATS)}"
            )
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = os.path.join(settings.UPLOAD_DIR, f"{analysis_id}.{file_extension}")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Initialize analysis status
        analysis_status[analysis_id] = {
            "status": "processing",
            "progress": 0.0,
            "message": "Analysis started",
            "timestamp": datetime.now()
        }
        
        # Create analysis request
        request_data = {
            "analysis_id": analysis_id,
            "file_path": file_path,
            "speaker_name": speaker_name,
            "speaker_role": speaker_role,
            "debate_topic": debate_topic,
            "team_side": team_side,
            "file_size": len(content)
        }
        
        # Start background processing
        background_tasks.add_task(process_video_analysis, request_data)
        
        return {
            "analysis_id": analysis_id,
            "status": "processing",
            "message": "Video uploaded successfully. Analysis started."
        }
        
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_video_analysis(request_data: dict):
    """
    Background task to process video analysis.
    
    Args:
        request_data: Analysis request data
    """
    analysis_id = request_data["analysis_id"]
    
    try:
        # Update status
        analysis_status[analysis_id].update({
            "progress": 0.1,
            "message": "Extracting audio from video..."
        })
        
        # Process the video
        result = await processor.process_video(
            video_path=request_data["file_path"],
            speaker_name=request_data["speaker_name"],
            speaker_role=request_data["speaker_role"],
            debate_topic=request_data["debate_topic"],
            team_side=request_data["team_side"],
            progress_callback=lambda p, m: update_analysis_progress(analysis_id, p, m)
        )
        
        # Store result
        await storage.store_analysis(analysis_id, result)
        
        # Update final status
        analysis_status[analysis_id].update({
            "status": "completed",
            "progress": 1.0,
            "message": "Analysis completed successfully"
        })
        
        # Clean up uploaded file
        try:
            os.remove(request_data["file_path"])
        except:
            pass  # File cleanup is not critical
            
    except Exception as e:
        logger.error(f"Error processing analysis {analysis_id}: {str(e)}")
        analysis_status[analysis_id].update({
            "status": "failed",
            "message": f"Analysis failed: {str(e)}"
        })


def update_analysis_progress(analysis_id: str, progress: float, message: str):
    """Update analysis progress."""
    if analysis_id in analysis_status:
        analysis_status[analysis_id].update({
            "progress": progress,
            "message": message
        })


@app.get("/analysis/{analysis_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(analysis_id: str):
    """
    Get analysis status.
    
    Args:
        analysis_id: Analysis ID
        
    Returns:
        Analysis status
    """
    if analysis_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    status_data = analysis_status[analysis_id]
    
    return AnalysisStatusResponse(
        analysis_id=analysis_id,
        status=status_data["status"],
        progress=status_data["progress"],
        message=status_data["message"],
        estimated_completion=None  # Could implement ETA calculation
    )


@app.get("/analysis/{analysis_id}", response_model=AnalysisResultResponse)
async def get_analysis_result(analysis_id: str):
    """
    Get complete analysis result.
    
    Args:
        analysis_id: Analysis ID
        
    Returns:
        Complete analysis result
    """
    # Check if analysis exists and is completed
    if analysis_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    status_data = analysis_status[analysis_id]
    if status_data["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Analysis not completed. Status: {status_data['status']}"
        )
    
    # Retrieve result from storage
    result = await storage.get_analysis(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    
    return result


@app.get("/analysis", response_model=List[AnalysisResultResponse])
async def list_analyses(
    speaker_name: Optional[str] = None,
    speaker_role: Optional[SpeakerRoleEnum] = None,
    limit: int = 10,
    offset: int = 0
):
    """
    List analyses with optional filtering.
    
    Args:
        speaker_name: Filter by speaker name
        speaker_role: Filter by speaker role
        limit: Maximum number of results
        offset: Offset for pagination
        
    Returns:
        List of analyses
    """
    analyses = await storage.list_analyses(
        speaker_name=speaker_name,
        speaker_role=speaker_role,
        limit=limit,
        offset=offset
    )
    
    return analyses


@app.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """
    Delete an analysis.
    
    Args:
        analysis_id: Analysis ID
        
    Returns:
        Success message
    """
    # Remove from status tracking
    if analysis_id in analysis_status:
        del analysis_status[analysis_id]
    
    # Remove from storage
    success = await storage.delete_analysis(analysis_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return {"message": "Analysis deleted successfully"}


@app.get("/analytics/summary")
async def get_analytics_summary(
    speaker_name: Optional[str] = None,
    days: int = 30
):
    """
    Get analytics summary.
    
    Args:
        speaker_name: Filter by speaker name
        days: Number of days to include
        
    Returns:
        Analytics summary
    """
    summary = await storage.get_analytics_summary(
        speaker_name=speaker_name,
        days=days
    )
    
    return summary


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An internal server error occurred"
        ).dict()
    )


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
