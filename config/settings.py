"""
Configuration settings for the WSD AI Judge system.
"""
import os
from typing import Dict, List
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://user:password@localhost/wsdai"
    
    # Model Configurations
    WHISPER_MODEL: str = "base"  # tiny, base, small, medium, large
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # Speech Processing Settings
    SAMPLE_RATE: int = 16000
    CHUNK_SIZE: int = 1024
    SILENCE_THRESHOLD: float = 0.01
    MIN_SPEECH_DURATION: float = 0.5
    
    # Video Processing Settings
    VIDEO_FPS: int = 30
    FACE_DETECTION_CONFIDENCE: float = 0.5
    EYE_CONTACT_THRESHOLD: float = 0.7
    
    # WSD Rubric Settings
    MAX_SCORE: int = 100
    MATTER_WEIGHT: float = 0.4
    MANNER_WEIGHT: float = 0.3
    METHOD_WEIGHT: float = 0.3
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
    ALLOWED_VIDEO_FORMATS: List[str] = ["mp4", "avi", "mov", "mkv"]
    UPLOAD_DIR: str = "uploads"
    
    # Debate-specific vocabulary for STT
    DEBATE_VOCABULARY: List[str] = [
        "comparative", "principle clash", "burden of proof", "status quo",
        "proposition", "opposition", "rebuttal", "substantive", "POI",
        "point of information", "signposting", "weighing", "impact",
        "mechanism", "stakeholder", "framework", "definitional challenge"
    ]
    
    class Config:
        env_file = ".env"


# Global settings instance
settings = Settings()


# WSD Rubric Configuration
WSD_RUBRIC = {
    "matter": {
        "weight": settings.MATTER_WEIGHT,
        "criteria": [
            "argument_quality",
            "evidence_strength", 
            "logical_consistency",
            "clash_engagement",
            "analysis_depth"
        ]
    },
    "manner": {
        "weight": settings.MANNER_WEIGHT,
        "criteria": [
            "clarity",
            "persuasiveness",
            "audience_engagement",
            "confidence",
            "vocal_variety"
        ]
    },
    "method": {
        "weight": settings.METHOD_WEIGHT,
        "criteria": [
            "structure",
            "signposting",
            "time_management",
            "role_fulfillment",
            "strategic_approach"
        ]
    }
}

# Feedback templates
FEEDBACK_TEMPLATES = {
    "strengths": {
        "matter": [
            "Strong analytical depth in {topic}",
            "Excellent use of evidence to support {argument}",
            "Clear logical progression in reasoning",
            "Effective engagement with opposition arguments"
        ],
        "manner": [
            "Confident and persuasive delivery",
            "Excellent vocal variety and emphasis",
            "Strong audience connection and eye contact",
            "Clear articulation throughout"
        ],
        "method": [
            "Well-structured speech with clear signposting",
            "Excellent time management",
            "Strong fulfillment of speaker role",
            "Strategic approach to debate"
        ]
    },
    "improvements": {
        "matter": [
            "Consider deeper analysis of {topic}",
            "Strengthen evidence for {argument}",
            "Address opposition arguments more directly",
            "Develop clearer causal links"
        ],
        "manner": [
            "Reduce reliance on notes for better audience connection",
            "Vary pace to emphasize key points",
            "Increase vocal confidence in rebuttals",
            "Improve clarity in rapid sections"
        ],
        "method": [
            "Clearer signposting between arguments",
            "Better time allocation across points",
            "Stronger conclusion summarizing key clash",
            "More strategic prioritization of arguments"
        ]
    }
}
