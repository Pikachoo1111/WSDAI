"""
Configuration settings for the WSD AI Judge system.
"""
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    model_config = {"extra": "allow", "env_file": ".env"}

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
    
    # WSD Rubric Settings (Official World Schools Debate Scoring)
    # Main speeches: 60-80 points
    MAIN_SPEECH_MIN_SCORE: int = 60
    MAIN_SPEECH_MAX_SCORE: int = 80
    # Reply speeches: 30-40 points
    REPLY_SPEECH_MIN_SCORE: int = 30
    REPLY_SPEECH_MAX_SCORE: int = 40

    # Official WSD Criteria Weights
    STYLE_WEIGHT: float = 0.4    # Communication, delivery, use of notes
    CONTENT_WEIGHT: float = 0.4  # Argumentation quality
    STRATEGY_WEIGHT: float = 0.2 # Issue prioritization, timing, structure
    
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


# Global settings instance
settings = Settings()


# Official World Schools Debate Rubric Configuration
WSD_RUBRIC = {
    "style": {
        "weight": settings.STYLE_WEIGHT,  # 40%
        "description": "Communication clarity using effective rate, pitch, tone, hand gestures, facial expressions. Notes for reference only, not reading.",
        "criteria": [
            "vocal_delivery",      # Rate, pitch, tone
            "physical_delivery",   # Hand gestures, facial expressions, posture
            "eye_contact",         # Audience engagement, not over-reliant on notes
            "clarity",             # Clear articulation and communication
            "note_usage"           # Appropriate use of notes for reference only
        ]
    },
    "content": {
        "weight": settings.CONTENT_WEIGHT,  # 40%
        "description": "Argumentation quality divorced from style. Weak arguments marked regardless of opponent response.",
        "criteria": [
            "argument_strength",   # Quality of individual arguments
            "evidence_quality",    # Supporting evidence and examples
            "logical_reasoning",   # Sound logical connections
            "analysis_depth",      # Depth of analysis and explanation
            "factual_accuracy"     # Accuracy of claims and evidence
        ]
    },
    "strategy": {
        "weight": settings.STRATEGY_WEIGHT,  # 20%
        "description": "Understanding issue importance, structure/timing, addressing critical issues appropriately.",
        "criteria": [
            "issue_prioritization", # Identifying most substantive issues
            "time_allocation",      # Allocating time based on issue importance
            "structural_choices",   # Speech organization and timing
            "poi_handling",         # Points of Information responses
            "strategic_focus"       # Overall strategic approach to debate
        ]
    }
}

# Official WSD Speaker Point Ranges
WSD_SPEAKER_POINT_RANGES = {
    "main_speech": {
        "exceptional": {"min": 78, "max": 80, "description": "Outstanding performance across all criteria"},
        "very_strong": {"min": 75, "max": 77, "description": "Strong performance with minor weaknesses"},
        "good": {"min": 72, "max": 74, "description": "Solid performance meeting expectations"},
        "satisfactory": {"min": 68, "max": 71, "description": "Adequate performance with clear areas for improvement"},
        "weak": {"min": 64, "max": 67, "description": "Below average performance with significant issues"},
        "poor": {"min": 60, "max": 63, "description": "Weak performance requiring substantial development"}
    },
    "reply_speech": {
        "exceptional": {"min": 39, "max": 40, "description": "Outstanding reply speech"},
        "very_strong": {"min": 37, "max": 38, "description": "Strong reply with minor weaknesses"},
        "good": {"min": 36, "max": 36, "description": "Solid reply meeting expectations"},
        "satisfactory": {"min": 34, "max": 35, "description": "Adequate reply with room for improvement"},
        "weak": {"min": 32, "max": 33, "description": "Below average reply"},
        "poor": {"min": 30, "max": 31, "description": "Weak reply requiring development"}
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
