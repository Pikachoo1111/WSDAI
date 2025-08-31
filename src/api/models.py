"""
Pydantic models for API requests and responses.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class SpeakerRoleEnum(str, Enum):
    """Speaker roles in World Schools Debate."""
    FIRST_PROP = "first_proposition"
    FIRST_OPP = "first_opposition"
    SECOND_PROP = "second_proposition"
    SECOND_OPP = "second_opposition"
    THIRD_PROP = "third_proposition"
    THIRD_OPP = "third_opposition"


class VideoUploadRequest(BaseModel):
    """Request model for video upload."""
    speaker_name: str = Field(..., description="Name of the speaker")
    speaker_role: SpeakerRoleEnum = Field(..., description="Speaker's role in the debate")
    debate_topic: str = Field(..., description="Topic of the debate")
    team_side: str = Field(..., description="Proposition or Opposition")


class TranscriptSegmentResponse(BaseModel):
    """Response model for transcript segments."""
    text: str
    start_time: float
    end_time: float
    confidence: float
    speaker_id: Optional[str] = None
    is_clear: bool
    words_per_minute: float


class STTAnalysisResponse(BaseModel):
    """Response model for STT analysis."""
    full_transcript: str
    segments: List[TranscriptSegmentResponse]
    clarity_score: float
    average_wpm: float
    total_duration: float
    filler_word_count: int
    pause_analysis: Dict[str, float]


class SentimentAnalysisResponse(BaseModel):
    """Response model for sentiment analysis."""
    confidence_score: float
    persuasiveness_score: float
    emotional_impact: float
    sentiment_label: str
    sentiment_confidence: float


class EyeContactResponse(BaseModel):
    """Response model for eye contact analysis."""
    audience_focus_percentage: float
    note_reliance_percentage: float
    eye_contact_consistency: float
    engagement_score: float


class PostureResponse(BaseModel):
    """Response model for posture analysis."""
    posture_stability: float
    confidence_indicators: float
    engagement_level: float
    professional_appearance: float


class GestureResponse(BaseModel):
    """Response model for gesture analysis."""
    gesture_frequency: float
    gesture_variety: float
    appropriate_gestures: float
    distracting_movements: float


class StyleAnalysisResponse(BaseModel):
    """Response model for style analysis."""
    eye_contact: EyeContactResponse
    posture: PostureResponse
    gestures: GestureResponse
    overall_engagement: float
    visual_confidence: float
    professionalism_score: float


class RubricScoreResponse(BaseModel):
    """Response model for individual rubric scores."""
    category: str
    criterion: str
    score: float
    max_score: float
    feedback: str


class WSDScoreResponse(BaseModel):
    """Response model for complete WSD evaluation."""
    matter_score: float
    manner_score: float
    method_score: float
    total_score: float
    rubric_scores: List[RubricScoreResponse]
    overall_feedback: str
    strengths: List[str]
    improvements: List[str]


class AnalysisResultResponse(BaseModel):
    """Complete analysis result response."""
    analysis_id: str
    speaker_name: str
    speaker_role: SpeakerRoleEnum
    debate_topic: str
    team_side: str
    timestamp: datetime
    
    # Analysis components
    stt_analysis: STTAnalysisResponse
    sentiment_analysis: SentimentAnalysisResponse
    style_analysis: StyleAnalysisResponse
    wsd_score: WSDScoreResponse
    
    # Processing metadata
    processing_time: float
    video_duration: float
    file_size: int


class AnalysisStatusResponse(BaseModel):
    """Response model for analysis status."""
    analysis_id: str
    status: str  # "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    estimated_completion: Optional[datetime] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, str]


class AnalysisListResponse(BaseModel):
    """Response for listing analyses."""
    analyses: List[AnalysisResultResponse]
    total_count: int
    page: int
    page_size: int


class ComparisonRequest(BaseModel):
    """Request for comparing multiple speeches."""
    analysis_ids: List[str] = Field(..., min_items=2, max_items=6)
    comparison_criteria: List[str] = Field(default=["matter", "manner", "method"])


class ComparisonResponse(BaseModel):
    """Response for speech comparison."""
    comparison_id: str
    analyses: List[AnalysisResultResponse]
    comparison_summary: Dict[str, Any]
    rankings: Dict[str, List[str]]  # Rankings by different criteria
    insights: List[str]


class FeedbackRequest(BaseModel):
    """Request for detailed feedback generation."""
    analysis_id: str
    focus_areas: Optional[List[str]] = None
    feedback_level: str = Field(default="detailed", regex="^(brief|detailed|comprehensive)$")


class DetailedFeedbackResponse(BaseModel):
    """Detailed feedback response."""
    analysis_id: str
    feedback_level: str
    
    # Detailed feedback by category
    matter_feedback: Dict[str, str]
    manner_feedback: Dict[str, str]
    method_feedback: Dict[str, str]
    
    # Actionable recommendations
    immediate_improvements: List[str]
    practice_exercises: List[str]
    long_term_goals: List[str]
    
    # Performance trends (if multiple analyses available)
    progress_indicators: Optional[Dict[str, float]] = None


class AnalyticsRequest(BaseModel):
    """Request for analytics dashboard data."""
    speaker_name: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    speaker_roles: Optional[List[SpeakerRoleEnum]] = None


class AnalyticsResponse(BaseModel):
    """Analytics dashboard response."""
    summary_stats: Dict[str, float]
    score_trends: Dict[str, List[float]]
    improvement_areas: List[str]
    strengths: List[str]
    performance_by_role: Dict[str, Dict[str, float]]
    recommendations: List[str]
