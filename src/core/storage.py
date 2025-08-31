"""
Storage layer for analysis results and metadata.
"""
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from config.settings import settings
from src.api.models import AnalysisResultResponse, SpeakerRoleEnum

logger = logging.getLogger(__name__)

Base = declarative_base()


class AnalysisRecord(Base):
    """Database model for analysis records."""
    __tablename__ = "analyses"
    
    analysis_id = Column(String, primary_key=True)
    speaker_name = Column(String, nullable=False)
    speaker_role = Column(String, nullable=False)
    debate_topic = Column(String, nullable=False)
    team_side = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    
    # Scores
    matter_score = Column(Float, nullable=False)
    manner_score = Column(Float, nullable=False)
    method_score = Column(Float, nullable=False)
    total_score = Column(Float, nullable=False)
    
    # Analysis data (stored as JSON)
    stt_analysis = Column(JSON, nullable=False)
    sentiment_analysis = Column(JSON, nullable=False)
    style_analysis = Column(JSON, nullable=False)
    wsd_score = Column(JSON, nullable=False)
    
    # Metadata
    processing_time = Column(Float, nullable=False)
    video_duration = Column(Float, nullable=False)
    file_size = Column(Integer, nullable=False)
    
    # Feedback and insights
    overall_feedback = Column(Text)
    strengths = Column(JSON)
    improvements = Column(JSON)


class AnalysisStorage:
    """
    Storage manager for analysis results.
    """
    
    def __init__(self):
        """Initialize storage manager."""
        self.engine = None
        self.SessionLocal = None
        
    async def initialize(self):
        """Initialize database connection."""
        try:
            # Create engine
            self.engine = create_engine(
                settings.DATABASE_URL,
                echo=settings.DEBUG,
                pool_pre_ping=True
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    async def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
    
    def get_session(self) -> Session:
        """Get database session."""
        if not self.SessionLocal:
            raise RuntimeError("Storage not initialized")
        return self.SessionLocal()
    
    async def store_analysis(self, analysis_id: str, result: AnalysisResultResponse) -> bool:
        """
        Store analysis result in database.
        
        Args:
            analysis_id: Analysis ID
            result: Analysis result to store
            
        Returns:
            Success status
        """
        try:
            session = self.get_session()
            
            # Create record
            record = AnalysisRecord(
                analysis_id=analysis_id,
                speaker_name=result.speaker_name,
                speaker_role=result.speaker_role.value,
                debate_topic=result.debate_topic,
                team_side=result.team_side,
                timestamp=result.timestamp,
                matter_score=result.wsd_score.matter_score,
                manner_score=result.wsd_score.manner_score,
                method_score=result.wsd_score.method_score,
                total_score=result.wsd_score.total_score,
                stt_analysis=result.stt_analysis.dict(),
                sentiment_analysis=result.sentiment_analysis.dict(),
                style_analysis=result.style_analysis.dict(),
                wsd_score=result.wsd_score.dict(),
                processing_time=result.processing_time,
                video_duration=result.video_duration,
                file_size=result.file_size,
                overall_feedback=result.wsd_score.overall_feedback,
                strengths=result.wsd_score.strengths,
                improvements=result.wsd_score.improvements
            )
            
            # Save to database
            session.add(record)
            session.commit()
            session.close()
            
            logger.info(f"Analysis {analysis_id} stored successfully")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Database error storing analysis {analysis_id}: {str(e)}")
            session.rollback()
            session.close()
            return False
        except Exception as e:
            logger.error(f"Error storing analysis {analysis_id}: {str(e)}")
            return False
    
    async def get_analysis(self, analysis_id: str) -> Optional[AnalysisResultResponse]:
        """
        Retrieve analysis result from database.
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            Analysis result or None if not found
        """
        try:
            session = self.get_session()
            
            record = session.query(AnalysisRecord).filter(
                AnalysisRecord.analysis_id == analysis_id
            ).first()
            
            session.close()
            
            if not record:
                return None
            
            # Convert back to response model
            result = AnalysisResultResponse(
                analysis_id=record.analysis_id,
                speaker_name=record.speaker_name,
                speaker_role=SpeakerRoleEnum(record.speaker_role),
                debate_topic=record.debate_topic,
                team_side=record.team_side,
                timestamp=record.timestamp,
                stt_analysis=record.stt_analysis,
                sentiment_analysis=record.sentiment_analysis,
                style_analysis=record.style_analysis,
                wsd_score=record.wsd_score,
                processing_time=record.processing_time,
                video_duration=record.video_duration,
                file_size=record.file_size
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving analysis {analysis_id}: {str(e)}")
            return None
    
    async def list_analyses(
        self,
        speaker_name: Optional[str] = None,
        speaker_role: Optional[SpeakerRoleEnum] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[AnalysisResultResponse]:
        """
        List analyses with optional filtering.
        
        Args:
            speaker_name: Filter by speaker name
            speaker_role: Filter by speaker role
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of analysis results
        """
        try:
            session = self.get_session()
            
            query = session.query(AnalysisRecord)
            
            # Apply filters
            if speaker_name:
                query = query.filter(AnalysisRecord.speaker_name.ilike(f"%{speaker_name}%"))
            
            if speaker_role:
                query = query.filter(AnalysisRecord.speaker_role == speaker_role.value)
            
            # Apply pagination and ordering
            records = query.order_by(AnalysisRecord.timestamp.desc()).offset(offset).limit(limit).all()
            
            session.close()
            
            # Convert to response models
            results = []
            for record in records:
                result = AnalysisResultResponse(
                    analysis_id=record.analysis_id,
                    speaker_name=record.speaker_name,
                    speaker_role=SpeakerRoleEnum(record.speaker_role),
                    debate_topic=record.debate_topic,
                    team_side=record.team_side,
                    timestamp=record.timestamp,
                    stt_analysis=record.stt_analysis,
                    sentiment_analysis=record.sentiment_analysis,
                    style_analysis=record.style_analysis,
                    wsd_score=record.wsd_score,
                    processing_time=record.processing_time,
                    video_duration=record.video_duration,
                    file_size=record.file_size
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error listing analyses: {str(e)}")
            return []
    
    async def delete_analysis(self, analysis_id: str) -> bool:
        """
        Delete analysis from database.
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            Success status
        """
        try:
            session = self.get_session()
            
            record = session.query(AnalysisRecord).filter(
                AnalysisRecord.analysis_id == analysis_id
            ).first()
            
            if record:
                session.delete(record)
                session.commit()
                session.close()
                logger.info(f"Analysis {analysis_id} deleted successfully")
                return True
            else:
                session.close()
                return False
                
        except SQLAlchemyError as e:
            logger.error(f"Database error deleting analysis {analysis_id}: {str(e)}")
            session.rollback()
            session.close()
            return False
        except Exception as e:
            logger.error(f"Error deleting analysis {analysis_id}: {str(e)}")
            return False
    
    async def get_analytics_summary(
        self,
        speaker_name: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get analytics summary for dashboard.
        
        Args:
            speaker_name: Filter by speaker name
            days: Number of days to include
            
        Returns:
            Analytics summary
        """
        try:
            session = self.get_session()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            query = session.query(AnalysisRecord).filter(
                AnalysisRecord.timestamp >= start_date
            )
            
            if speaker_name:
                query = query.filter(AnalysisRecord.speaker_name.ilike(f"%{speaker_name}%"))
            
            records = query.all()
            session.close()
            
            if not records:
                return {
                    "total_analyses": 0,
                    "average_scores": {},
                    "score_trends": {},
                    "common_strengths": [],
                    "common_improvements": []
                }
            
            # Calculate summary statistics
            total_analyses = len(records)
            
            # Average scores
            avg_matter = sum(r.matter_score for r in records) / total_analyses
            avg_manner = sum(r.manner_score for r in records) / total_analyses
            avg_method = sum(r.method_score for r in records) / total_analyses
            avg_total = sum(r.total_score for r in records) / total_analyses
            
            # Score trends (simplified - by week)
            score_trends = self._calculate_score_trends(records)
            
            # Common feedback themes
            all_strengths = []
            all_improvements = []
            
            for record in records:
                if record.strengths:
                    all_strengths.extend(record.strengths)
                if record.improvements:
                    all_improvements.extend(record.improvements)
            
            # Get most common themes
            common_strengths = self._get_common_themes(all_strengths)
            common_improvements = self._get_common_themes(all_improvements)
            
            return {
                "total_analyses": total_analyses,
                "average_scores": {
                    "matter": round(avg_matter, 1),
                    "manner": round(avg_manner, 1),
                    "method": round(avg_method, 1),
                    "total": round(avg_total, 1)
                },
                "score_trends": score_trends,
                "common_strengths": common_strengths[:5],
                "common_improvements": common_improvements[:5],
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating analytics summary: {str(e)}")
            return {}
    
    def _calculate_score_trends(self, records: List[AnalysisRecord]) -> Dict[str, List[float]]:
        """Calculate score trends over time."""
        # Group by week and calculate averages
        weekly_scores = {}
        
        for record in records:
            week_key = record.timestamp.strftime("%Y-W%U")
            
            if week_key not in weekly_scores:
                weekly_scores[week_key] = {
                    "matter": [],
                    "manner": [],
                    "method": [],
                    "total": []
                }
            
            weekly_scores[week_key]["matter"].append(record.matter_score)
            weekly_scores[week_key]["manner"].append(record.manner_score)
            weekly_scores[week_key]["method"].append(record.method_score)
            weekly_scores[week_key]["total"].append(record.total_score)
        
        # Calculate averages
        trends = {
            "matter": [],
            "manner": [],
            "method": [],
            "total": []
        }
        
        for week_key in sorted(weekly_scores.keys()):
            week_data = weekly_scores[week_key]
            trends["matter"].append(round(sum(week_data["matter"]) / len(week_data["matter"]), 1))
            trends["manner"].append(round(sum(week_data["manner"]) / len(week_data["manner"]), 1))
            trends["method"].append(round(sum(week_data["method"]) / len(week_data["method"]), 1))
            trends["total"].append(round(sum(week_data["total"]) / len(week_data["total"]), 1))
        
        return trends
    
    def _get_common_themes(self, themes: List[str]) -> List[str]:
        """Get most common themes from feedback."""
        if not themes:
            return []
        
        # Simple frequency counting
        theme_counts = {}
        for theme in themes:
            # Extract key words from feedback
            words = theme.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    theme_counts[word] = theme_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [theme for theme, count in sorted_themes if count > 1]
