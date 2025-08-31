"""
Core debate analysis processor that orchestrates all analysis components.
"""
import os
import time
import asyncio
import logging
from typing import Callable, Optional
from datetime import datetime

import cv2
import librosa
from moviepy.editor import VideoFileClip

from src.speech.stt_processor import DebateSTTProcessor
from src.speech.sentiment_analyzer import SpeechSentimentAnalyzer
from src.video.style_analyzer import VideoStyleAnalyzer
from src.judge.wsd_rubric import WSDRubricEvaluator, SpeakerRole
from src.api.models import (
    AnalysisResultResponse, STTAnalysisResponse, SentimentAnalysisResponse,
    StyleAnalysisResponse, WSDScoreResponse, TranscriptSegmentResponse,
    RubricScoreResponse, EyeContactResponse, PostureResponse, GestureResponse,
    SpeakerRoleEnum
)

logger = logging.getLogger(__name__)


class DebateAnalysisProcessor:
    """
    Main processor that coordinates all analysis components.
    """
    
    def __init__(self):
        """Initialize the analysis processor."""
        self.stt_processor = DebateSTTProcessor()
        self.sentiment_analyzer = SpeechSentimentAnalyzer()
        self.style_analyzer = VideoStyleAnalyzer()
        self.rubric_evaluator = WSDRubricEvaluator()
        
    def extract_audio_from_video(self, video_path: str, audio_path: str) -> float:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            audio_path: Path to save extracted audio
            
        Returns:
            Duration of the video in seconds
        """
        try:
            # Load video
            video = VideoFileClip(video_path)
            duration = video.duration
            
            # Extract audio
            audio = video.audio
            audio.write_audiofile(audio_path, verbose=False, logger=None)
            
            # Clean up
            video.close()
            audio.close()
            
            return duration
            
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise
    
    def validate_video_file(self, video_path: str) -> dict:
        """
        Validate video file and extract metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video metadata dictionary
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError("Cannot open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Validate minimum requirements
            if duration < 30:  # Minimum 30 seconds
                raise ValueError("Video too short (minimum 30 seconds)")
            
            if duration > 600:  # Maximum 10 minutes
                raise ValueError("Video too long (maximum 10 minutes)")
            
            if width < 480 or height < 360:
                logger.warning("Low resolution video may affect analysis quality")
            
            return {
                "duration": duration,
                "fps": fps,
                "width": width,
                "height": height,
                "frame_count": frame_count
            }
            
        except Exception as e:
            logger.error(f"Error validating video: {str(e)}")
            raise
    
    def convert_speaker_role(self, role_enum: SpeakerRoleEnum) -> SpeakerRole:
        """Convert API enum to internal enum."""
        role_mapping = {
            SpeakerRoleEnum.FIRST_PROP: SpeakerRole.FIRST_PROP,
            SpeakerRoleEnum.FIRST_OPP: SpeakerRole.FIRST_OPP,
            SpeakerRoleEnum.SECOND_PROP: SpeakerRole.SECOND_PROP,
            SpeakerRoleEnum.SECOND_OPP: SpeakerRole.SECOND_OPP,
            SpeakerRoleEnum.THIRD_PROP: SpeakerRole.THIRD_PROP,
            SpeakerRoleEnum.THIRD_OPP: SpeakerRole.THIRD_OPP,
        }
        return role_mapping[role_enum]
    
    def convert_stt_result_to_response(self, stt_result) -> STTAnalysisResponse:
        """Convert STT result to API response model."""
        segments = [
            TranscriptSegmentResponse(
                text=seg.text,
                start_time=seg.start_time,
                end_time=seg.end_time,
                confidence=seg.confidence,
                speaker_id=seg.speaker_id,
                is_clear=seg.is_clear,
                words_per_minute=seg.words_per_minute
            )
            for seg in stt_result.segments
        ]
        
        return STTAnalysisResponse(
            full_transcript=stt_result.full_transcript,
            segments=segments,
            clarity_score=stt_result.clarity_score,
            average_wpm=stt_result.average_wpm,
            total_duration=stt_result.total_duration,
            filler_word_count=stt_result.filler_word_count,
            pause_analysis=stt_result.pause_analysis
        )
    
    def convert_sentiment_result_to_response(self, sentiment_result) -> SentimentAnalysisResponse:
        """Convert sentiment result to API response model."""
        return SentimentAnalysisResponse(
            confidence_score=sentiment_result.confidence_score,
            persuasiveness_score=sentiment_result.persuasiveness_score,
            emotional_impact=sentiment_result.emotional_impact,
            sentiment_label=sentiment_result.sentiment_label,
            sentiment_confidence=sentiment_result.sentiment_confidence
        )
    
    def convert_style_result_to_response(self, style_result) -> StyleAnalysisResponse:
        """Convert style result to API response model."""
        return StyleAnalysisResponse(
            eye_contact=EyeContactResponse(
                audience_focus_percentage=style_result.eye_contact.audience_focus_percentage,
                note_reliance_percentage=style_result.eye_contact.note_reliance_percentage,
                eye_contact_consistency=style_result.eye_contact.eye_contact_consistency,
                engagement_score=style_result.eye_contact.engagement_score
            ),
            posture=PostureResponse(
                posture_stability=style_result.posture.posture_stability,
                confidence_indicators=style_result.posture.confidence_indicators,
                engagement_level=style_result.posture.engagement_level,
                professional_appearance=style_result.posture.professional_appearance
            ),
            gestures=GestureResponse(
                gesture_frequency=style_result.gestures.gesture_frequency,
                gesture_variety=style_result.gestures.gesture_variety,
                appropriate_gestures=style_result.gestures.appropriate_gestures,
                distracting_movements=style_result.gestures.distracting_movements
            ),
            overall_engagement=style_result.overall_engagement,
            visual_confidence=style_result.visual_confidence,
            professionalism_score=style_result.professionalism_score
        )
    
    def convert_wsd_score_to_response(self, wsd_score) -> WSDScoreResponse:
        """Convert WSD score to API response model."""
        rubric_scores = [
            RubricScoreResponse(
                category=score.category,
                criterion=score.criterion,
                score=score.score,
                max_score=score.max_score,
                feedback=score.feedback
            )
            for score in wsd_score.rubric_scores
        ]
        
        return WSDScoreResponse(
            matter_score=wsd_score.matter_score,
            manner_score=wsd_score.manner_score,
            method_score=wsd_score.method_score,
            total_score=wsd_score.total_score,
            rubric_scores=rubric_scores,
            overall_feedback=wsd_score.overall_feedback,
            strengths=wsd_score.strengths,
            improvements=wsd_score.improvements
        )
    
    async def process_video(
        self,
        video_path: str,
        speaker_name: str,
        speaker_role: SpeakerRoleEnum,
        debate_topic: str,
        team_side: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> AnalysisResultResponse:
        """
        Process a debate video through the complete analysis pipeline.
        
        Args:
            video_path: Path to the video file
            speaker_name: Name of the speaker
            speaker_role: Speaker's role in the debate
            debate_topic: Topic of the debate
            team_side: Proposition or Opposition
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete analysis result
        """
        start_time = time.time()
        analysis_id = os.path.basename(video_path).split('.')[0]
        
        try:
            # Update progress
            if progress_callback:
                progress_callback(0.05, "Validating video file...")
            
            # Validate video
            video_metadata = self.validate_video_file(video_path)
            logger.info(f"Video validated: {video_metadata}")
            
            # Extract audio
            if progress_callback:
                progress_callback(0.15, "Extracting audio from video...")
            
            audio_path = video_path.replace('.mp4', '.wav').replace('.avi', '.wav').replace('.mov', '.wav')
            video_duration = self.extract_audio_from_video(video_path, audio_path)
            
            # Speech-to-Text Analysis
            if progress_callback:
                progress_callback(0.30, "Transcribing speech...")
            
            stt_result = self.stt_processor.process_audio(audio_path)
            logger.info(f"STT completed: {len(stt_result.full_transcript)} characters transcribed")
            
            # Sentiment Analysis
            if progress_callback:
                progress_callback(0.50, "Analyzing speech sentiment and intonation...")
            
            # Load audio for sentiment analysis
            audio_data, sr = librosa.load(audio_path)
            sentiment_result = self.sentiment_analyzer.analyze_speech_sentiment(
                audio_data, sr, stt_result.full_transcript
            )
            logger.info(f"Sentiment analysis completed: {sentiment_result.sentiment_label}")
            
            # Video Style Analysis
            if progress_callback:
                progress_callback(0.70, "Analyzing video style and delivery...")
            
            style_result = self.style_analyzer.analyze_video_style(video_path)
            logger.info(f"Style analysis completed: engagement={style_result.overall_engagement:.2f}")
            
            # WSD Rubric Evaluation
            if progress_callback:
                progress_callback(0.85, "Evaluating against WSD rubric...")
            
            speaker_role_internal = self.convert_speaker_role(speaker_role)
            wsd_score = self.rubric_evaluator.evaluate_speech(
                stt_result, sentiment_result, style_result, speaker_role_internal
            )
            logger.info(f"WSD evaluation completed: total={wsd_score.total_score:.1f}")
            
            # Clean up temporary audio file
            try:
                os.remove(audio_path)
            except:
                pass
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update final progress
            if progress_callback:
                progress_callback(1.0, "Analysis completed successfully!")
            
            # Create response
            result = AnalysisResultResponse(
                analysis_id=analysis_id,
                speaker_name=speaker_name,
                speaker_role=speaker_role,
                debate_topic=debate_topic,
                team_side=team_side,
                timestamp=datetime.now(),
                stt_analysis=self.convert_stt_result_to_response(stt_result),
                sentiment_analysis=self.convert_sentiment_result_to_response(sentiment_result),
                style_analysis=self.convert_style_result_to_response(style_result),
                wsd_score=self.convert_wsd_score_to_response(wsd_score),
                processing_time=processing_time,
                video_duration=video_duration,
                file_size=os.path.getsize(video_path)
            )
            
            logger.info(f"Analysis completed successfully in {processing_time:.1f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            if progress_callback:
                progress_callback(0.0, f"Analysis failed: {str(e)}")
            raise
