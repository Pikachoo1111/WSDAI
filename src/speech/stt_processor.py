"""
Robust Speech-to-Text processor optimized for debate speech.
Handles fast-paced speech, overlapping dialogue, and debate-specific vocabulary.
"""
import whisper
import torch
import librosa
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import webrtcvad
from pydub import AudioSegment
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a segment of transcribed speech."""
    text: str
    start_time: float
    end_time: float
    confidence: float
    speaker_id: Optional[str] = None
    is_clear: bool = True
    words_per_minute: float = 0.0


@dataclass
class STTResult:
    """Complete STT processing result."""
    full_transcript: str
    segments: List[TranscriptSegment]
    clarity_score: float
    average_wpm: float
    total_duration: float
    filler_word_count: int
    pause_analysis: Dict[str, float]


class DebateSTTProcessor:
    """
    Advanced Speech-to-Text processor designed for debate analysis.
    """
    
    def __init__(self, model_size: str = None):
        """Initialize the STT processor."""
        self.model_size = model_size or settings.WHISPER_MODEL
        self.model = None
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        self.debate_vocab = set(settings.DEBATE_VOCABULARY)
        self.filler_words = {
            "um", "uh", "er", "ah", "like", "you know", "so", "well",
            "actually", "basically", "literally", "obviously"
        }
        
    def load_model(self):
        """Load the Whisper model."""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            
    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Preprocess audio for optimal STT performance.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=settings.SAMPLE_RATE)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        # Apply noise reduction
        audio = self._reduce_noise(audio, sr)
        
        return audio, sr
    
    def _reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply basic noise reduction."""
        # Simple spectral subtraction for noise reduction
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from first 0.5 seconds
        noise_frame_count = int(0.5 * sr / 512)
        noise_spectrum = np.mean(magnitude[:, :noise_frame_count], axis=1, keepdims=True)
        
        # Subtract noise
        clean_magnitude = magnitude - 0.5 * noise_spectrum
        clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
        
        # Reconstruct audio
        clean_stft = clean_magnitude * np.exp(1j * phase)
        clean_audio = librosa.istft(clean_stft)
        
        return clean_audio
    
    def detect_speech_segments(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """
        Detect speech segments using Voice Activity Detection.
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            List of (start_time, end_time) tuples for speech segments
        """
        # Convert to 16kHz for VAD
        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio
            
        # Convert to bytes
        audio_bytes = (audio_16k * 32767).astype(np.int16).tobytes()
        
        # Process in 30ms frames
        frame_duration = 30  # ms
        frame_size = int(16000 * frame_duration / 1000)
        
        speech_segments = []
        current_segment_start = None
        
        for i in range(0, len(audio_bytes) - frame_size, frame_size):
            frame = audio_bytes[i:i + frame_size]
            is_speech = self.vad.is_speech(frame, 16000)
            
            time_offset = i / (16000 * 2)  # 2 bytes per sample
            
            if is_speech and current_segment_start is None:
                current_segment_start = time_offset
            elif not is_speech and current_segment_start is not None:
                speech_segments.append((current_segment_start, time_offset))
                current_segment_start = None
                
        # Handle case where speech continues to end
        if current_segment_start is not None:
            speech_segments.append((current_segment_start, len(audio_bytes) / (16000 * 2)))
            
        return speech_segments
    
    def transcribe_segment(self, audio: np.ndarray, start_time: float, end_time: float) -> TranscriptSegment:
        """
        Transcribe a specific audio segment.
        
        Args:
            audio: Full audio array
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            
        Returns:
            TranscriptSegment object
        """
        # Extract segment
        start_sample = int(start_time * settings.SAMPLE_RATE)
        end_sample = int(end_time * settings.SAMPLE_RATE)
        segment_audio = audio[start_sample:end_sample]
        
        # Transcribe with Whisper
        result = self.model.transcribe(segment_audio)
        
        # Calculate metrics
        text = result["text"].strip()
        duration = end_time - start_time
        word_count = len(text.split())
        wpm = (word_count / duration) * 60 if duration > 0 else 0
        
        # Assess clarity
        confidence = self._calculate_confidence(result)
        is_clear = confidence > 0.7 and wpm < 300  # Not too fast
        
        return TranscriptSegment(
            text=text,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            is_clear=is_clear,
            words_per_minute=wpm
        )
    
    def _calculate_confidence(self, whisper_result: Dict) -> float:
        """Calculate confidence score from Whisper result."""
        # Whisper doesn't provide direct confidence, so we estimate
        # based on text characteristics and model behavior
        text = whisper_result["text"]
        
        # Factors that indicate lower confidence
        confidence = 1.0
        
        # Penalize very short or very long segments
        word_count = len(text.split())
        if word_count < 3:
            confidence *= 0.8
        elif word_count > 50:
            confidence *= 0.9
            
        # Penalize excessive punctuation or unclear markers
        unclear_markers = ["[", "]", "(", ")", "...", "???"]
        for marker in unclear_markers:
            if marker in text:
                confidence *= 0.9
                
        return max(0.1, confidence)
    
    def analyze_filler_words(self, text: str) -> int:
        """Count filler words in transcript."""
        words = text.lower().split()
        return sum(1 for word in words if word in self.filler_words)
    
    def analyze_pauses(self, segments: List[TranscriptSegment]) -> Dict[str, float]:
        """Analyze pause patterns between segments."""
        if len(segments) < 2:
            return {"average_pause": 0.0, "max_pause": 0.0, "total_pause_time": 0.0}
            
        pauses = []
        for i in range(1, len(segments)):
            pause_duration = segments[i].start_time - segments[i-1].end_time
            if pause_duration > 0:
                pauses.append(pause_duration)
                
        if not pauses:
            return {"average_pause": 0.0, "max_pause": 0.0, "total_pause_time": 0.0}
            
        return {
            "average_pause": np.mean(pauses),
            "max_pause": np.max(pauses),
            "total_pause_time": np.sum(pauses)
        }
    
    def process_audio(self, audio_path: str) -> STTResult:
        """
        Complete STT processing pipeline.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            STTResult with complete analysis
        """
        self.load_model()
        
        # Preprocess audio
        audio, sr = self.preprocess_audio(audio_path)
        total_duration = len(audio) / sr
        
        # Detect speech segments
        speech_segments = self.detect_speech_segments(audio, sr)
        
        # Transcribe each segment
        transcript_segments = []
        for start_time, end_time in speech_segments:
            if end_time - start_time >= settings.MIN_SPEECH_DURATION:
                segment = self.transcribe_segment(audio, start_time, end_time)
                transcript_segments.append(segment)
        
        # Combine full transcript
        full_transcript = " ".join(segment.text for segment in transcript_segments)
        
        # Calculate overall metrics
        clarity_scores = [seg.confidence for seg in transcript_segments if seg.is_clear]
        clarity_score = np.mean(clarity_scores) if clarity_scores else 0.0
        
        wpm_values = [seg.words_per_minute for seg in transcript_segments]
        average_wpm = np.mean(wpm_values) if wpm_values else 0.0
        
        filler_count = self.analyze_filler_words(full_transcript)
        pause_analysis = self.analyze_pauses(transcript_segments)
        
        return STTResult(
            full_transcript=full_transcript,
            segments=transcript_segments,
            clarity_score=clarity_score,
            average_wpm=average_wpm,
            total_duration=total_duration,
            filler_word_count=filler_count,
            pause_analysis=pause_analysis
        )
