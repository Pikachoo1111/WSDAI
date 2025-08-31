"""
Speech sentiment and intonation analysis for debate evaluation.
Analyzes emotional impact, confidence, and persuasiveness of delivery.
"""
import librosa
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Sentiment analysis result for a speech segment."""
    confidence_score: float
    persuasiveness_score: float
    emotional_impact: float
    sentiment_label: str
    sentiment_confidence: float


@dataclass
class IntonationFeatures:
    """Intonation and prosodic features."""
    pitch_mean: float
    pitch_std: float
    pitch_range: float
    energy_mean: float
    energy_std: float
    speaking_rate: float
    emphasis_points: List[float]


class SpeechSentimentAnalyzer:
    """
    Analyzes speech sentiment and intonation patterns for debate evaluation.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_models(self):
        """Load sentiment analysis models."""
        if self.model is None:
            logger.info(f"Loading sentiment model: {settings.SENTIMENT_MODEL}")
            self.tokenizer = AutoTokenizer.from_pretrained(settings.SENTIMENT_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(settings.SENTIMENT_MODEL)
            self.model.to(self.device)
            self.model.eval()
    
    def extract_prosodic_features(self, audio: np.ndarray, sr: int) -> IntonationFeatures:
        """
        Extract prosodic features from audio signal.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            IntonationFeatures object
        """
        # Extract fundamental frequency (pitch)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Remove unvoiced frames
        f0_voiced = f0[voiced_flag]
        
        # Pitch statistics
        pitch_mean = np.nanmean(f0_voiced) if len(f0_voiced) > 0 else 0
        pitch_std = np.nanstd(f0_voiced) if len(f0_voiced) > 0 else 0
        pitch_range = np.nanmax(f0_voiced) - np.nanmin(f0_voiced) if len(f0_voiced) > 0 else 0
        
        # Energy features
        rms_energy = librosa.feature.rms(y=audio)[0]
        energy_mean = np.mean(rms_energy)
        energy_std = np.std(rms_energy)
        
        # Speaking rate (approximate)
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        speaking_rate = len(onset_frames) / (len(audio) / sr) if len(audio) > 0 else 0
        
        # Emphasis detection (high energy + pitch peaks)
        emphasis_points = self._detect_emphasis_points(f0, rms_energy, sr)
        
        return IntonationFeatures(
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            pitch_range=pitch_range,
            energy_mean=energy_mean,
            energy_std=energy_std,
            speaking_rate=speaking_rate,
            emphasis_points=emphasis_points
        )
    
    def _detect_emphasis_points(self, f0: np.ndarray, energy: np.ndarray, sr: int) -> List[float]:
        """Detect points of emphasis in speech."""
        # Normalize features
        f0_norm = (f0 - np.nanmean(f0)) / np.nanstd(f0) if np.nanstd(f0) > 0 else f0
        energy_norm = (energy - np.mean(energy)) / np.std(energy) if np.std(energy) > 0 else energy
        
        # Find peaks in combined signal
        combined_signal = np.nan_to_num(f0_norm) + energy_norm
        
        # Simple peak detection
        emphasis_threshold = np.mean(combined_signal) + 1.5 * np.std(combined_signal)
        emphasis_frames = np.where(combined_signal > emphasis_threshold)[0]
        
        # Convert to time points
        hop_length = 512  # Default librosa hop length
        emphasis_times = librosa.frames_to_time(emphasis_frames, sr=sr, hop_length=hop_length)
        
        return emphasis_times.tolist()
    
    def analyze_text_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment of transcript text.
        
        Args:
            text: Transcript text
            
        Returns:
            Tuple of (sentiment_label, confidence)
        """
        self.load_models()
        
        # Tokenize text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get top prediction
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = torch.max(predictions).item()
        
        # Map to sentiment labels (assuming RoBERTa sentiment model)
        label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment_label = label_mapping.get(predicted_class, "neutral")
        
        return sentiment_label, confidence
    
    def calculate_confidence_score(self, intonation: IntonationFeatures, text: str) -> float:
        """
        Calculate confidence score based on prosodic features and text.
        
        Args:
            intonation: Prosodic features
            text: Transcript text
            
        Returns:
            Confidence score (0-1)
        """
        confidence_indicators = []
        
        # Pitch variation indicates confidence
        if intonation.pitch_std > 0:
            pitch_variation = min(intonation.pitch_std / 50.0, 1.0)  # Normalize
            confidence_indicators.append(pitch_variation)
        
        # Steady energy indicates confidence
        if intonation.energy_std > 0:
            energy_stability = 1.0 - min(intonation.energy_std / intonation.energy_mean, 1.0)
            confidence_indicators.append(energy_stability)
        
        # Moderate speaking rate indicates confidence
        optimal_rate = 4.0  # syllables per second
        rate_score = 1.0 - abs(intonation.speaking_rate - optimal_rate) / optimal_rate
        confidence_indicators.append(max(0, rate_score))
        
        # Text-based confidence indicators
        uncertainty_words = ["maybe", "perhaps", "i think", "possibly", "might", "could be"]
        uncertainty_count = sum(1 for word in uncertainty_words if word in text.lower())
        text_confidence = max(0, 1.0 - uncertainty_count / 10.0)
        confidence_indicators.append(text_confidence)
        
        return np.mean(confidence_indicators) if confidence_indicators else 0.5
    
    def calculate_persuasiveness_score(self, intonation: IntonationFeatures, sentiment: str, text: str) -> float:
        """
        Calculate persuasiveness score based on delivery and content.
        
        Args:
            intonation: Prosodic features
            sentiment: Sentiment label
            text: Transcript text
            
        Returns:
            Persuasiveness score (0-1)
        """
        persuasiveness_factors = []
        
        # Emphasis usage
        emphasis_density = len(intonation.emphasis_points) / max(1, len(text.split()) / 10)
        emphasis_score = min(emphasis_density / 2.0, 1.0)  # Optimal around 2 emphasis per 10 words
        persuasiveness_factors.append(emphasis_score)
        
        # Pitch range (varied pitch is more engaging)
        if intonation.pitch_range > 0:
            pitch_variety = min(intonation.pitch_range / 100.0, 1.0)
            persuasiveness_factors.append(pitch_variety)
        
        # Sentiment appropriateness (positive/confident sentiment is persuasive)
        sentiment_score = 0.8 if sentiment == "positive" else 0.5 if sentiment == "neutral" else 0.3
        persuasiveness_factors.append(sentiment_score)
        
        # Strong language indicators
        strong_words = ["clearly", "obviously", "definitely", "absolutely", "certainly", "undoubtedly"]
        strong_word_count = sum(1 for word in strong_words if word in text.lower())
        strong_language_score = min(strong_word_count / 5.0, 1.0)
        persuasiveness_factors.append(strong_language_score)
        
        return np.mean(persuasiveness_factors) if persuasiveness_factors else 0.5
    
    def calculate_emotional_impact(self, intonation: IntonationFeatures, sentiment_confidence: float) -> float:
        """
        Calculate emotional impact score.
        
        Args:
            intonation: Prosodic features
            sentiment_confidence: Confidence in sentiment prediction
            
        Returns:
            Emotional impact score (0-1)
        """
        impact_factors = []
        
        # Energy variation creates emotional impact
        if intonation.energy_mean > 0:
            energy_variation = min(intonation.energy_std / intonation.energy_mean, 1.0)
            impact_factors.append(energy_variation)
        
        # Pitch variation creates emotional engagement
        if intonation.pitch_std > 0:
            pitch_impact = min(intonation.pitch_std / 30.0, 1.0)
            impact_factors.append(pitch_impact)
        
        # Strong sentiment confidence indicates emotional clarity
        impact_factors.append(sentiment_confidence)
        
        # Emphasis points create emotional peaks
        emphasis_impact = min(len(intonation.emphasis_points) / 10.0, 1.0)
        impact_factors.append(emphasis_impact)
        
        return np.mean(impact_factors) if impact_factors else 0.5
    
    def analyze_speech_sentiment(self, audio: np.ndarray, sr: int, transcript: str) -> SentimentResult:
        """
        Complete sentiment analysis of speech.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            transcript: Speech transcript
            
        Returns:
            SentimentResult object
        """
        # Extract prosodic features
        intonation = self.extract_prosodic_features(audio, sr)
        
        # Analyze text sentiment
        sentiment_label, sentiment_confidence = self.analyze_text_sentiment(transcript)
        
        # Calculate derived scores
        confidence_score = self.calculate_confidence_score(intonation, transcript)
        persuasiveness_score = self.calculate_persuasiveness_score(intonation, sentiment_label, transcript)
        emotional_impact = self.calculate_emotional_impact(intonation, sentiment_confidence)
        
        return SentimentResult(
            confidence_score=confidence_score,
            persuasiveness_score=persuasiveness_score,
            emotional_impact=emotional_impact,
            sentiment_label=sentiment_label,
            sentiment_confidence=sentiment_confidence
        )
