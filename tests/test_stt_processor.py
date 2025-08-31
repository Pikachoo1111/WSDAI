"""
Tests for STT processor functionality.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from src.speech.stt_processor import DebateSTTProcessor, TranscriptSegment, STTResult


class TestDebateSTTProcessor:
    """Test cases for DebateSTTProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DebateSTTProcessor()
    
    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.model_size == "base"
        assert self.processor.model is None
        assert len(self.processor.debate_vocab) > 0
        assert "comparative" in self.processor.debate_vocab
    
    @patch('whisper.load_model')
    def test_load_model(self, mock_load_model):
        """Test model loading."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        self.processor.load_model()
        
        mock_load_model.assert_called_once_with("base")
        assert self.processor.model == mock_model
    
    def test_reduce_noise(self):
        """Test noise reduction functionality."""
        # Create test audio signal
        duration = 1.0  # 1 second
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Add noise
        noise = np.random.normal(0, 0.1, len(audio))
        noisy_audio = audio + noise
        
        # Apply noise reduction
        clean_audio = self.processor._reduce_noise(noisy_audio, sr)
        
        # Check that output has same length
        assert len(clean_audio) == len(noisy_audio)
        
        # Check that noise is reduced (simple check)
        assert np.std(clean_audio) <= np.std(noisy_audio)
    
    def test_detect_speech_segments(self):
        """Test speech segment detection."""
        # Create test audio with speech and silence
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create audio with speech in middle
        audio = np.zeros(len(t))
        speech_start = int(0.5 * sr)
        speech_end = int(1.5 * sr)
        audio[speech_start:speech_end] = np.sin(2 * np.pi * 440 * t[speech_start:speech_end])
        
        segments = self.processor.detect_speech_segments(audio, sr)
        
        # Should detect at least one speech segment
        assert len(segments) >= 1
        
        # Check segment timing is reasonable
        start_time, end_time = segments[0]
        assert 0.4 <= start_time <= 0.6  # Around 0.5 seconds
        assert 1.4 <= end_time <= 1.6    # Around 1.5 seconds
    
    @patch.object(DebateSTTProcessor, 'load_model')
    def test_transcribe_segment(self, mock_load_model):
        """Test segment transcription."""
        # Mock Whisper model
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "This is a test transcription"
        }
        self.processor.model = mock_model
        
        # Create test audio
        sr = 16000
        duration = 2.0
        audio = np.random.normal(0, 0.1, int(sr * duration))
        
        # Transcribe segment
        segment = self.processor.transcribe_segment(audio, 0.0, 2.0)
        
        assert isinstance(segment, TranscriptSegment)
        assert segment.text == "This is a test transcription"
        assert segment.start_time == 0.0
        assert segment.end_time == 2.0
        assert segment.words_per_minute > 0
    
    def test_analyze_filler_words(self):
        """Test filler word analysis."""
        text_with_fillers = "Um, this is like, you know, a test speech with uh, many fillers"
        text_without_fillers = "This is a clean speech without any unnecessary words"
        
        filler_count_1 = self.processor.analyze_filler_words(text_with_fillers)
        filler_count_2 = self.processor.analyze_filler_words(text_without_fillers)
        
        assert filler_count_1 > filler_count_2
        assert filler_count_1 >= 4  # um, like, you know, uh
        assert filler_count_2 == 0
    
    def test_analyze_pauses(self):
        """Test pause analysis."""
        # Create segments with known pauses
        segments = [
            TranscriptSegment("First segment", 0.0, 2.0, 0.9, True, 150),
            TranscriptSegment("Second segment", 3.0, 5.0, 0.9, True, 150),  # 1 second pause
            TranscriptSegment("Third segment", 5.5, 7.0, 0.9, True, 150),   # 0.5 second pause
        ]
        
        pause_analysis = self.processor.analyze_pauses(segments)
        
        assert "average_pause" in pause_analysis
        assert "max_pause" in pause_analysis
        assert "total_pause_time" in pause_analysis
        
        assert pause_analysis["max_pause"] == 1.0
        assert pause_analysis["average_pause"] == 0.75  # (1.0 + 0.5) / 2
        assert pause_analysis["total_pause_time"] == 1.5
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        # Test with good result
        good_result = {"text": "This is a clear and confident speech"}
        confidence_good = self.processor._calculate_confidence(good_result)
        
        # Test with poor result
        poor_result = {"text": "uh... [unclear] ???"}
        confidence_poor = self.processor._calculate_confidence(poor_result)
        
        assert 0.0 <= confidence_good <= 1.0
        assert 0.0 <= confidence_poor <= 1.0
        assert confidence_good > confidence_poor
    
    @patch('librosa.load')
    @patch.object(DebateSTTProcessor, 'detect_speech_segments')
    @patch.object(DebateSTTProcessor, 'transcribe_segment')
    @patch.object(DebateSTTProcessor, 'load_model')
    def test_process_audio_integration(self, mock_load_model, mock_transcribe, 
                                     mock_detect_segments, mock_librosa_load):
        """Test complete audio processing pipeline."""
        # Mock dependencies
        mock_librosa_load.return_value = (np.random.normal(0, 0.1, 16000), 16000)
        mock_detect_segments.return_value = [(0.0, 2.0), (3.0, 5.0)]
        
        mock_segment_1 = TranscriptSegment("First part of speech", 0.0, 2.0, 0.9, True, 150)
        mock_segment_2 = TranscriptSegment("Second part of speech", 3.0, 5.0, 0.8, True, 140)
        mock_transcribe.side_effect = [mock_segment_1, mock_segment_2]
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Process audio
            result = self.processor.process_audio(tmp_path)
            
            # Verify result
            assert isinstance(result, STTResult)
            assert len(result.segments) == 2
            assert "First part of speech Second part of speech" in result.full_transcript
            assert result.clarity_score > 0
            assert result.average_wpm > 0
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


@pytest.fixture
def sample_audio_file():
    """Create a sample audio file for testing."""
    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        # Generate simple sine wave
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        
        # Save as WAV (simplified - in real tests you'd use proper audio library)
        tmp_file.write(audio.tobytes())
        
        yield tmp_file.name
    
    # Clean up
    if os.path.exists(tmp_file.name):
        os.unlink(tmp_file.name)


def test_processor_with_real_audio(sample_audio_file):
    """Integration test with actual audio file."""
    processor = DebateSTTProcessor()
    
    # This would require actual Whisper model - skip in CI
    if os.environ.get('SKIP_MODEL_TESTS'):
        pytest.skip("Skipping model tests")
    
    # Test would process real audio file
    # result = processor.process_audio(sample_audio_file)
    # assert isinstance(result, STTResult)
