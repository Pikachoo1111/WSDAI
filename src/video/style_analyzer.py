"""
Video style analysis for debate evaluation.
Analyzes eye contact, posture, gestures, and visual engagement.
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class GestureAnalysis:
    """Analysis of hand gestures and body language."""
    gesture_frequency: float
    gesture_variety: float
    appropriate_gestures: float
    distracting_movements: float


@dataclass
class PostureAnalysis:
    """Analysis of posture and body positioning."""
    posture_stability: float
    confidence_indicators: float
    engagement_level: float
    professional_appearance: float


@dataclass
class EyeContactAnalysis:
    """Analysis of eye contact patterns."""
    audience_focus_percentage: float
    note_reliance_percentage: float
    eye_contact_consistency: float
    engagement_score: float


@dataclass
class StyleAnalysisResult:
    """Complete visual style analysis result."""
    eye_contact: EyeContactAnalysis
    posture: PostureAnalysis
    gestures: GestureAnalysis
    overall_engagement: float
    visual_confidence: float
    professionalism_score: float


class VideoStyleAnalyzer:
    """
    Analyzes visual delivery aspects of debate speeches.
    """
    
    def __init__(self):
        """Initialize the video style analyzer."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe models
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=settings.FACE_DETECTION_CONFIDENCE
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
    
    def extract_eye_gaze_direction(self, face_landmarks, frame_shape: Tuple[int, int]) -> Tuple[float, float]:
        """
        Estimate gaze direction from facial landmarks.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_shape: (height, width) of frame
            
        Returns:
            Tuple of (horizontal_gaze, vertical_gaze) normalized to [-1, 1]
        """
        if not face_landmarks.landmark:
            return 0.0, 0.0
            
        h, w = frame_shape
        
        # Key eye landmarks
        left_eye_center = face_landmarks.landmark[159]  # Left eye center
        right_eye_center = face_landmarks.landmark[386]  # Right eye center
        nose_tip = face_landmarks.landmark[1]  # Nose tip
        
        # Convert to pixel coordinates
        left_eye = (left_eye_center.x * w, left_eye_center.y * h)
        right_eye = (right_eye_center.x * w, right_eye_center.y * h)
        nose = (nose_tip.x * w, nose_tip.y * h)
        
        # Calculate eye center
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        
        # Estimate gaze direction relative to nose
        horizontal_gaze = (eye_center[0] - nose[0]) / (w / 2)
        vertical_gaze = (eye_center[1] - nose[1]) / (h / 2)
        
        # Normalize to [-1, 1]
        horizontal_gaze = np.clip(horizontal_gaze, -1, 1)
        vertical_gaze = np.clip(vertical_gaze, -1, 1)
        
        return horizontal_gaze, vertical_gaze
    
    def analyze_eye_contact(self, video_path: str) -> EyeContactAnalysis:
        """
        Analyze eye contact patterns throughout the video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            EyeContactAnalysis object
        """
        cap = cv2.VideoCapture(video_path)
        
        gaze_data = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip frames for performance (analyze every 5th frame)
            if frame_count % 5 != 0:
                continue
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process face landmarks
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h_gaze, v_gaze = self.extract_eye_gaze_direction(
                        face_landmarks, 
                        (frame.shape[0], frame.shape[1])
                    )
                    gaze_data.append((h_gaze, v_gaze))
        
        cap.release()
        
        if not gaze_data:
            return EyeContactAnalysis(0.0, 1.0, 0.0, 0.0)
        
        # Analyze gaze patterns
        gaze_array = np.array(gaze_data)
        
        # Audience focus: gaze directed forward (small horizontal deviation)
        audience_frames = np.sum(np.abs(gaze_array[:, 0]) < settings.EYE_CONTACT_THRESHOLD)
        audience_percentage = audience_frames / len(gaze_data)
        
        # Note reliance: gaze directed downward
        note_frames = np.sum(gaze_array[:, 1] > 0.3)  # Looking down
        note_percentage = note_frames / len(gaze_data)
        
        # Consistency: low variance in gaze direction
        gaze_variance = np.var(gaze_array[:, 0])
        consistency = max(0, 1.0 - gaze_variance)
        
        # Overall engagement score
        engagement = audience_percentage * consistency
        
        return EyeContactAnalysis(
            audience_focus_percentage=audience_percentage,
            note_reliance_percentage=note_percentage,
            eye_contact_consistency=consistency,
            engagement_score=engagement
        )
    
    def analyze_posture(self, video_path: str) -> PostureAnalysis:
        """
        Analyze posture and body positioning.
        
        Args:
            video_path: Path to video file
            
        Returns:
            PostureAnalysis object
        """
        cap = cv2.VideoCapture(video_path)
        
        posture_data = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % 10 != 0:
                continue
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose landmarks
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Key posture points
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                
                # Calculate shoulder alignment
                shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
                
                # Calculate head position relative to shoulders
                shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
                head_alignment = abs(nose.x - shoulder_center_x)
                
                posture_data.append({
                    'shoulder_alignment': shoulder_diff,
                    'head_alignment': head_alignment,
                    'confidence_y': nose.y  # Higher y = lower confidence posture
                })
        
        cap.release()
        
        if not posture_data:
            return PostureAnalysis(0.5, 0.5, 0.5, 0.5)
        
        # Calculate posture metrics
        shoulder_alignments = [p['shoulder_alignment'] for p in posture_data]
        head_alignments = [p['head_alignment'] for p in posture_data]
        confidence_positions = [p['confidence_y'] for p in posture_data]
        
        # Stability: low variance in posture
        stability = max(0, 1.0 - np.var(shoulder_alignments) * 10)
        
        # Confidence: upright posture (low y values for head)
        confidence = max(0, 1.0 - np.mean(confidence_positions))
        
        # Engagement: consistent, aligned posture
        engagement = stability * (1.0 - np.mean(head_alignments))
        
        # Professionalism: overall posture quality
        professionalism = (stability + confidence + engagement) / 3
        
        return PostureAnalysis(
            posture_stability=stability,
            confidence_indicators=confidence,
            engagement_level=engagement,
            professional_appearance=professionalism
        )
    
    def analyze_gestures(self, video_path: str) -> GestureAnalysis:
        """
        Analyze hand gestures and movements.
        
        Args:
            video_path: Path to video file
            
        Returns:
            GestureAnalysis object
        """
        cap = cv2.VideoCapture(video_path)
        
        gesture_data = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % 5 != 0:
                continue
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand landmarks
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Calculate hand movement
                    landmarks = hand_landmarks.landmark
                    wrist = landmarks[0]
                    fingertips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
                    
                    # Calculate gesture activity
                    finger_spread = np.var([tip.x for tip in fingertips])
                    hand_height = wrist.y
                    
                    gesture_data.append({
                        'activity': finger_spread,
                        'height': hand_height,
                        'visible': True
                    })
            else:
                gesture_data.append({
                    'activity': 0,
                    'height': 1.0,
                    'visible': False
                })
        
        cap.release()
        
        if not gesture_data:
            return GestureAnalysis(0.0, 0.0, 0.5, 0.0)
        
        # Calculate gesture metrics
        visible_frames = sum(1 for g in gesture_data if g['visible'])
        total_frames = len(gesture_data)
        
        frequency = visible_frames / total_frames if total_frames > 0 else 0
        
        activities = [g['activity'] for g in gesture_data if g['visible']]
        variety = np.std(activities) if activities else 0
        
        # Appropriate gestures: moderate frequency and variety
        appropriate = min(frequency * 2, 1.0) * min(variety * 5, 1.0)
        
        # Distracting movements: excessive activity
        excessive_activity = sum(1 for a in activities if a > 0.1)
        distracting = excessive_activity / len(activities) if activities else 0
        
        return GestureAnalysis(
            gesture_frequency=frequency,
            gesture_variety=variety,
            appropriate_gestures=appropriate,
            distracting_movements=distracting
        )
    
    def analyze_video_style(self, video_path: str) -> StyleAnalysisResult:
        """
        Complete video style analysis.
        
        Args:
            video_path: Path to video file
            
        Returns:
            StyleAnalysisResult object
        """
        logger.info(f"Analyzing video style for: {video_path}")
        
        # Analyze each component
        eye_contact = self.analyze_eye_contact(video_path)
        posture = self.analyze_posture(video_path)
        gestures = self.analyze_gestures(video_path)
        
        # Calculate overall scores
        overall_engagement = (
            eye_contact.engagement_score * 0.4 +
            posture.engagement_level * 0.3 +
            gestures.appropriate_gestures * 0.3
        )
        
        visual_confidence = (
            eye_contact.audience_focus_percentage * 0.3 +
            posture.confidence_indicators * 0.4 +
            (1.0 - eye_contact.note_reliance_percentage) * 0.3
        )
        
        professionalism_score = (
            posture.professional_appearance * 0.4 +
            eye_contact.eye_contact_consistency * 0.3 +
            (1.0 - gestures.distracting_movements) * 0.3
        )
        
        return StyleAnalysisResult(
            eye_contact=eye_contact,
            posture=posture,
            gestures=gestures,
            overall_engagement=overall_engagement,
            visual_confidence=visual_confidence,
            professionalism_score=professionalism_score
        )
