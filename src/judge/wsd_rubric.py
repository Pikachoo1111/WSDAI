"""
World Schools Debate rubric implementation and scoring system.
Evaluates speeches based on Matter, Manner, and Method criteria.
"""
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging

from config.settings import WSD_RUBRIC, settings
from src.speech.stt_processor import STTResult
from src.speech.sentiment_analyzer import SentimentResult
from src.video.style_analyzer import StyleAnalysisResult

logger = logging.getLogger(__name__)


class SpeakerRole(Enum):
    """Speaker roles in World Schools Debate."""
    FIRST_PROP = "first_proposition"
    FIRST_OPP = "first_opposition"
    SECOND_PROP = "second_proposition"
    SECOND_OPP = "second_opposition"
    THIRD_PROP = "third_proposition"
    THIRD_OPP = "third_opposition"


@dataclass
class RubricScore:
    """Individual rubric component score."""
    category: str
    criterion: str
    score: float
    max_score: float
    feedback: str


@dataclass
class WSDScore:
    """Complete WSD evaluation score."""
    matter_score: float
    manner_score: float
    method_score: float
    total_score: float
    rubric_scores: List[RubricScore]
    overall_feedback: str
    strengths: List[str]
    improvements: List[str]


class WSDRubricEvaluator:
    """
    Evaluates debate speeches according to WSD rubric standards.
    """
    
    def __init__(self):
        """Initialize the rubric evaluator."""
        self.rubric = WSD_RUBRIC
        self.max_score = settings.MAX_SCORE
        
    def evaluate_matter(self, stt_result: STTResult, speaker_role: SpeakerRole) -> Tuple[float, List[RubricScore]]:
        """
        Evaluate Matter (content quality) component.
        
        Args:
            stt_result: Speech-to-text analysis result
            speaker_role: Speaker's role in the debate
            
        Returns:
            Tuple of (matter_score, detailed_scores)
        """
        matter_scores = []
        transcript = stt_result.full_transcript.lower()
        
        # Argument Quality
        argument_score = self._assess_argument_quality(transcript, speaker_role)
        matter_scores.append(RubricScore(
            category="matter",
            criterion="argument_quality",
            score=argument_score,
            max_score=20.0,
            feedback=self._generate_argument_feedback(argument_score, transcript)
        ))
        
        # Evidence Strength
        evidence_score = self._assess_evidence_strength(transcript)
        matter_scores.append(RubricScore(
            category="matter",
            criterion="evidence_strength",
            score=evidence_score,
            max_score=20.0,
            feedback=self._generate_evidence_feedback(evidence_score, transcript)
        ))
        
        # Logical Consistency
        logic_score = self._assess_logical_consistency(transcript, stt_result.segments)
        matter_scores.append(RubricScore(
            category="matter",
            criterion="logical_consistency",
            score=logic_score,
            max_score=20.0,
            feedback=self._generate_logic_feedback(logic_score)
        ))
        
        # Clash Engagement
        clash_score = self._assess_clash_engagement(transcript, speaker_role)
        matter_scores.append(RubricScore(
            category="matter",
            criterion="clash_engagement",
            score=clash_score,
            max_score=20.0,
            feedback=self._generate_clash_feedback(clash_score, speaker_role)
        ))
        
        # Analysis Depth
        depth_score = self._assess_analysis_depth(transcript)
        matter_scores.append(RubricScore(
            category="matter",
            criterion="analysis_depth",
            score=depth_score,
            max_score=20.0,
            feedback=self._generate_depth_feedback(depth_score)
        ))
        
        # Calculate weighted matter score
        total_matter = sum(score.score for score in matter_scores)
        matter_percentage = (total_matter / 100.0) * self.max_score * self.rubric["matter"]["weight"]
        
        return matter_percentage, matter_scores
    
    def evaluate_manner(self, sentiment_result: SentimentResult, style_result: StyleAnalysisResult, 
                       stt_result: STTResult) -> Tuple[float, List[RubricScore]]:
        """
        Evaluate Manner (delivery style) component.
        
        Args:
            sentiment_result: Speech sentiment analysis
            style_result: Video style analysis
            stt_result: Speech-to-text result
            
        Returns:
            Tuple of (manner_score, detailed_scores)
        """
        manner_scores = []
        
        # Clarity
        clarity_score = self._assess_clarity(stt_result, style_result)
        manner_scores.append(RubricScore(
            category="manner",
            criterion="clarity",
            score=clarity_score,
            max_score=20.0,
            feedback=self._generate_clarity_feedback(clarity_score, stt_result)
        ))
        
        # Persuasiveness
        persuasiveness_score = sentiment_result.persuasiveness_score * 20
        manner_scores.append(RubricScore(
            category="manner",
            criterion="persuasiveness",
            score=persuasiveness_score,
            max_score=20.0,
            feedback=self._generate_persuasiveness_feedback(persuasiveness_score)
        ))
        
        # Audience Engagement
        engagement_score = style_result.overall_engagement * 20
        manner_scores.append(RubricScore(
            category="manner",
            criterion="audience_engagement",
            score=engagement_score,
            max_score=20.0,
            feedback=self._generate_engagement_feedback(engagement_score, style_result)
        ))
        
        # Confidence
        confidence_score = (sentiment_result.confidence_score + style_result.visual_confidence) / 2 * 20
        manner_scores.append(RubricScore(
            category="manner",
            criterion="confidence",
            score=confidence_score,
            max_score=20.0,
            feedback=self._generate_confidence_feedback(confidence_score)
        ))
        
        # Vocal Variety
        vocal_score = sentiment_result.emotional_impact * 20
        manner_scores.append(RubricScore(
            category="manner",
            criterion="vocal_variety",
            score=vocal_score,
            max_score=20.0,
            feedback=self._generate_vocal_feedback(vocal_score)
        ))
        
        # Calculate weighted manner score
        total_manner = sum(score.score for score in manner_scores)
        manner_percentage = (total_manner / 100.0) * self.max_score * self.rubric["manner"]["weight"]
        
        return manner_percentage, manner_scores
    
    def evaluate_method(self, stt_result: STTResult, speaker_role: SpeakerRole) -> Tuple[float, List[RubricScore]]:
        """
        Evaluate Method (structure and strategy) component.
        
        Args:
            stt_result: Speech-to-text analysis result
            speaker_role: Speaker's role in the debate
            
        Returns:
            Tuple of (method_score, detailed_scores)
        """
        method_scores = []
        transcript = stt_result.full_transcript.lower()
        
        # Structure
        structure_score = self._assess_structure(transcript, stt_result.segments)
        method_scores.append(RubricScore(
            category="method",
            criterion="structure",
            score=structure_score,
            max_score=20.0,
            feedback=self._generate_structure_feedback(structure_score)
        ))
        
        # Signposting
        signpost_score = self._assess_signposting(transcript)
        method_scores.append(RubricScore(
            category="method",
            criterion="signposting",
            score=signpost_score,
            max_score=20.0,
            feedback=self._generate_signpost_feedback(signpost_score)
        ))
        
        # Time Management
        time_score = self._assess_time_management(stt_result.total_duration, stt_result.average_wpm)
        method_scores.append(RubricScore(
            category="method",
            criterion="time_management",
            score=time_score,
            max_score=20.0,
            feedback=self._generate_time_feedback(time_score, stt_result.total_duration)
        ))
        
        # Role Fulfillment
        role_score = self._assess_role_fulfillment(transcript, speaker_role)
        method_scores.append(RubricScore(
            category="method",
            criterion="role_fulfillment",
            score=role_score,
            max_score=20.0,
            feedback=self._generate_role_feedback(role_score, speaker_role)
        ))
        
        # Strategic Approach
        strategy_score = self._assess_strategic_approach(transcript, speaker_role)
        method_scores.append(RubricScore(
            category="method",
            criterion="strategic_approach",
            score=strategy_score,
            max_score=20.0,
            feedback=self._generate_strategy_feedback(strategy_score)
        ))
        
        # Calculate weighted method score
        total_method = sum(score.score for score in method_scores)
        method_percentage = (total_method / 100.0) * self.max_score * self.rubric["method"]["weight"]
        
        return method_percentage, method_scores
    
    def _assess_argument_quality(self, transcript: str, speaker_role: SpeakerRole) -> float:
        """Assess the quality of arguments presented."""
        quality_indicators = []
        
        # Check for causal language
        causal_words = ["because", "therefore", "thus", "consequently", "as a result", "leads to"]
        causal_count = sum(1 for word in causal_words if word in transcript)
        quality_indicators.append(min(causal_count / 3.0, 1.0) * 20)
        
        # Check for comparative analysis
        comparative_words = ["compared to", "in contrast", "however", "whereas", "on the other hand"]
        comparative_count = sum(1 for phrase in comparative_words if phrase in transcript)
        quality_indicators.append(min(comparative_count / 2.0, 1.0) * 20)
        
        # Check for depth indicators
        depth_words = ["specifically", "particularly", "furthermore", "moreover", "additionally"]
        depth_count = sum(1 for word in depth_words if word in transcript)
        quality_indicators.append(min(depth_count / 3.0, 1.0) * 20)
        
        return np.mean(quality_indicators) if quality_indicators else 10.0
    
    def _assess_evidence_strength(self, transcript: str) -> float:
        """Assess the strength and use of evidence."""
        evidence_indicators = ["studies show", "research indicates", "statistics", "data", 
                             "according to", "evidence suggests", "proven", "demonstrated"]
        
        evidence_count = sum(1 for indicator in evidence_indicators if indicator in transcript)
        return min(evidence_count / 2.0, 1.0) * 20
    
    def _assess_logical_consistency(self, transcript: str, segments) -> float:
        """Assess logical flow and consistency."""
        # Check for logical connectors
        connectors = ["firstly", "secondly", "thirdly", "finally", "in conclusion", 
                     "next", "then", "furthermore", "moreover"]
        
        connector_count = sum(1 for connector in connectors if connector in transcript)
        logical_flow = min(connector_count / 4.0, 1.0)
        
        # Check for contradictions (simple heuristic)
        contradiction_pairs = [("always", "never"), ("all", "none"), ("increase", "decrease")]
        contradictions = 0
        for pos, neg in contradiction_pairs:
            if pos in transcript and neg in transcript:
                contradictions += 1
        
        consistency = max(0, 1.0 - contradictions / 3.0)
        
        return (logical_flow + consistency) / 2 * 20
    
    def _assess_clash_engagement(self, transcript: str, speaker_role: SpeakerRole) -> float:
        """Assess engagement with opposing arguments."""
        clash_words = ["however", "but", "opponents argue", "they claim", "in response", 
                      "rebuttal", "counter", "disagree", "refute"]
        
        clash_count = sum(1 for word in clash_words if word in transcript)
        
        # Different expectations based on speaker role
        if speaker_role in [SpeakerRole.FIRST_PROP, SpeakerRole.FIRST_OPP]:
            expected_clash = 1  # Less clash expected in opening speeches
        else:
            expected_clash = 3  # More clash expected in later speeches
            
        return min(clash_count / expected_clash, 1.0) * 20
    
    def _assess_analysis_depth(self, transcript: str) -> float:
        """Assess depth of analysis."""
        depth_indicators = ["why", "how", "what if", "implications", "consequences", 
                          "impact", "significance", "means that", "results in"]
        
        depth_count = sum(1 for indicator in depth_indicators if indicator in transcript)
        return min(depth_count / 5.0, 1.0) * 20
    
    def _assess_clarity(self, stt_result: STTResult, style_result: StyleAnalysisResult) -> float:
        """Assess overall clarity of delivery."""
        # Combine STT clarity with visual clarity
        speech_clarity = stt_result.clarity_score
        visual_clarity = 1.0 - style_result.eye_contact.note_reliance_percentage
        
        return (speech_clarity + visual_clarity) / 2 * 20
    
    def _assess_structure(self, transcript: str, segments) -> float:
        """Assess speech structure."""
        structure_words = ["introduction", "firstly", "secondly", "thirdly", "in conclusion", 
                          "to summarize", "finally", "my first point", "my second point"]
        
        structure_count = sum(1 for word in structure_words if word in transcript)
        return min(structure_count / 3.0, 1.0) * 20
    
    def _assess_signposting(self, transcript: str) -> float:
        """Assess use of signposting."""
        signpost_phrases = ["moving on to", "next", "now", "turning to", "my next point", 
                           "in addition", "furthermore", "finally", "to conclude"]
        
        signpost_count = sum(1 for phrase in signpost_phrases if phrase in transcript)
        return min(signpost_count / 4.0, 1.0) * 20
    
    def _assess_time_management(self, duration: float, wpm: float) -> float:
        """Assess time management (assuming 8-minute speeches)."""
        target_duration = 8 * 60  # 8 minutes in seconds
        duration_score = max(0, 1.0 - abs(duration - target_duration) / target_duration)
        
        # Optimal WPM is around 150-180
        optimal_wpm = 165
        wpm_score = max(0, 1.0 - abs(wpm - optimal_wpm) / optimal_wpm)
        
        return (duration_score + wpm_score) / 2 * 20
    
    def _assess_role_fulfillment(self, transcript: str, speaker_role: SpeakerRole) -> float:
        """Assess fulfillment of specific speaker role."""
        role_keywords = {
            SpeakerRole.FIRST_PROP: ["define", "model", "case", "framework"],
            SpeakerRole.FIRST_OPP: ["accept", "reject", "definition", "counter-model"],
            SpeakerRole.SECOND_PROP: ["rebuild", "extend", "rebuttal"],
            SpeakerRole.SECOND_OPP: ["attack", "rebuttal", "counter"],
            SpeakerRole.THIRD_PROP: ["summarize", "weigh", "why we win"],
            SpeakerRole.THIRD_OPP: ["summarize", "weigh", "why they lose"]
        }
        
        expected_keywords = role_keywords.get(speaker_role, [])
        keyword_count = sum(1 for keyword in expected_keywords if keyword in transcript)
        
        return min(keyword_count / len(expected_keywords), 1.0) * 20 if expected_keywords else 15.0
    
    def _assess_strategic_approach(self, transcript: str, speaker_role: SpeakerRole) -> float:
        """Assess strategic thinking and approach."""
        strategy_words = ["strategy", "approach", "focus", "priority", "key", "crucial", 
                         "most important", "central", "fundamental"]
        
        strategy_count = sum(1 for word in strategy_words if word in transcript)
        return min(strategy_count / 3.0, 1.0) * 20
    
    # Feedback generation methods
    def _generate_argument_feedback(self, score: float, transcript: str) -> str:
        """Generate feedback for argument quality."""
        if score >= 16:
            return "Strong analytical arguments with clear reasoning"
        elif score >= 12:
            return "Good arguments but could benefit from deeper analysis"
        else:
            return "Arguments need stronger logical development and evidence"
    
    def _generate_evidence_feedback(self, score: float, transcript: str) -> str:
        """Generate feedback for evidence use."""
        if score >= 16:
            return "Excellent use of evidence and examples"
        elif score >= 12:
            return "Good evidence but could be more specific"
        else:
            return "Needs more concrete evidence and examples"
    
    def _generate_logic_feedback(self, score: float) -> str:
        """Generate feedback for logical consistency."""
        if score >= 16:
            return "Clear logical progression throughout speech"
        elif score >= 12:
            return "Generally logical but some unclear connections"
        else:
            return "Logical flow needs improvement"
    
    def _generate_clash_feedback(self, score: float, speaker_role: SpeakerRole) -> str:
        """Generate feedback for clash engagement."""
        if score >= 16:
            return "Excellent engagement with opposing arguments"
        elif score >= 12:
            return "Good clash but could address more opposition points"
        else:
            return "Needs more direct engagement with opposing case"
    
    def _generate_depth_feedback(self, score: float) -> str:
        """Generate feedback for analysis depth."""
        if score >= 16:
            return "Deep, sophisticated analysis of issues"
        elif score >= 12:
            return "Good analysis but could explore implications further"
        else:
            return "Analysis needs more depth and exploration"
    
    def _generate_clarity_feedback(self, score: float, stt_result: STTResult) -> str:
        """Generate feedback for clarity."""
        if score >= 16:
            return "Excellent clarity in delivery and articulation"
        elif score >= 12:
            return f"Generally clear but {stt_result.filler_word_count} filler words noted"
        else:
            return "Clarity needs improvement - slow down and articulate clearly"
    
    def _generate_persuasiveness_feedback(self, score: float) -> str:
        """Generate feedback for persuasiveness."""
        if score >= 16:
            return "Highly persuasive delivery with strong conviction"
        elif score >= 12:
            return "Persuasive but could show more confidence"
        else:
            return "Delivery needs more conviction and persuasive power"
    
    def _generate_engagement_feedback(self, score: float, style_result: StyleAnalysisResult) -> str:
        """Generate feedback for audience engagement."""
        eye_contact = style_result.eye_contact.audience_focus_percentage
        if score >= 16:
            return "Excellent audience connection and engagement"
        elif score >= 12:
            return f"Good engagement but increase eye contact ({eye_contact:.1%} audience focus)"
        else:
            return "Needs better audience connection - reduce note reliance"
    
    def _generate_confidence_feedback(self, score: float) -> str:
        """Generate feedback for confidence."""
        if score >= 16:
            return "Confident and assured delivery"
        elif score >= 12:
            return "Generally confident but some hesitation noted"
        else:
            return "Needs more confidence in delivery and posture"
    
    def _generate_vocal_feedback(self, score: float) -> str:
        """Generate feedback for vocal variety."""
        if score >= 16:
            return "Excellent vocal variety and emphasis"
        elif score >= 12:
            return "Good vocal variety but could vary pace more"
        else:
            return "Needs more vocal variety to maintain interest"
    
    def _generate_structure_feedback(self, score: float) -> str:
        """Generate feedback for structure."""
        if score >= 16:
            return "Well-structured speech with clear organization"
        elif score >= 12:
            return "Good structure but could be clearer"
        else:
            return "Needs clearer structure and organization"
    
    def _generate_signpost_feedback(self, score: float) -> str:
        """Generate feedback for signposting."""
        if score >= 16:
            return "Excellent signposting throughout speech"
        elif score >= 12:
            return "Good signposting but could be more consistent"
        else:
            return "Needs clearer signposting between points"
    
    def _generate_time_feedback(self, score: float, duration: float) -> str:
        """Generate feedback for time management."""
        minutes = duration / 60
        if score >= 16:
            return f"Excellent time management ({minutes:.1f} minutes)"
        elif score >= 12:
            return f"Good timing but could be optimized ({minutes:.1f} minutes)"
        else:
            return f"Time management needs work ({minutes:.1f} minutes)"
    
    def _generate_role_feedback(self, score: float, speaker_role: SpeakerRole) -> str:
        """Generate feedback for role fulfillment."""
        role_name = speaker_role.value.replace("_", " ").title()
        if score >= 16:
            return f"Excellent fulfillment of {role_name} role"
        elif score >= 12:
            return f"Good {role_name} speech but could fulfill role better"
        else:
            return f"Needs to better fulfill {role_name} expectations"
    
    def _generate_strategy_feedback(self, score: float) -> str:
        """Generate feedback for strategic approach."""
        if score >= 16:
            return "Strong strategic thinking and prioritization"
        elif score >= 12:
            return "Good strategy but could be more focused"
        else:
            return "Needs clearer strategic approach and focus"
    
    def evaluate_speech(self, stt_result: STTResult, sentiment_result: SentimentResult, 
                       style_result: StyleAnalysisResult, speaker_role: SpeakerRole) -> WSDScore:
        """
        Complete WSD evaluation of a speech.
        
        Args:
            stt_result: Speech-to-text analysis
            sentiment_result: Sentiment analysis
            style_result: Video style analysis
            speaker_role: Speaker's role in debate
            
        Returns:
            Complete WSDScore object
        """
        # Evaluate each component
        matter_score, matter_details = self.evaluate_matter(stt_result, speaker_role)
        manner_score, manner_details = self.evaluate_manner(sentiment_result, style_result, stt_result)
        method_score, method_details = self.evaluate_method(stt_result, speaker_role)
        
        # Calculate total score
        total_score = matter_score + manner_score + method_score
        
        # Combine all detailed scores
        all_scores = matter_details + manner_details + method_details
        
        # Generate overall feedback
        overall_feedback = self._generate_overall_feedback(matter_score, manner_score, method_score)
        
        # Generate strengths and improvements
        strengths = self._identify_strengths(all_scores)
        improvements = self._identify_improvements(all_scores)
        
        return WSDScore(
            matter_score=matter_score,
            manner_score=manner_score,
            method_score=method_score,
            total_score=total_score,
            rubric_scores=all_scores,
            overall_feedback=overall_feedback,
            strengths=strengths,
            improvements=improvements
        )
    
    def _generate_overall_feedback(self, matter: float, manner: float, method: float) -> str:
        """Generate overall performance feedback."""
        total = matter + manner + method
        
        if total >= 80:
            return "Excellent speech demonstrating strong debating skills across all areas."
        elif total >= 70:
            return "Strong speech with good command of debate fundamentals."
        elif total >= 60:
            return "Solid speech with clear areas for improvement identified."
        else:
            return "Developing speech - focus on fundamental debate skills."
    
    def _identify_strengths(self, scores: List[RubricScore]) -> List[str]:
        """Identify top performing areas."""
        strong_scores = [score for score in scores if score.score >= 16]
        return [f"{score.criterion.replace('_', ' ').title()}: {score.feedback}" 
                for score in strong_scores[:3]]
    
    def _identify_improvements(self, scores: List[RubricScore]) -> List[str]:
        """Identify areas needing improvement."""
        weak_scores = [score for score in scores if score.score < 12]
        return [f"{score.criterion.replace('_', ' ').title()}: {score.feedback}" 
                for score in weak_scores[:3]]
