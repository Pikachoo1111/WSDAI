"""
World Schools Debate rubric implementation and scoring system.
Evaluates speeches based on Matter, Manner, and Method criteria.
"""
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging
import json
import requests
import asyncio

from config.settings import WSD_RUBRIC, WSD_SPEAKER_POINT_RANGES, settings
from src.speech.stt_processor import STTResult
from src.speech.sentiment_analyzer import SentimentResult
from src.video.style_analyzer import StyleAnalysisResult

logger = logging.getLogger(__name__)


class LLMEvaluator:
    """
    LLM-based evaluation service for subjective debate assessment.
    Uses Hack Club AI API for intelligent scoring.
    """

    def __init__(self):
        self.api_url = "https://ai.hackclub.com/chat/completions"
        self.model = "qwen/qwen3-32b"  # Best model for reasoning tasks

    async def evaluate_component(self, component: str, transcript: str,
                                speaker_role: str, context: Dict) -> Dict:
        """
        Evaluate a specific WSD component using LLM reasoning.

        Args:
            component: "content", "style", or "strategy"
            transcript: Full speech transcript
            speaker_role: Speaker's role in debate
            context: Additional context (timing, visual analysis, etc.)

        Returns:
            Dict with score (0-1) and detailed feedback
        """
        prompt = self._build_evaluation_prompt(component, transcript, speaker_role, context)

        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,  # Lower temperature for consistent scoring
                    "max_tokens": 1000
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return self._parse_llm_response(content)
            else:
                logger.error(f"LLM API error: {response.status_code}")
                return self._fallback_evaluation(component)

        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return self._fallback_evaluation(component)

    def _build_evaluation_prompt(self, component: str, transcript: str,
                                speaker_role: str, context: Dict) -> str:
        """Build evaluation prompt for specific component."""

        base_prompt = f"""You are an expert World Schools Debate judge evaluating a {speaker_role} speech.

COMPONENT TO EVALUATE: {component.upper()}

OFFICIAL WSD CRITERIA:
"""

        if component == "content":
            criteria_prompt = """
CONTENT (40% of score): Argumentation quality divorced from delivery style
- Argument strength: Quality of individual arguments and reasoning
- Evidence quality: Supporting evidence, examples, and credibility
- Logical reasoning: Sound logical connections and flow
- Analysis depth: Depth of explanation and impact analysis
- Factual accuracy: Accuracy of claims and evidence presented

Weak arguments should be marked regardless of whether opponents respond to them.
"""
        elif component == "style":
            criteria_prompt = f"""
STYLE (40% of score): Communication clarity and delivery effectiveness
- Vocal delivery: Rate, pitch, tone (Context: {context.get('wpm', 'N/A')} WPM, confidence: {context.get('vocal_confidence', 'N/A')})
- Physical delivery: Gestures, expressions, posture (Context: gesture effectiveness: {context.get('gesture_score', 'N/A')})
- Eye contact: Audience engagement vs note reliance (Context: {context.get('audience_focus', 'N/A')}% audience focus)
- Clarity: Clear articulation and communication (Context: {context.get('filler_words', 'N/A')} filler words)
- Note usage: Reference only, not reading (Context: {context.get('note_reliance', 'N/A')}% note reliance)
"""
        else:  # strategy
            criteria_prompt = f"""
STRATEGY (20% of score): Understanding issue importance and strategic choices
- Issue prioritization: Identifying most substantive issues
- Time allocation: Allocating time based on importance (Context: {context.get('duration', 'N/A')} minutes)
- Structural choices: Speech organization appropriate for role
- POI handling: Strategic Points of Information responses
- Strategic focus: Overall approach to winning the debate

Role expectations for {speaker_role}:
{self._get_role_expectations(speaker_role)}
"""

        evaluation_prompt = f"""
SPEECH TRANSCRIPT:
{transcript}

EVALUATION TASK:
1. Analyze the speech against the {component} criteria above
2. Consider the speaker's role and strategic context
3. Provide a score from 0.0 to 1.0 (where 1.0 is exceptional)
4. Give specific feedback with examples from the speech

RESPONSE FORMAT (JSON):
{{
    "score": 0.75,
    "reasoning": "Detailed analysis of strengths and weaknesses...",
    "specific_feedback": "Actionable suggestions for improvement...",
    "examples": ["Quote from speech demonstrating strength/weakness"]
}}

Be objective, fair, and focus on the specific criteria for {component}.
"""

        return base_prompt + criteria_prompt + evaluation_prompt

    def _get_role_expectations(self, speaker_role: str) -> str:
        """Get strategic expectations for specific speaker role."""
        expectations = {
            "first_proposition": "Establish case framework, define terms, present core arguments",
            "first_opposition": "Accept/reject definitions, establish counter-framework, present opposition case",
            "second_proposition": "Respond to opposition attacks, rebuild case, extend arguments",
            "second_opposition": "Attack proposition case, extend opposition arguments, engage with clash",
            "third_proposition": "Summarize clash, weigh competing claims, conclude why proposition wins",
            "third_opposition": "Summarize clash, weigh competing claims, conclude why opposition wins"
        }
        return expectations.get(speaker_role, "Fulfill role-appropriate strategic objectives")

    def _parse_llm_response(self, content: str) -> Dict:
        """Parse LLM response into structured format."""
        try:
            # Try to extract JSON from response
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = content[start:end]
                result = json.loads(json_str)

                # Validate and normalize
                score = float(result.get('score', 0.5))
                score = max(0.0, min(1.0, score))  # Clamp to [0,1]

                return {
                    'score': score,
                    'reasoning': result.get('reasoning', 'No detailed reasoning provided'),
                    'feedback': result.get('specific_feedback', 'No specific feedback provided'),
                    'examples': result.get('examples', [])
                }
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")

        # Fallback parsing
        return {
            'score': 0.5,
            'reasoning': content[:200] + "..." if len(content) > 200 else content,
            'feedback': "Unable to parse detailed feedback",
            'examples': []
        }

    def _fallback_evaluation(self, component: str) -> Dict:
        """Fallback evaluation when LLM fails."""
        return {
            'score': 0.6,  # Neutral score
            'reasoning': f"LLM evaluation unavailable for {component}. Using fallback scoring.",
            'feedback': f"Unable to provide detailed {component} feedback due to technical issues.",
            'examples': []
        }


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
    Evaluates debate speeches according to WSD rubric standards using LLM-based assessment.
    """

    def __init__(self):
        """Initialize the rubric evaluator."""
        self.rubric = WSD_RUBRIC
        self.main_speech_min = settings.MAIN_SPEECH_MIN_SCORE
        self.main_speech_max = settings.MAIN_SPEECH_MAX_SCORE
        self.reply_speech_min = settings.REPLY_SPEECH_MIN_SCORE
        self.reply_speech_max = settings.REPLY_SPEECH_MAX_SCORE
        self.main_speech_range = self.main_speech_max - self.main_speech_min  # 20 points
        self.reply_speech_range = self.reply_speech_max - self.reply_speech_min  # 10 points
        self.speaker_point_ranges = WSD_SPEAKER_POINT_RANGES
        self.llm_evaluator = LLMEvaluator()
        
    async def evaluate_content(self, stt_result: STTResult, speaker_role: SpeakerRole) -> Tuple[float, List[RubricScore]]:
        """
        Evaluate Content (40% of total score) using LLM-based assessment.

        Official WSD Criteria: Focus on argumentation divorced from style. Weak arguments
        should be marked accordingly, even if the other team does not expose them.

        Args:
            stt_result: Speech-to-text analysis result
            speaker_role: Speaker's role in the debate

        Returns:
            Tuple of (content_score_0_to_1, detailed_scores)
        """
        transcript = stt_result.full_transcript

        # Prepare context for LLM evaluation
        context = {
            'duration': stt_result.total_duration / 60,
            'word_count': len(transcript.split()),
            'clarity_score': stt_result.clarity_score
        }

        # Get LLM evaluation for content
        try:
            llm_result = await self.llm_evaluator.evaluate_component(
                component="content",
                transcript=transcript,
                speaker_role=speaker_role.value,
                context=context
            )

            content_score = llm_result['score']

            # Create detailed rubric scores based on LLM analysis
            content_scores = [
                RubricScore(
                    category="content",
                    criterion="overall_content",
                    score=content_score,
                    max_score=1.0,
                    feedback=llm_result['feedback']
                )
            ]

            return content_score, content_scores

        except Exception as e:
            logger.error(f"LLM content evaluation failed: {e}")
            # Fallback to basic evaluation
            return self._fallback_content_evaluation(transcript, speaker_role)
    
    async def evaluate_style(self, sentiment_result: SentimentResult, style_result: StyleAnalysisResult,
                            stt_result: STTResult) -> Tuple[float, List[RubricScore]]:
        """
        Evaluate Style (40% of total score) using LLM-based assessment with multimodal context.

        Official WSD Criteria: Speakers should communicate clearly using effective rate, pitch,
        tone, hand gestures, facial expressions, etc. Notes for reference only, not reading.

        Args:
            sentiment_result: Speech sentiment analysis
            style_result: Video style analysis
            stt_result: Speech-to-text result

        Returns:
            Tuple of (style_score_0_to_1, detailed_scores)
        """
        transcript = stt_result.full_transcript

        # Prepare rich context for LLM evaluation
        context = {
            'wpm': stt_result.average_wpm,
            'vocal_confidence': sentiment_result.confidence_score,
            'gesture_score': style_result.gesture_analysis.effectiveness,
            'audience_focus': style_result.eye_contact.audience_focus_percentage * 100,
            'filler_words': stt_result.filler_word_count,
            'note_reliance': style_result.eye_contact.note_reliance_percentage * 100,
            'clarity_score': stt_result.clarity_score,
            'emotional_impact': sentiment_result.emotional_impact,
            'visual_confidence': style_result.visual_confidence
        }

        # Get LLM evaluation for style
        try:
            llm_result = await self.llm_evaluator.evaluate_component(
                component="style",
                transcript=transcript,
                speaker_role="general",  # Style is less role-dependent
                context=context
            )

            style_score = llm_result['score']

            # Create detailed rubric scores based on LLM analysis
            style_scores = [
                RubricScore(
                    category="style",
                    criterion="overall_style",
                    score=style_score,
                    max_score=1.0,
                    feedback=llm_result['feedback']
                )
            ]

            return style_score, style_scores

        except Exception as e:
            logger.error(f"LLM style evaluation failed: {e}")
            # Fallback to basic evaluation
            return self._fallback_style_evaluation(sentiment_result, style_result, stt_result)
    
    async def evaluate_strategy(self, stt_result: STTResult, speaker_role: SpeakerRole) -> Tuple[float, List[RubricScore]]:
        """
        Evaluate Strategy (20% of total score) using LLM-based assessment.

        Official WSD Criteria: Whether speaker understands importance of issues and allocates
        time based on relative importance. Includes POI handling and strategic focus.

        Args:
            stt_result: Speech-to-text analysis result
            speaker_role: Speaker's role in the debate

        Returns:
            Tuple of (strategy_score_0_to_1, detailed_scores)
        """
        transcript = stt_result.full_transcript

        # Prepare context for LLM evaluation
        context = {
            'duration': stt_result.total_duration / 60,
            'target_duration': 8.0,  # Standard WSD speech length
            'word_count': len(transcript.split()),
            'average_wpm': stt_result.average_wpm
        }

        # Get LLM evaluation for strategy
        try:
            llm_result = await self.llm_evaluator.evaluate_component(
                component="strategy",
                transcript=transcript,
                speaker_role=speaker_role.value,
                context=context
            )

            strategy_score = llm_result['score']

            # Create detailed rubric scores based on LLM analysis
            strategy_scores = [
                RubricScore(
                    category="strategy",
                    criterion="overall_strategy",
                    score=strategy_score,
                    max_score=1.0,
                    feedback=llm_result['feedback']
                )
            ]

            return strategy_score, strategy_scores

        except Exception as e:
            logger.error(f"LLM strategy evaluation failed: {e}")
            # Fallback to basic evaluation
            return self._fallback_strategy_evaluation(transcript, speaker_role, stt_result)

    # Fallback evaluation methods when LLM is unavailable
    def _fallback_content_evaluation(self, transcript: str, speaker_role: SpeakerRole) -> Tuple[float, List[RubricScore]]:
        """Fallback content evaluation using basic heuristics."""
        # Simple keyword-based scoring
        argument_words = ["because", "therefore", "evidence", "research", "study", "data"]
        argument_count = sum(1 for word in argument_words if word in transcript.lower())
        score = min(argument_count / 10.0, 1.0) * 0.6 + 0.3  # 0.3 to 0.9 range

        return score, [RubricScore(
            category="content",
            criterion="fallback_content",
            score=score,
            max_score=1.0,
            feedback="Basic content evaluation - LLM unavailable"
        )]

    def _fallback_style_evaluation(self, sentiment_result: SentimentResult,
                                  style_result: StyleAnalysisResult,
                                  stt_result: STTResult) -> Tuple[float, List[RubricScore]]:
        """Fallback style evaluation using multimodal data."""
        # Combine available metrics
        vocal_score = sentiment_result.confidence_score
        visual_score = style_result.overall_engagement
        clarity_score = stt_result.clarity_score

        style_score = (vocal_score + visual_score + clarity_score) / 3

        return style_score, [RubricScore(
            category="style",
            criterion="fallback_style",
            score=style_score,
            max_score=1.0,
            feedback="Basic style evaluation - LLM unavailable"
        )]

    def _fallback_strategy_evaluation(self, transcript: str, speaker_role: SpeakerRole,
                                     stt_result: STTResult) -> Tuple[float, List[RubricScore]]:
        """Fallback strategy evaluation using basic heuristics."""
        # Time management component
        duration_minutes = stt_result.total_duration / 60
        time_score = max(0, 1.0 - abs(duration_minutes - 8.0) / 8.0)

        # Structure component
        structure_words = ["first", "second", "third", "finally", "in conclusion"]
        structure_count = sum(1 for word in structure_words if word in transcript.lower())
        structure_score = min(structure_count / 3.0, 1.0)

        strategy_score = (time_score + structure_score) / 2

        return strategy_score, [RubricScore(
            category="strategy",
            criterion="fallback_strategy",
            score=strategy_score,
            max_score=1.0,
            feedback="Basic strategy evaluation - LLM unavailable"
        )]

    def _assess_argument_quality(self, transcript: str, speaker_role: SpeakerRole) -> float:
        """Assess the quality of arguments presented."""
        quality_indicators = []

        # Check for causal language
        causal_words = ["because", "therefore", "thus", "consequently", "as a result", "leads to"]
        causal_count = sum(1 for word in causal_words if word in transcript)
        quality_indicators.append(min(causal_count / 3.0, 1.0))

        # Check for comparative analysis
        comparative_words = ["compared to", "in contrast", "however", "whereas", "on the other hand"]
        comparative_count = sum(1 for phrase in comparative_words if phrase in transcript)
        quality_indicators.append(min(comparative_count / 2.0, 1.0))

        # Check for depth indicators
        depth_words = ["specifically", "particularly", "furthermore", "moreover", "additionally"]
        depth_count = sum(1 for word in depth_words if word in transcript)
        quality_indicators.append(min(depth_count / 3.0, 1.0))

        # Return score out of 2.0
        base_score = np.mean(quality_indicators) if quality_indicators else 0.5
        return base_score * 2.0
    
    def _assess_evidence_strength(self, transcript: str) -> float:
        """Assess the strength and use of evidence."""
        evidence_indicators = ["studies show", "research indicates", "statistics", "data",
                             "according to", "evidence suggests", "proven", "demonstrated"]

        evidence_count = sum(1 for indicator in evidence_indicators if indicator in transcript)
        return min(evidence_count / 2.0, 1.0) * 2.0
    
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

        return (logical_flow + consistency) / 2 * 2.0
    
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

        return min(clash_count / expected_clash, 1.0) * 1.0  # Out of 1.0

    def _assess_analysis_depth(self, transcript: str) -> float:
        """Assess depth of analysis."""
        depth_indicators = ["why", "how", "what if", "implications", "consequences",
                          "impact", "significance", "means that", "results in"]

        depth_count = sum(1 for indicator in depth_indicators if indicator in transcript)
        return min(depth_count / 5.0, 1.0) * 1.0  # Out of 1.0

    def _assess_clarity(self, stt_result: STTResult, style_result: StyleAnalysisResult) -> float:
        """Assess overall clarity of delivery."""
        # Combine STT clarity with visual clarity
        speech_clarity = stt_result.clarity_score
        visual_clarity = 1.0 - style_result.eye_contact.note_reliance_percentage

        return (speech_clarity + visual_clarity) / 2 * 1.5  # Out of 1.5
    
    def _assess_structure(self, transcript: str, segments) -> float:
        """Assess speech structure."""
        structure_words = ["introduction", "firstly", "secondly", "thirdly", "in conclusion",
                          "to summarize", "finally", "my first point", "my second point"]

        structure_count = sum(1 for word in structure_words if word in transcript)
        return min(structure_count / 3.0, 1.0) * 1.5  # Out of 1.5

    def _assess_signposting(self, transcript: str) -> float:
        """Assess use of signposting."""
        signpost_phrases = ["moving on to", "next", "now", "turning to", "my next point",
                           "in addition", "furthermore", "finally", "to conclude"]

        signpost_count = sum(1 for phrase in signpost_phrases if phrase in transcript)
        return min(signpost_count / 4.0, 1.0) * 1.0  # Out of 1.0

    def _assess_time_management(self, duration: float, wpm: float) -> float:
        """Assess time management (assuming 8-minute speeches)."""
        target_duration = 8 * 60  # 8 minutes in seconds
        duration_score = max(0, 1.0 - abs(duration - target_duration) / target_duration)

        # Optimal WPM is around 150-180
        optimal_wpm = 165
        wpm_score = max(0, 1.0 - abs(wpm - optimal_wpm) / optimal_wpm)

        return (duration_score + wpm_score) / 2 * 1.0  # Out of 1.0
    
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

        return min(keyword_count / len(expected_keywords), 1.0) * 1.5 if expected_keywords else 0.75  # Out of 1.5

    def _assess_strategic_approach(self, transcript: str, speaker_role: SpeakerRole) -> float:
        """Assess strategic thinking and approach."""
        strategy_words = ["strategy", "approach", "focus", "priority", "key", "crucial",
                         "most important", "central", "fundamental"]

        strategy_count = sum(1 for word in strategy_words if word in transcript)
        return min(strategy_count / 3.0, 1.0) * 1.0  # Out of 1.0

    # Official WSD Assessment Methods

    def _assess_argument_strength(self, transcript: str, speaker_role: SpeakerRole) -> float:
        """Assess quality of individual arguments (Content criterion)."""
        strength_indicators = []

        # Check for strong reasoning words
        reasoning_words = ["because", "therefore", "thus", "since", "given that", "due to"]
        reasoning_count = sum(1 for word in reasoning_words if word in transcript)
        strength_indicators.append(min(reasoning_count / 4.0, 1.0))

        # Check for substantive analysis
        analysis_words = ["impact", "consequence", "implication", "significance", "means that"]
        analysis_count = sum(1 for word in analysis_words if word in transcript)
        strength_indicators.append(min(analysis_count / 3.0, 1.0))

        # Check for comparative analysis
        comparative_words = ["compared to", "in contrast", "however", "whereas", "alternatively"]
        comparative_count = sum(1 for phrase in comparative_words if phrase in transcript)
        strength_indicators.append(min(comparative_count / 2.0, 1.0))

        return np.mean(strength_indicators) if strength_indicators else 0.5

    def _assess_evidence_quality(self, transcript: str) -> float:
        """Assess supporting evidence and examples (Content criterion)."""
        evidence_indicators = []

        # Check for credible sources
        source_words = ["according to", "research shows", "studies indicate", "data reveals",
                       "experts suggest", "statistics show", "report states"]
        source_count = sum(1 for phrase in source_words if phrase in transcript)
        evidence_indicators.append(min(source_count / 2.0, 1.0))

        # Check for specific examples
        example_words = ["for example", "for instance", "such as", "including", "specifically"]
        example_count = sum(1 for phrase in example_words if phrase in transcript)
        evidence_indicators.append(min(example_count / 3.0, 1.0))

        # Check for quantitative evidence
        quant_words = ["percent", "million", "billion", "increase", "decrease", "rate"]
        quant_count = sum(1 for word in quant_words if word in transcript)
        evidence_indicators.append(min(quant_count / 2.0, 1.0))

        return np.mean(evidence_indicators) if evidence_indicators else 0.4

    def _assess_logical_reasoning(self, transcript: str) -> float:
        """Assess sound logical connections (Content criterion)."""
        logic_indicators = []

        # Check for logical connectors
        connectors = ["therefore", "thus", "consequently", "as a result", "this leads to",
                     "it follows that", "this means", "hence"]
        connector_count = sum(1 for connector in connectors if connector in transcript)
        logic_indicators.append(min(connector_count / 3.0, 1.0))

        # Check for cause-effect relationships
        causal_words = ["causes", "results in", "leads to", "produces", "creates", "generates"]
        causal_count = sum(1 for word in causal_words if word in transcript)
        logic_indicators.append(min(causal_count / 2.0, 1.0))

        # Check for logical structure words
        structure_words = ["first", "second", "third", "finally", "in conclusion", "to summarize"]
        structure_count = sum(1 for word in structure_words if word in transcript)
        logic_indicators.append(min(structure_count / 4.0, 1.0))

        return np.mean(logic_indicators) if logic_indicators else 0.5

    def _assess_factual_accuracy(self, transcript: str) -> float:
        """Assess accuracy of claims and evidence (Content criterion)."""
        # This is a simplified assessment - in practice would need fact-checking
        accuracy_indicators = []

        # Check for hedging language (indicates uncertainty/accuracy awareness)
        hedge_words = ["likely", "probably", "suggests", "indicates", "appears", "seems"]
        hedge_count = sum(1 for word in hedge_words if word in transcript)
        accuracy_indicators.append(min(hedge_count / 2.0, 1.0))

        # Check for specific claims (more verifiable)
        specific_words = ["exactly", "precisely", "specifically", "in particular", "namely"]
        specific_count = sum(1 for word in specific_words if word in transcript)
        accuracy_indicators.append(min(specific_count / 2.0, 1.0))

        # Default to moderate accuracy without external fact-checking
        return max(0.6, np.mean(accuracy_indicators)) if accuracy_indicators else 0.6

    # Official WSD Style Assessment Methods

    def _assess_vocal_delivery(self, sentiment_result: SentimentResult, stt_result: STTResult) -> float:
        """Assess vocal delivery - rate, pitch, tone (Style criterion)."""
        delivery_indicators = []

        # Rate assessment (words per minute)
        wpm = stt_result.average_wpm
        optimal_wpm = 165  # Optimal speaking rate
        rate_score = max(0, 1.0 - abs(wpm - optimal_wpm) / optimal_wpm)
        delivery_indicators.append(rate_score)

        # Vocal variety from sentiment analysis
        vocal_variety = sentiment_result.emotional_impact
        delivery_indicators.append(vocal_variety)

        # Confidence in delivery
        confidence = sentiment_result.confidence_score
        delivery_indicators.append(confidence)

        return np.mean(delivery_indicators) if delivery_indicators else 0.5

    def _assess_physical_delivery(self, style_result: StyleAnalysisResult) -> float:
        """Assess physical delivery - gestures, expressions, posture (Style criterion)."""
        physical_indicators = []

        # Gesture effectiveness
        gesture_score = style_result.gesture_analysis.effectiveness
        physical_indicators.append(gesture_score)

        # Facial expressiveness
        expression_score = style_result.facial_analysis.expressiveness
        physical_indicators.append(expression_score)

        # Overall visual confidence
        confidence_score = style_result.visual_confidence
        physical_indicators.append(confidence_score)

        return np.mean(physical_indicators) if physical_indicators else 0.5

    def _assess_eye_contact(self, style_result: StyleAnalysisResult) -> float:
        """Assess eye contact and audience engagement (Style criterion)."""
        # Higher audience focus percentage = better score
        audience_focus = style_result.eye_contact.audience_focus_percentage
        return audience_focus

    def _assess_communication_clarity(self, stt_result: STTResult) -> float:
        """Assess communication clarity and articulation (Style criterion)."""
        clarity_indicators = []

        # STT clarity score
        stt_clarity = stt_result.clarity_score
        clarity_indicators.append(stt_clarity)

        # Filler word penalty
        filler_penalty = max(0, 1.0 - stt_result.filler_word_count / 20.0)
        clarity_indicators.append(filler_penalty)

        # Speaking pace consistency
        pace_consistency = 1.0 - (stt_result.pace_variation / 100.0) if hasattr(stt_result, 'pace_variation') else 0.8
        clarity_indicators.append(pace_consistency)

        return np.mean(clarity_indicators) if clarity_indicators else 0.7

    def _assess_note_usage(self, style_result: StyleAnalysisResult) -> float:
        """Assess appropriate use of notes for reference only (Style criterion)."""
        # Lower note reliance = better score (notes should be for reference only)
        note_reliance = style_result.eye_contact.note_reliance_percentage
        return max(0, 1.0 - note_reliance)

    # Official WSD Strategy Assessment Methods

    def _assess_issue_prioritization(self, transcript: str, speaker_role: SpeakerRole) -> float:
        """Assess identification of most substantive issues (Strategy criterion)."""
        priority_indicators = []

        # Check for prioritization language
        priority_words = ["most important", "key issue", "central", "crucial", "primary",
                         "main point", "fundamental", "critical"]
        priority_count = sum(1 for phrase in priority_words if phrase in transcript)
        priority_indicators.append(min(priority_count / 3.0, 1.0))

        # Check for comparative importance
        comparison_words = ["more important than", "less significant", "priority", "focus on"]
        comparison_count = sum(1 for phrase in comparison_words if phrase in transcript)
        priority_indicators.append(min(comparison_count / 2.0, 1.0))

        return np.mean(priority_indicators) if priority_indicators else 0.5

    def _assess_time_allocation(self, stt_result: STTResult, transcript: str) -> float:
        """Assess time allocation based on issue importance (Strategy criterion)."""
        allocation_indicators = []

        # Check if speech uses appropriate time (8 minutes for main speeches)
        duration_minutes = stt_result.total_duration / 60
        target_duration = 8.0
        time_usage = max(0, 1.0 - abs(duration_minutes - target_duration) / target_duration)
        allocation_indicators.append(time_usage)

        # Check for time awareness language
        time_words = ["time", "briefly", "in detail", "quickly", "focus", "spend time"]
        time_count = sum(1 for word in time_words if word in transcript)
        time_awareness = min(time_count / 3.0, 1.0)
        allocation_indicators.append(time_awareness)

        return np.mean(allocation_indicators) if allocation_indicators else 0.6

    def _assess_structural_choices(self, transcript: str, speaker_role: SpeakerRole) -> float:
        """Assess speech organization and timing choices (Strategy criterion)."""
        structure_indicators = []

        # Check for clear structure
        structure_words = ["first", "second", "third", "finally", "in conclusion",
                          "to begin", "next", "lastly"]
        structure_count = sum(1 for word in structure_words if word in transcript)
        structure_indicators.append(min(structure_count / 4.0, 1.0))

        # Check for role-appropriate structure
        role_structure_score = self._assess_role_appropriate_structure(transcript, speaker_role)
        structure_indicators.append(role_structure_score)

        return np.mean(structure_indicators) if structure_indicators else 0.5

    def _assess_poi_handling(self, transcript: str) -> float:
        """Assess Points of Information handling (Strategy criterion)."""
        poi_indicators = []

        # Check for POI-related language
        poi_words = ["point of information", "thank you", "no thank you", "yes", "question"]
        poi_count = sum(1 for phrase in poi_words if phrase in transcript)

        # If no POI language detected, assume moderate handling
        if poi_count == 0:
            return 0.7  # Neutral score when POIs not clearly present

        # If POI language present, assess quality
        poi_quality = min(poi_count / 2.0, 1.0)
        return poi_quality

    def _assess_strategic_focus(self, transcript: str, speaker_role: SpeakerRole) -> float:
        """Assess overall strategic approach to debate (Strategy criterion)."""
        focus_indicators = []

        # Check for strategic language
        strategy_words = ["strategy", "approach", "win", "defeat", "advantage", "weakness"]
        strategy_count = sum(1 for word in strategy_words if word in transcript)
        focus_indicators.append(min(strategy_count / 2.0, 1.0))

        # Check for role-specific strategic focus
        role_focus_score = self._assess_role_strategic_focus(transcript, speaker_role)
        focus_indicators.append(role_focus_score)

        return np.mean(focus_indicators) if focus_indicators else 0.5

    def _assess_role_appropriate_structure(self, transcript: str, speaker_role: SpeakerRole) -> float:
        """Assess if structure is appropriate for speaker role."""
        role_structure_words = {
            SpeakerRole.FIRST_PROP: ["define", "model", "case", "framework", "establish"],
            SpeakerRole.FIRST_OPP: ["accept", "reject", "definition", "counter", "oppose"],
            SpeakerRole.SECOND_PROP: ["rebuild", "extend", "rebuttal", "respond", "strengthen"],
            SpeakerRole.SECOND_OPP: ["attack", "refute", "counter", "challenge", "undermine"],
            SpeakerRole.THIRD_PROP: ["summarize", "weigh", "conclude", "why we win"],
            SpeakerRole.THIRD_OPP: ["summarize", "weigh", "conclude", "why they lose"]
        }

        expected_words = role_structure_words.get(speaker_role, [])
        if not expected_words:
            return 0.7

        word_count = sum(1 for word in expected_words if word in transcript)
        return min(word_count / len(expected_words), 1.0)

    def _assess_role_strategic_focus(self, transcript: str, speaker_role: SpeakerRole) -> float:
        """Assess strategic focus appropriate for speaker role."""
        # Different roles have different strategic priorities
        if speaker_role in [SpeakerRole.FIRST_PROP, SpeakerRole.FIRST_OPP]:
            # Opening speakers: case establishment
            focus_words = ["establish", "prove", "demonstrate", "show", "case"]
        elif speaker_role in [SpeakerRole.SECOND_PROP, SpeakerRole.SECOND_OPP]:
            # Middle speakers: clash and extension
            focus_words = ["respond", "counter", "extend", "build", "clash"]
        else:
            # Closing speakers: weighing and conclusion
            focus_words = ["weigh", "conclude", "overall", "balance", "win"]

        focus_count = sum(1 for word in focus_words if word in transcript)
        return min(focus_count / 3.0, 1.0)

    # Official WSD Feedback Generation Methods

    # Style Feedback Methods
    def _generate_vocal_delivery_feedback(self, score: float) -> str:
        """Generate feedback for vocal delivery (Style)."""
        if score >= 0.8:
            return "Excellent vocal delivery with effective rate, pitch, and tone"
        elif score >= 0.6:
            return "Good vocal delivery but could improve pace or vocal variety"
        else:
            return "Vocal delivery needs improvement - work on rate, pitch, and tone"

    def _generate_physical_delivery_feedback(self, score: float) -> str:
        """Generate feedback for physical delivery (Style)."""
        if score >= 0.8:
            return "Strong physical delivery with effective gestures and expressions"
        elif score >= 0.6:
            return "Good physical presence but could use more expressive gestures"
        else:
            return "Physical delivery needs improvement - work on gestures and posture"

    def _generate_eye_contact_feedback(self, score: float, style_result: StyleAnalysisResult) -> str:
        """Generate feedback for eye contact (Style)."""
        audience_focus = style_result.eye_contact.audience_focus_percentage
        if score >= 0.8:
            return f"Excellent audience engagement with {audience_focus:.1%} audience focus"
        elif score >= 0.6:
            return f"Good eye contact ({audience_focus:.1%}) but could engage audience more"
        else:
            return f"Poor eye contact ({audience_focus:.1%}) - look at audience more, not notes"

    def _generate_communication_clarity_feedback(self, score: float, stt_result: STTResult) -> str:
        """Generate feedback for communication clarity (Style)."""
        if score >= 0.8:
            return "Excellent clarity and articulation throughout speech"
        elif score >= 0.6:
            return f"Generally clear but {stt_result.filler_word_count} filler words noted"
        else:
            return "Clarity needs improvement - slow down and articulate more clearly"

    def _generate_note_usage_feedback(self, score: float, style_result: StyleAnalysisResult) -> str:
        """Generate feedback for note usage (Style)."""
        note_reliance = style_result.eye_contact.note_reliance_percentage
        if score >= 0.8:
            return f"Appropriate use of notes for reference ({note_reliance:.1%} reliance)"
        elif score >= 0.6:
            return f"Good note usage but reduce reliance ({note_reliance:.1%})"
        else:
            return f"Over-reliant on notes ({note_reliance:.1%}) - use for reference only"

    # Strategy Feedback Methods
    def _generate_issue_prioritization_feedback(self, score: float) -> str:
        """Generate feedback for issue prioritization (Strategy)."""
        if score >= 0.8:
            return "Excellent identification and prioritization of key issues"
        elif score >= 0.6:
            return "Good issue identification but could prioritize more clearly"
        else:
            return "Needs better identification of most important issues"

    def _generate_time_allocation_feedback(self, score: float, stt_result: STTResult) -> str:
        """Generate feedback for time allocation (Strategy)."""
        duration = stt_result.total_duration / 60
        if score >= 0.8:
            return f"Excellent time management ({duration:.1f} minutes)"
        elif score >= 0.6:
            return f"Good timing but could optimize allocation ({duration:.1f} minutes)"
        else:
            return f"Poor time allocation - {duration:.1f} minutes, adjust for importance"

    def _generate_structural_choices_feedback(self, score: float) -> str:
        """Generate feedback for structural choices (Strategy)."""
        if score >= 0.8:
            return "Excellent speech organization and structural choices"
        elif score >= 0.6:
            return "Good structure but could organize more strategically"
        else:
            return "Poor structural choices - organize based on strategic importance"

    def _generate_poi_handling_feedback(self, score: float) -> str:
        """Generate feedback for POI handling (Strategy)."""
        if score >= 0.8:
            return "Excellent handling of Points of Information"
        elif score >= 0.6:
            return "Good POI responses but could be more strategic"
        else:
            return "POI handling needs improvement - be more strategic in responses"

    def _generate_strategic_focus_feedback(self, score: float) -> str:
        """Generate feedback for strategic focus (Strategy)."""
        if score >= 0.8:
            return "Strong strategic approach with clear focus on winning"
        elif score >= 0.6:
            return "Good strategy but could be more focused on key advantages"
        else:
            return "Lacks strategic focus - identify key paths to victory"

    # Content Feedback Methods
    def _generate_argument_strength_feedback(self, score: float) -> str:
        """Generate feedback for argument strength (Content)."""
        if score >= 0.8:
            return "Strong, well-reasoned arguments with clear logical development"
        elif score >= 0.6:
            return "Good arguments but could benefit from stronger reasoning"
        else:
            return "Arguments need clearer logical development and stronger reasoning"

    def _generate_evidence_quality_feedback(self, score: float) -> str:
        """Generate feedback for evidence quality (Content)."""
        if score >= 0.8:
            return "Excellent use of credible evidence and specific examples"
        elif score >= 0.6:
            return "Good evidence but could be more specific and credible"
        else:
            return "Needs stronger, more credible evidence and examples"

    def _generate_logical_reasoning_feedback(self, score: float) -> str:
        """Generate feedback for logical reasoning (Content)."""
        if score >= 0.8:
            return "Clear logical connections and sound reasoning throughout"
        elif score >= 0.6:
            return "Generally logical but some connections could be clearer"
        else:
            return "Logical reasoning needs improvement - unclear connections"

    def _generate_analysis_depth_feedback(self, score: float) -> str:
        """Generate feedback for analysis depth (Content)."""
        if score >= 0.8:
            return "Deep, sophisticated analysis with clear implications"
        elif score >= 0.6:
            return "Good analysis but could explore implications further"
        else:
            return "Analysis needs more depth and exploration of implications"

    def _generate_factual_accuracy_feedback(self, score: float) -> str:
        """Generate feedback for factual accuracy (Content)."""
        if score >= 0.8:
            return "Claims appear accurate with appropriate hedging where needed"
        elif score >= 0.6:
            return "Generally accurate but some claims could be more precise"
        else:
            return "Accuracy of claims questionable - needs better verification"

    # Legacy feedback methods (to be updated)
    def _generate_argument_feedback(self, score: float, transcript: str) -> str:
        """Generate feedback for argument quality."""
        if score >= 1.6:
            return "Strong analytical arguments with clear reasoning"
        elif score >= 1.2:
            return "Good arguments but could benefit from deeper analysis"
        else:
            return "Arguments need stronger logical development and evidence"

    def _generate_evidence_feedback(self, score: float, transcript: str) -> str:
        """Generate feedback for evidence use."""
        if score >= 1.6:
            return "Excellent use of evidence and examples"
        elif score >= 1.2:
            return "Good evidence but could be more specific"
        else:
            return "Needs more concrete evidence and examples"

    def _generate_logic_feedback(self, score: float) -> str:
        """Generate feedback for logical consistency."""
        if score >= 1.6:
            return "Clear logical progression throughout speech"
        elif score >= 1.2:
            return "Generally logical but some unclear connections"
        else:
            return "Logical flow needs improvement"
    
    def _generate_clash_feedback(self, score: float, speaker_role: SpeakerRole) -> str:
        """Generate feedback for clash engagement."""
        if score >= 0.8:
            return "Excellent engagement with opposing arguments"
        elif score >= 0.6:
            return "Good clash but could address more opposition points"
        else:
            return "Needs more direct engagement with opposing case"

    def _generate_depth_feedback(self, score: float) -> str:
        """Generate feedback for analysis depth."""
        if score >= 0.8:
            return "Deep, sophisticated analysis of issues"
        elif score >= 0.6:
            return "Good analysis but could explore implications further"
        else:
            return "Analysis needs more depth and exploration"

    def _generate_clarity_feedback(self, score: float, stt_result: STTResult) -> str:
        """Generate feedback for clarity."""
        if score >= 1.2:
            return "Excellent clarity in delivery and articulation"
        elif score >= 0.9:
            return f"Generally clear but {stt_result.filler_word_count} filler words noted"
        else:
            return "Clarity needs improvement - slow down and articulate clearly"

    def _generate_persuasiveness_feedback(self, score: float) -> str:
        """Generate feedback for persuasiveness."""
        if score >= 1.2:
            return "Highly persuasive delivery with strong conviction"
        elif score >= 0.9:
            return "Persuasive but could show more confidence"
        else:
            return "Delivery needs more conviction and persuasive power"
    
    def _generate_engagement_feedback(self, score: float, style_result: StyleAnalysisResult) -> str:
        """Generate feedback for audience engagement."""
        eye_contact = style_result.eye_contact.audience_focus_percentage
        if score >= 1.2:
            return "Excellent audience connection and engagement"
        elif score >= 0.9:
            return f"Good engagement but increase eye contact ({eye_contact:.1%} audience focus)"
        else:
            return "Needs better audience connection - reduce note reliance"

    def _generate_confidence_feedback(self, score: float) -> str:
        """Generate feedback for confidence."""
        if score >= 0.8:
            return "Confident and assured delivery"
        elif score >= 0.6:
            return "Generally confident but some hesitation noted"
        else:
            return "Needs more confidence in delivery and posture"

    def _generate_vocal_feedback(self, score: float) -> str:
        """Generate feedback for vocal variety."""
        if score >= 0.4:
            return "Excellent vocal variety and emphasis"
        elif score >= 0.3:
            return "Good vocal variety but could vary pace more"
        else:
            return "Needs more vocal variety to maintain interest"

    def _generate_structure_feedback(self, score: float) -> str:
        """Generate feedback for structure."""
        if score >= 1.2:
            return "Well-structured speech with clear organization"
        elif score >= 0.9:
            return "Good structure but could be clearer"
        else:
            return "Needs clearer structure and organization"

    def _generate_signpost_feedback(self, score: float) -> str:
        """Generate feedback for signposting."""
        if score >= 0.8:
            return "Excellent signposting throughout speech"
        elif score >= 0.6:
            return "Good signposting but could be more consistent"
        else:
            return "Needs clearer signposting between points"

    def _generate_time_feedback(self, score: float, duration: float) -> str:
        """Generate feedback for time management."""
        minutes = duration / 60
        if score >= 0.8:
            return f"Excellent time management ({minutes:.1f} minutes)"
        elif score >= 0.6:
            return f"Good timing but could be optimized ({minutes:.1f} minutes)"
        else:
            return f"Time management needs work ({minutes:.1f} minutes)"
    
    def _generate_role_feedback(self, score: float, speaker_role: SpeakerRole) -> str:
        """Generate feedback for role fulfillment."""
        role_name = speaker_role.value.replace("_", " ").title()
        if score >= 1.2:
            return f"Excellent fulfillment of {role_name} role"
        elif score >= 0.9:
            return f"Good {role_name} speech but could fulfill role better"
        else:
            return f"Needs to better fulfill {role_name} expectations"

    def _generate_strategy_feedback(self, score: float) -> str:
        """Generate feedback for strategic approach."""
        if score >= 0.8:
            return "Strong strategic thinking and prioritization"
        elif score >= 0.6:
            return "Good strategy but could be more focused"
        else:
            return "Needs clearer strategic approach and focus"
    
    async def evaluate_speech(self, stt_result: STTResult, sentiment_result: SentimentResult,
                             style_result: StyleAnalysisResult, speaker_role: SpeakerRole,
                             is_reply_speech: bool = False) -> WSDScore:
        """
        Complete WSD evaluation of a speech using LLM-powered official World Schools Debate criteria.

        Args:
            stt_result: Speech-to-text analysis
            sentiment_result: Sentiment analysis
            style_result: Video style analysis
            speaker_role: Speaker's role in debate
            is_reply_speech: Whether this is a reply speech (30-40 points) or main speech (60-80 points)

        Returns:
            Complete WSDScore object
        """
        # Evaluate each component using LLM-powered official WSD criteria
        style_score, style_details = await self.evaluate_style(sentiment_result, style_result, stt_result)
        content_score, content_details = await self.evaluate_content(stt_result, speaker_role)
        strategy_score, strategy_details = await self.evaluate_strategy(stt_result, speaker_role)

        # Calculate total score using official WSD weightings
        # Style: 40%, Content: 40%, Strategy: 20%
        weighted_score = (
            style_score * self.rubric["style"]["weight"] +
            content_score * self.rubric["content"]["weight"] +
            strategy_score * self.rubric["strategy"]["weight"]
        )

        # Convert to appropriate point range
        if is_reply_speech:
            # Reply speeches: 30-40 points
            total_score = self.reply_speech_min + (weighted_score * self.reply_speech_range)
            score_type = "reply_speech"
        else:
            # Main speeches: 60-80 points
            total_score = self.main_speech_min + (weighted_score * self.main_speech_range)
            score_type = "main_speech"

        # Ensure score stays within bounds
        if is_reply_speech:
            total_score = max(self.reply_speech_min, min(self.reply_speech_max, total_score))
        else:
            total_score = max(self.main_speech_min, min(self.main_speech_max, total_score))

        # Combine all detailed scores
        all_scores = style_details + content_details + strategy_details

        # Generate overall feedback
        overall_feedback = self._generate_overall_feedback(style_score, content_score, strategy_score, score_type)

        # Generate strengths and improvements
        strengths = self._identify_strengths(all_scores)
        improvements = self._identify_improvements(all_scores)

        return WSDScore(
            matter_score=content_score,  # Map to legacy field names
            manner_score=style_score,
            method_score=strategy_score,
            total_score=total_score,
            rubric_scores=all_scores,
            overall_feedback=overall_feedback,
            strengths=strengths,
            improvements=improvements
        )
    
    def _generate_overall_feedback(self, style: float, content: float, strategy: float, score_type: str) -> str:
        """Generate overall performance feedback using official WSD criteria."""
        # Calculate weighted total (all scores are 0-1, so total is 0-1)
        weighted_total = (
            style * self.rubric["style"]["weight"] +
            content * self.rubric["content"]["weight"] +
            strategy * self.rubric["strategy"]["weight"]
        )

        # Convert to actual point range
        if score_type == "reply_speech":
            total_points = self.reply_speech_min + (weighted_total * self.reply_speech_range)
            if total_points >= 39:
                return "Outstanding reply speech with exceptional analysis and weighing."
            elif total_points >= 37:
                return "Strong reply speech with clear strategic focus."
            elif total_points >= 36:
                return "Solid reply speech meeting expectations."
            elif total_points >= 34:
                return "Adequate reply speech with room for improvement."
            elif total_points >= 32:
                return "Below average reply speech with significant issues."
            else:
                return "Weak reply speech requiring substantial development."
        else:
            total_points = self.main_speech_min + (weighted_total * self.main_speech_range)
            if total_points >= 78:
                return "Outstanding speech demonstrating exceptional skill in style, content, and strategy."
            elif total_points >= 75:
                return "Strong speech with clear competence across all WSD criteria."
            elif total_points >= 72:
                return "Solid speech meeting expectations with good technique."
            elif total_points >= 68:
                return "Adequate speech with room for improvement in key areas."
            elif total_points >= 64:
                return "Below average speech with significant areas needing development."
            else:
                return "Weak speech requiring substantial improvement in fundamental skills."
    
    def _identify_strengths(self, scores: List[RubricScore]) -> List[str]:
        """Identify top performing areas."""
        # Identify scores that are in the top 80% of their max score
        strong_scores = [score for score in scores if score.score >= score.max_score * 0.8]
        return [f"{score.criterion.replace('_', ' ').title()}: {score.feedback}"
                for score in strong_scores[:3]]

    def _identify_improvements(self, scores: List[RubricScore]) -> List[str]:
        """Identify areas needing improvement."""
        # Identify scores that are below 60% of their max score
        weak_scores = [score for score in scores if score.score < score.max_score * 0.6]
        return [f"{score.criterion.replace('_', ' ').title()}: {score.feedback}"
                for score in weak_scores[:3]]
