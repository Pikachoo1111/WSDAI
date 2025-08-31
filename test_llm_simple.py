#!/usr/bin/env python3
"""
Simple test for LLM-based World Schools Debate evaluation.
Tests just the LLM evaluator without full system dependencies.
"""

import asyncio
import json
import requests
from typing import Dict

class SimpleLLMEvaluator:
    """Simplified LLM evaluator for testing."""
    
    def __init__(self):
        self.api_url = "https://ai.hackclub.com/chat/completions"
        self.model = "qwen/qwen3-32b"
        
    async def evaluate_component(self, component: str, transcript: str, 
                                speaker_role: str, context: Dict) -> Dict:
        """Evaluate a specific WSD component using LLM reasoning."""
        prompt = self._build_evaluation_prompt(component, transcript, speaker_role, context)
        
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return self._parse_llm_response(content)
            else:
                print(f"❌ LLM API error: {response.status_code}")
                return self._fallback_evaluation(component)
                
        except Exception as e:
            print(f"❌ LLM evaluation failed: {e}")
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
- Vocal delivery: Rate, pitch, tone (Context: {context.get('wpm', 'N/A')} WPM)
- Physical delivery: Gestures, expressions, posture
- Eye contact: Audience engagement vs note reliance
- Clarity: Clear articulation and communication (Context: {context.get('filler_words', 'N/A')} filler words)
- Note usage: Reference only, not reading
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
            print(f"⚠️  Failed to parse LLM response: {e}")
            
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
            'score': 0.6,
            'reasoning': f"LLM evaluation unavailable for {component}. Using fallback scoring.",
            'feedback': f"Unable to provide detailed {component} feedback due to technical issues.",
            'examples': []
        }

async def test_llm_evaluation():
    """Test the LLM-based evaluation system."""
    
    print("🎯 Testing AI-Powered World Schools Debate Judge")
    print("=" * 60)
    print("🔗 Using Hack Club AI API (qwen/qwen3-32b)")
    print()
    
    evaluator = SimpleLLMEvaluator()
    
    # Test case: Strong content example
    test_transcript = """
    Honorable judges, the motion before us today asks whether artificial intelligence 
    will fundamentally improve education. I stand firmly with the proposition.
    
    Let me establish our framework clearly. When we speak of AI in education, we mean 
    adaptive learning systems that personalize instruction based on individual student 
    needs and learning patterns.
    
    My first argument centers on personalized learning at scale. Research from Stanford 
    University shows that students using AI-powered tutoring systems improved their 
    math scores by 34% compared to traditional methods. This is because AI can identify 
    exactly where each student struggles and provide targeted support.
    
    Second, AI democratizes access to quality education. In rural Kenya, AI-powered 
    tablets have brought world-class instruction to students who previously had no 
    access to qualified teachers. Literacy rates in these communities have doubled 
    in just two years.
    
    Third, AI frees teachers to focus on what they do best - inspiring and mentoring 
    students. When AI handles routine tasks like grading, teachers can spend more time 
    on creative lesson planning and one-on-one support.
    
    The opposition will argue about job displacement and privacy concerns. However, 
    evidence shows AI augments rather than replaces teachers, and proper governance 
    frameworks can address privacy issues.
    
    In conclusion, AI represents the most significant opportunity to improve educational 
    outcomes in our lifetime. We must embrace this technology to ensure every child 
    receives the personalized, high-quality education they deserve.
    """
    
    context = {
        'duration': 7.8,
        'word_count': len(test_transcript.split()),
        'wpm': 165,
        'filler_words': 2
    }
    
    components = ["content", "style", "strategy"]
    results = {}
    
    print("📊 Evaluating Sample First Proposition Speech...")
    print("-" * 50)
    
    for component in components:
        print(f"\n🔍 Evaluating {component.upper()}...")
        
        try:
            result = await evaluator.evaluate_component(
                component=component,
                transcript=test_transcript,
                speaker_role="first_proposition",
                context=context
            )
            
            results[component] = result
            
            print(f"✅ Score: {result['score']:.2f}/1.0")
            print(f"💭 Key Insight: {result['reasoning'][:150]}...")
            print(f"📋 Feedback: {result['feedback'][:120]}...")
            
            if result['examples']:
                print(f"📌 Example: {result['examples'][0][:80]}...")
                
        except Exception as e:
            print(f"❌ Error evaluating {component}: {e}")
    
    # Calculate final WSD score
    if all(comp in results for comp in components):
        print(f"\n🏆 FINAL WSD SCORE CALCULATION")
        print("=" * 40)
        
        style_score = results['style']['score']
        content_score = results['content']['score']
        strategy_score = results['strategy']['score']
        
        # Official WSD weightings: Style 40%, Content 40%, Strategy 20%
        weighted_score = (style_score * 0.4 + content_score * 0.4 + strategy_score * 0.2)
        
        # Convert to 60-80 point scale for main speeches
        final_points = 60 + (weighted_score * 20)
        
        print(f"Style (40%):    {style_score:.2f} × 0.4 = {style_score * 0.4:.2f}")
        print(f"Content (40%):  {content_score:.2f} × 0.4 = {content_score * 0.4:.2f}")
        print(f"Strategy (20%): {strategy_score:.2f} × 0.2 = {strategy_score * 0.2:.2f}")
        print("-" * 40)
        print(f"Weighted Total: {weighted_score:.2f}")
        print(f"Final Score:    {final_points:.1f}/80 points")
        
        # Performance category
        if final_points >= 78:
            category = "Outstanding"
        elif final_points >= 75:
            category = "Strong"
        elif final_points >= 72:
            category = "Solid"
        elif final_points >= 68:
            category = "Adequate"
        elif final_points >= 64:
            category = "Below Average"
        else:
            category = "Weak"
            
        print(f"Performance:    {category}")
    
    print(f"\n🎉 LLM Evaluation Testing Complete!")
    print("\n🔧 Key Advantages of LLM-Based Evaluation:")
    print("✅ Intelligent analysis beyond keyword matching")
    print("✅ Contextual understanding of argument quality")
    print("✅ Role-specific strategic assessment")
    print("✅ Nuanced scoring with detailed reasoning")
    print("✅ Actionable feedback for improvement")
    print("✅ Consistent application of official WSD criteria")

if __name__ == "__main__":
    print("🚀 Starting LLM-Based WSD Evaluation Test")
    
    try:
        asyncio.run(test_llm_evaluation())
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        print("💡 Make sure you have internet connection for Hack Club AI API")
