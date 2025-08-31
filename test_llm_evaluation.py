#!/usr/bin/env python3
"""
Test script for LLM-based World Schools Debate evaluation system.
Demonstrates the intelligent scoring capabilities using Hack Club AI.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.judge.wsd_rubric import LLMEvaluator, SpeakerRole

async def test_llm_evaluation():
    """Test the LLM-based evaluation system with sample debate content."""
    
    print("🎯 Testing AI-Powered World Schools Debate Judge")
    print("=" * 60)
    
    evaluator = LLMEvaluator()
    
    # Sample debate transcripts for testing
    test_cases = [
        {
            "name": "Strong First Proposition Speech",
            "transcript": """
            Good morning judges. Today I stand before you to argue that artificial intelligence 
            will fundamentally improve education for all students worldwide.
            
            Let me begin by defining our terms. When we say artificial intelligence in education, 
            we mean adaptive learning systems that personalize instruction based on individual 
            student needs and learning patterns.
            
            Our case rests on three fundamental pillars. First, AI enables personalized learning 
            at scale. Research from Stanford University shows that students using AI-powered 
            tutoring systems improved their math scores by 34% compared to traditional methods. 
            This is because AI can identify exactly where each student struggles and provide 
            targeted support.
            
            Second, AI democratizes access to quality education. In rural areas of Kenya, 
            AI-powered tablets have brought world-class instruction to students who previously 
            had no access to qualified teachers. The impact is transformational - literacy 
            rates in these communities have doubled in just two years.
            
            Third, AI frees teachers to focus on what they do best - inspiring and mentoring 
            students. When AI handles routine tasks like grading and progress tracking, teachers 
            can spend more time on creative lesson planning and one-on-one student support.
            
            The opposition will likely argue about job displacement and privacy concerns. 
            However, the evidence shows that AI augments rather than replaces teachers, 
            and proper data governance frameworks can address privacy issues.
            
            In conclusion, artificial intelligence represents the most significant opportunity 
            to improve educational outcomes in our lifetime. We must embrace this technology 
            to ensure every child receives the personalized, high-quality education they deserve.
            """,
            "role": "first_proposition",
            "component": "content"
        },
        {
            "name": "Weak Strategy Example",
            "transcript": """
            Um, so like, I think technology is bad for schools. First point is that computers 
            are expensive. Second point is that kids spend too much time on screens already. 
            Third point is that teachers are better than robots.
            
            So yeah, computers cost money and schools don't have money. Also, kids play games 
            instead of learning. And teachers are humans so they understand kids better.
            
            The other team is wrong because technology is not good. We should keep schools 
            the way they are because that's how it's always been done.
            
            In conclusion, technology is bad for education. Thank you.
            """,
            "role": "second_opposition",
            "component": "strategy"
        },
        {
            "name": "Mixed Quality Speech",
            "transcript": """
            Honorable judges, my worthy opponents have presented a compelling case, but I believe 
            they have fundamentally misunderstood the core issue at stake today.
            
            While they focus on the potential benefits of AI in education, they ignore the 
            critical question of equity and access. Yes, AI can personalize learning, but 
            who gets access to this technology? The evidence shows that AI educational tools 
            are primarily available to wealthy schools and districts, actually widening the 
            educational gap rather than closing it.
            
            Furthermore, we must consider the psychological impact on students. A study from 
            MIT found that students who relied heavily on AI tutoring systems showed decreased 
            motivation and reduced problem-solving skills when the technology was removed. 
            We're creating a generation dependent on artificial assistance.
            
            However, I want to be clear - we're not anti-technology. We simply believe that 
            human connection and traditional pedagogical methods should remain at the center 
            of education. AI should supplement, not supplant, human instruction.
            
            The path forward requires careful implementation with strong safeguards to ensure 
            equity and preserve the essential human elements of learning.
            """,
            "role": "second_opposition",
            "component": "content"
        }
    ]
    
    # Test each case
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {case['name']}")
        print("-" * 40)
        
        context = {
            'duration': 7.5,
            'word_count': len(case['transcript'].split()),
            'clarity_score': 0.85
        }
        
        try:
            result = await evaluator.evaluate_component(
                component=case['component'],
                transcript=case['transcript'],
                speaker_role=case['role'],
                context=context
            )
            
            print(f"🎯 Component: {case['component'].upper()}")
            print(f"📊 Score: {result['score']:.2f}/1.0")
            print(f"💭 Reasoning: {result['reasoning'][:200]}...")
            print(f"📋 Feedback: {result['feedback'][:150]}...")
            
            if result['examples']:
                print(f"📌 Examples: {result['examples'][0][:100]}...")
                
        except Exception as e:
            print(f"❌ Error evaluating {case['name']}: {e}")
    
    print(f"\n🎉 LLM Evaluation Testing Complete!")
    print("\n🔧 Key Features Demonstrated:")
    print("✅ Intelligent content analysis beyond keyword matching")
    print("✅ Role-specific strategic evaluation")
    print("✅ Nuanced scoring with detailed reasoning")
    print("✅ Actionable feedback for improvement")
    print("✅ Fallback mechanisms for reliability")

async def test_component_comparison():
    """Test how LLM evaluates different components of the same speech."""
    
    print("\n🔍 Component Comparison Test")
    print("=" * 40)
    
    evaluator = LLMEvaluator()
    
    # High-quality speech for multi-component analysis
    sample_speech = """
    Honorable judges, the motion before us today asks whether we should prioritize economic 
    growth over environmental protection. I stand firmly with the proposition.
    
    Let me establish our framework clearly. When we speak of economic growth, we mean sustainable 
    development that creates jobs, reduces poverty, and improves living standards. Environmental 
    protection, while important, must be balanced against immediate human needs.
    
    My first argument centers on poverty alleviation. According to the World Bank, economic 
    growth has lifted over 1 billion people out of extreme poverty in the last 30 years. 
    Environmental policies, while well-intentioned, often impose costs that disproportionately 
    burden the poor. Carbon taxes, for example, increase energy costs for families already 
    struggling to make ends meet.
    
    Second, economic growth enables environmental solutions. Wealthy nations consistently 
    outperform poor nations on environmental metrics because they can afford clean technology. 
    The Environmental Kuznets Curve demonstrates that environmental quality improves as 
    per capita income rises beyond a certain threshold.
    
    Third, we must consider the moral imperative. While climate change poses future risks, 
    poverty kills people today. Every day we delay economic development, real people suffer 
    from lack of healthcare, education, and basic necessities.
    
    I anticipate the opposition will argue about irreversible environmental damage. However, 
    human ingenuity has consistently found solutions to environmental challenges when economic 
    incentives align properly.
    
    In conclusion, prioritizing economic growth is not just economically sound - it's morally 
    necessary. We cannot sacrifice the welfare of today's poor for uncertain future benefits.
    """
    
    components = ["content", "style", "strategy"]
    context = {
        'duration': 8.2,
        'word_count': len(sample_speech.split()),
        'wpm': 165,
        'vocal_confidence': 0.8,
        'audience_focus': 75,
        'filler_words': 3
    }
    
    results = {}
    
    for component in components:
        try:
            result = await evaluator.evaluate_component(
                component=component,
                transcript=sample_speech,
                speaker_role="first_proposition",
                context=context
            )
            results[component] = result
            print(f"\n📊 {component.upper()} Score: {result['score']:.2f}")
            print(f"💡 Key Insight: {result['reasoning'][:150]}...")
            
        except Exception as e:
            print(f"❌ Error evaluating {component}: {e}")
    
    # Calculate overall WSD score
    if all(comp in results for comp in components):
        weighted_score = (
            results['style']['score'] * 0.4 +
            results['content']['score'] * 0.4 +
            results['strategy']['score'] * 0.2
        )
        final_points = 60 + (weighted_score * 20)  # Main speech: 60-80 points
        
        print(f"\n🏆 FINAL WSD SCORE")
        print(f"Style (40%): {results['style']['score']:.2f}")
        print(f"Content (40%): {results['content']['score']:.2f}")
        print(f"Strategy (20%): {results['strategy']['score']:.2f}")
        print(f"Total: {final_points:.1f}/80 points")

if __name__ == "__main__":
    print("🚀 Starting LLM-Based WSD Evaluation Tests")
    
    try:
        asyncio.run(test_llm_evaluation())
        asyncio.run(test_component_comparison())
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        print("💡 Make sure you have internet connection for Hack Club AI API")
