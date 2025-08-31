#!/usr/bin/env python3
"""
Main entry point for running the WSD AI Judge system.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.settings import settings


def setup_logging(debug: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/wsdai.log', mode='a')
        ]
    )


def run_api_server():
    """Run the FastAPI server."""
    import uvicorn
    from src.api.main import app
    
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )


def run_single_analysis(video_path: str, speaker_name: str, speaker_role: str,
                       debate_topic: str, team_side: str, is_reply_speech: bool = False):
    """Run analysis on a single video file."""
    import asyncio
    from src.core.processor import DebateAnalysisProcessor
    from src.api.models import SpeakerRoleEnum
    
    # Validate inputs
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    try:
        role_enum = SpeakerRoleEnum(speaker_role)
    except ValueError:
        print(f"Error: Invalid speaker role. Must be one of: {list(SpeakerRoleEnum)}")
        return
    
    async def process():
        processor = DebateAnalysisProcessor()
        
        def progress_callback(progress: float, message: str):
            print(f"[{progress*100:.1f}%] {message}")
        
        try:
            result = await processor.process_video(
                video_path=video_path,
                speaker_name=speaker_name,
                speaker_role=role_enum,
                debate_topic=debate_topic,
                team_side=team_side,
                is_reply_speech=is_reply_speech,
                progress_callback=progress_callback
            )
            
            # Print results
            print("\n" + "="*60)
            print("ANALYSIS RESULTS")
            print("="*60)
            print(f"Speaker: {result.speaker_name}")
            print(f"Role: {result.speaker_role.value}")
            print(f"Speech Type: {'Reply Speech (30-40 pts)' if is_reply_speech else 'Main Speech (60-80 pts)'}")
            print(f"Topic: {result.debate_topic}")
            print(f"Side: {result.team_side}")
            print(f"Processing Time: {result.processing_time:.1f}s")
            print(f"Video Duration: {result.video_duration:.1f}s")

            print("\nOFFICIAL WSD SCORES:")
            print(f"Style (40%): {result.wsd_score.manner_score:.2f}")
            print(f"Content (40%): {result.wsd_score.matter_score:.2f}")
            print(f"Strategy (20%): {result.wsd_score.method_score:.2f}")
            print(f"Total Speaker Points: {result.wsd_score.total_score:.1f}")
            
            print(f"\nOverall Feedback:")
            print(result.wsd_score.overall_feedback)
            
            print(f"\nStrengths:")
            for strength in result.wsd_score.strengths:
                print(f"- {strength}")
            
            print(f"\nAreas for Improvement:")
            for improvement in result.wsd_score.improvements:
                print(f"- {improvement}")
            
            print("\nTranscript:")
            print(result.stt_analysis.full_transcript)
            
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return
    
    # Run the analysis
    asyncio.run(process())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="WSD AI Judge System")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run the API server")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single video")
    analyze_parser.add_argument("video_path", help="Path to video file")
    analyze_parser.add_argument("--speaker-name", required=True, help="Speaker name")
    analyze_parser.add_argument("--speaker-role", required=True, 
                               choices=[role.value for role in SpeakerRoleEnum],
                               help="Speaker role")
    analyze_parser.add_argument("--debate-topic", required=True, help="Debate topic")
    analyze_parser.add_argument("--team-side", required=True,
                               choices=["Proposition", "Opposition"],
                               help="Team side")
    analyze_parser.add_argument("--reply-speech", action="store_true",
                               help="Mark as reply speech (30-40 points instead of 60-80)")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    
    # Set up logging
    setup_logging(args.debug)
    
    if args.command == "server":
        print("Starting WSD AI Judge API server...")
        run_api_server()
    
    elif args.command == "analyze":
        print("Running single video analysis...")
        run_single_analysis(
            video_path=args.video_path,
            speaker_name=args.speaker_name,
            speaker_role=args.speaker_role,
            debate_topic=args.debate_topic,
            team_side=args.team_side,
            is_reply_speech=args.reply_speech
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
