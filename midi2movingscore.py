#!/usr/bin/env python3
"""
MIDI to Moving Score Converter
Converts MIDI files to moving score visualizations.
"""

import argparse
import os
import sys

from music21 import converter, key, stream
import verovio
from score2movie import generate_movie, VideoType

def main(score_file_path: str, performance_midi_path: str | None = None, score_midi_path: str | None = None) -> None:
    """
    Main function to process MIDI file.
    
    Args:
        score_file_path: Absolute file path to the score file (MIDI or MusicXML)
    """
    # Validate that the file exists
    if not os.path.exists(score_file_path):
        print(f"Error: File not found: {score_file_path}", file=sys.stderr)
        sys.exit(1)
    
    # Validate that it's a file (not a directory)
    if not os.path.isfile(score_file_path):
        print(f"Error: Path is not a file: {score_file_path}", file=sys.stderr)
        sys.exit(1)
    
    # Validate file extension
    if not score_file_path.lower().endswith(('.mid', '.midi', '.xml', '.musicxml', '.mxl')):
        print(
            f"Warning: Score file does not look like MIDI/MusicXML (.mid/.midi/.xml/.musicxml/.mxl): {score_file_path}",
            file=sys.stderr,
        )
    
    print(f"Processing score file: {score_file_path}")
    
    try:
        # Parse the MIDI file
        # midi_info = parse_midi_file(midi_file_path)
        
        # # Print extracted information
        # print_midi_info(midi_info)
        
        # # Create the video
        # output_path = os.path.splitext(midi_file_path)[0] + ".mp4"
        # print(f"\nCreating video: {output_path}")
        # create_video(midi_info, output_path)
        # print(f"Video created successfully: {output_path}")

        score = converter.parse(score_file_path)
        # score.makeNotation(inPlace=True)
        output_path = os.path.splitext(score_file_path)[0] + ".mp4"
        generate_movie(
            score,
            output_path,
            video_type=VideoType.TWO_LINE_SEQUENTIAL,
            performance_midi_path=performance_midi_path,
            score_path=score_file_path,
            score_midi_path=score_midi_path,
        )
        
        
    except Exception as e:
        print(f"Error processing MIDI file: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a score file (MIDI or MusicXML) to a moving score visualization"
    )
    parser.add_argument(
        "score_file",
        type=str,
        help="Absolute file path to the score file (.mid/.midi/.xml/.musicxml/.mxl)"
    )
    parser.add_argument(
        "--score-midi",
        type=str,
        default=None,
        help="Optional absolute path to a score MIDI file for score–performance alignment",
    )
    parser.add_argument(
        "--performance-midi",
        type=str,
        default=None,
        help="Optional absolute path to a performance MIDI file for score–performance alignment",
    )
    
    args = parser.parse_args()
    main(args.score_file, args.performance_midi, args.score_midi)
