#!/usr/bin/env python3
"""
Score to Movie Converter
Converts music21 Score objects to moving score visualizations.
"""

from music21 import stream, tempo, key, clef, pitch, spanner
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoClip
from fractions import Fraction
from typing import Optional, List, Tuple

import numpy as np
import math
import random
import sys

# Constants
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
STAFF_LINE_SPACING = 18
HANDS_SPACING_OFFSET = STAFF_LINE_SPACING*3
LINE_THICKNESS = 3
X_MARGIN = 150
X_SIGN_OFFSET = 10
X_CLEF_OFFSET = X_SIGN_OFFSET
X_TIME_SIG_OFFSET = X_SIGN_OFFSET
DEFAULT_MEASURE_COUNT_PER_GROUP = 6
ACCIDENTAL_SPACE = 15
MINIMUM_DISTANCE_BETWEEN_NOTES = 2 + ACCIDENTAL_SPACE
GROUP_FADE_DURATION = 0.2
FLAT_OFFSET = -6
CHORD_STEM_X_OFFSET = -1
TIE_END_MARGIN = 5  # Margin in pixels from measure end for ties that continue beyond the measure group
SPARKLE_ANIMATION_DURATION = 0.5

PLAYHEAD_WIDTH = 3
PLAYHEAD_COLOR = (255, 0, 0)  # Red

FONT_SIZES = {
    'clef': 66,
    'time_sig': 64,
    'accidental': 56,
    'noteWithStem': 56,
    'noteWithoutStem': 62,
    'rest': 70,
    'brace': 260,
}

# Musical symbol Unicode characters (SMuFL/Bravura font)
# Standard Unicode range: U+1D100-U+1D1FF
# SMuFL Private Use Area: U+E000-U+F8FF (for extended symbols)
MUSICAL_SYMBOLS = {
    # Clefs
    'treble_clef': '\U0001D11E',      # U+1D11E: Musical Symbol G Clef (𝄞)
    'bass_clef': '\U0001D122',        # U+1D122: Musical Symbol F Clef (𝄢)
    
    # Notes with upward stems (standard Unicode)
    'whole_note': '\U0001D15D',      # U+1D15D: Musical Symbol Whole Note (𝅝)
    'half_note': '\U0001D15E',        # U+1D15E: Musical Symbol Half Note (𝅗𝅥)
    'quarter_note': '\U0001D15F',     # U+1D15F: Musical Symbol Quarter Note (𝅘𝅥)
    'eighth_note': '\U0001D160',      # U+1D160: Musical Symbol Eighth Note (𝅘𝅥𝅮)
    'sixteenth_note': '\U0001D161',   # U+1D161: Musical Symbol Sixteenth Note (𝅘𝅥𝅯)
    
    # Notes with downward stems (SMuFL PUA)
    'quarter_note_down': '\uE1D6',    
    'eighth_note_down': '\uE1D8',     # U+E1D6: Eighth note with stem down
    'sixteenth_note_down': '\uE1DA',  # U+E1D7: Sixteenth note with stem down
    'thirty_second_note_down': '\uE1DC',  # U+E1D8: Thirty-second note with stem down
    'sixty_fourth_note_down': '\uE1DE',   # U+E1D9: Sixty-fourth note with stem down
    'half_note_down': '\uE1D4',       # U+E1D4: Half note with stem down
    
    # Noteheads (for chords or manual stem drawing if needed)
    'notehead_black': '\uE0A4',   # U+1D158: Notehead Black (𝅘)
    'notehead_half': '\uE0A3',    # U+1D159: Notehead Half (𝅙)
    'notehead_whole': '\uE0A2',   # U+1D15A: Notehead Whole (𝅚)
    
    # Accidentals
    'sharp': '\u266F',                 # U+266F: Music Sharp Sign (♯)
    'flat': '\u266D',                  # U+266D: Music Flat Sign (♭)
    'natural': '\u266E',               # U+266E: Music Natural Sign (♮)
    
    # Barlines
    'barline_single': '\U0001D100',    # U+1D100: Single Barline
    'barline_double': '\U0001D101',    # U+1D101: Double Barline
    
    # Rests
    'whole_rest': '\uE4E3',        # U+1D13B: Whole Rest (𝄻)
    'half_rest': '\uE4E4',         # U+1D13C: Half Rest (𝄼)
    'quarter_rest': '\uE4E5',      # U+1D13D: Quarter Rest (𝄽)
    'eighth_rest': '\uE4E6',       # U+1D13E: Eighth Rest (𝄾)
    'sixteenth_rest': '\uE4E7',     # U+1D13F: Sixteenth Rest (𝄿)

    'augmentation_dot': '\uE1E7',
    'brace': '\uE000',

    # Time Signatures
    '2': '\uE082',
    '3': '\uE083',
    '4': '\uE084',
    '5': '\uE085',
    '6': '\uE086',
    '7': '\uE087',
    '8': '\uE088',
    '9': '\uE089',
}

NOTE_UP_MAP = {
    'whole': MUSICAL_SYMBOLS['whole_note'],
    'half': MUSICAL_SYMBOLS['half_note'],
    'quarter': MUSICAL_SYMBOLS['quarter_note'],
    'eighth': MUSICAL_SYMBOLS['eighth_note'],
    'sixteenth': MUSICAL_SYMBOLS['sixteenth_note'],
    'thirty-second': MUSICAL_SYMBOLS['sixteenth_note'],
    'dotted-half': MUSICAL_SYMBOLS['half_note'],
    'dotted-quarter': MUSICAL_SYMBOLS['quarter_note'],
    'dotted-eighth': MUSICAL_SYMBOLS['eighth_note'],
    'sixty-fourth': MUSICAL_SYMBOLS['sixteenth_note'],
    'double-whole': MUSICAL_SYMBOLS['whole_note'],
}
NOTE_DOWN_MAP = {
    'whole': MUSICAL_SYMBOLS['whole_note'],
    'half': MUSICAL_SYMBOLS['half_note_down'],
    'dotted-half': MUSICAL_SYMBOLS['half_note_down'],
    'quarter': MUSICAL_SYMBOLS['quarter_note_down'],
    'dotted-quarter': MUSICAL_SYMBOLS['quarter_note_down'],
    'eighth': MUSICAL_SYMBOLS['eighth_note_down'],
    'dotted-eighth': MUSICAL_SYMBOLS['eighth_note_down'],
    'sixteenth': MUSICAL_SYMBOLS['sixteenth_note_down'],
    'thirty-second': MUSICAL_SYMBOLS['sixteenth_note_down'],
    'sixty-fourth': MUSICAL_SYMBOLS['sixteenth_note_down'],
    'double-whole': MUSICAL_SYMBOLS['whole_note'],
}
NOTE_HEAD_MAP = {
    'whole': MUSICAL_SYMBOLS['whole_note'],
    'half': MUSICAL_SYMBOLS['half_note'],
    'half': MUSICAL_SYMBOLS['notehead_half'],
    'dotted-half': MUSICAL_SYMBOLS['notehead_half'],
    'quarter': MUSICAL_SYMBOLS['notehead_black'],
    'eighth': MUSICAL_SYMBOLS['notehead_black'],
    'sixteenth': MUSICAL_SYMBOLS['notehead_black'],
    'thirty-second': MUSICAL_SYMBOLS['notehead_black'],
    'sixty-fourth': MUSICAL_SYMBOLS['notehead_black'],
    'dotted-quarter': MUSICAL_SYMBOLS['notehead_black'],
    'dotted-eighth': MUSICAL_SYMBOLS['notehead_black'],
}

# Rest symbol mapping
REST_MAP = {
    'whole': MUSICAL_SYMBOLS['whole_rest'],
    'double-whole': MUSICAL_SYMBOLS['whole_rest'],
    'half': MUSICAL_SYMBOLS['half_rest'],
    'dotted-half': MUSICAL_SYMBOLS['half_rest'],
    'quarter': MUSICAL_SYMBOLS['quarter_rest'],
    'dotted-quarter': MUSICAL_SYMBOLS['quarter_rest'],
    'eighth': MUSICAL_SYMBOLS['eighth_rest'],
    'dotted-eighth': MUSICAL_SYMBOLS['eighth_rest'],
    'sixteenth': MUSICAL_SYMBOLS['sixteenth_rest'],
    'thirty-second': MUSICAL_SYMBOLS['sixteenth_rest'],
    'sixty-fourth': MUSICAL_SYMBOLS['sixteenth_rest'],
}

def draw_note_sparkle(
    draw: ImageDraw.Draw,
    x: float,
    y: float,
    t: float,
    duration: float = SPARKLE_ANIMATION_DURATION,
    seed: int = 0,
    num_sparks: int = 24,
    max_radius: float = 35.0,
):
    """
    Draw a sparkle/starburst animation snapshot using a PIL ImageDraw object.

    Parameters:
        draw        : PIL.ImageDraw.Draw (must be RGBA-capable)
        x, y        : center position (note head)
        t           : current animation time (seconds)
        duration    : total animation duration
        seed        : deterministic seed per note
        num_sparks  : number of spark rays
        max_radius  : how far sparks travel
    """

    if t < 0 or t > duration:
        return
    x += 8
    # Normalize time [0..1]
    p = t / duration

    # Ease-out curve
    ease = 1 - (1 - p) ** 3

    rng = random.Random(seed)

    for i in range(num_sparks):
        angle = (2 * math.pi / num_sparks) * i + rng.uniform(-0.1, 0.1)
        radius = ease * max_radius * rng.uniform(0.7, 1.0)

        sx = x + math.cos(angle) * radius
        sy = y + math.sin(angle) * radius

        spark_size = max(2, int(4 * (1 - p)))
        alpha = int(255 * (1 - p))

        draw.ellipse(
            (
                sx - spark_size,
                sy - spark_size,
                sx + spark_size,
                sy + spark_size,
            ),
            fill=(225, 215, 0, alpha),
        )

    # Central flash
    center_size = int(6 * (1 - p) + 2)
    center_alpha = int(200 * (1 - p))

    draw.ellipse(
        (
            x - center_size,
            y - center_size,
            x + center_size,
            y + center_size,
        ),
        fill=(225, 215, 0, center_alpha),
    )


def is_dotted(note_type: str) -> bool:
    """
    Check if a note or rest type is dotted.
    
    Args:
        note_type: Note or rest type string (e.g., 'dotted-half', 'quarter', etc.)
    
    Returns:
        True if the type is dotted, False otherwise
    """
    return note_type and 'dotted' in note_type.lower()

def draw_augmentation_dot(draw: ImageDraw.Draw, x: int, y: float, font: ImageFont.FreeTypeFont) -> None:
    """
    Draw an augmentation dot next to a note or rest.
    
    Args:
        draw: PIL ImageDraw object
        x: X position for the dot (to the right of the note/rest)
        y: Y position for the dot (typically in a space between staff lines)
        font: Font to use for drawing the dot
    """
    dot_char = MUSICAL_SYMBOLS['augmentation_dot']
    draw.text((x, y), dot_char, fill=(0, 0, 0), font=font, anchor='lt')

def draw_brace(draw: ImageDraw.Draw, x: int, y: float, font: ImageFont.FreeTypeFont) -> None:
    brace_char = MUSICAL_SYMBOLS['brace']
    # Render brace at base size
    bbox = draw.textbbox((0, 0), brace_char, font=font)
    brace_width = bbox[2] - bbox[0]
    brace_height = bbox[3] - bbox[1]
    draw.text((x - brace_width, y), brace_char, fill=(0, 0, 0), font=font, anchor='lt')


# Key signature spacing constants
KEY_SIGNATURE_ACCIDENTAL_SPACING = 1  # Pixels between accidentals in key signature
MAX_KEY_SIGNATURE_ACCIDENTALS = 7  # Maximum sharps/flats (A# minor or Cb major)

FPS = 15

def load_music_font(size: int):
    # Bravura font paths (SMuFL compliant)
    bravura_paths = [
        "/Users/theo/Library/Fonts/Bravura.otf",
        "/Users/theo/Library/Fonts/Bravura.ttf",
        "/System/Library/Fonts/Bravura.otf",
        "/Library/Fonts/Bravura.otf",
    ]
    
    for path in bravura_paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    
    raise FileNotFoundError("Bravura font not found. Please install Bravura font.")

def calculate_key_signature_width(draw: ImageDraw.Draw, keySignature: key.KeySignature) -> float:
    """
    Calculate the estimated pixel width needed to draw the given key signature.
    
    Key signatures are drawn with accidentals stacked vertically with spacing.
    The width depends on the number of sharps or flats in the key signature.
    
    Args:
        draw: PIL ImageDraw object for measuring text
        keySignature: key.KeySignature object from music21
        
    Returns:
        Total width in pixels needed for the key signature (0 if no accidentals)
    """
    # Get the number of sharps (positive) or flats (negative)
    sharps = keySignature.sharps
    
    # If no accidentals (C major or A minor), return 0
    if sharps == 0:
        return 0.0
    
    accidental_font = load_music_font(FONT_SIZES['accidental'])
    
    # Determine which accidental to use and get its width
    if sharps > 0:
        # Use sharp symbol for sharps
        accidental_char = '\u266F'  # ♯
    else:
        # Use flat symbol for flats
        accidental_char = '\u266D'  # ♭
    
    # Measure width of a single accidental
    accidental_bbox = draw.textbbox((0, 0), accidental_char, font=accidental_font)
    single_accidental_width = accidental_bbox[2] - accidental_bbox[0]
    
    # Calculate total width: single accidental width + spacing between accidentals
    # For n accidentals, there are (n-1) gaps between them
    num_accidentals = abs(sharps)
    total_width = single_accidental_width + (num_accidentals - 1) * KEY_SIGNATURE_ACCIDENTAL_SPACING
    
    return total_width

class MeasureGroupInfo:
    keySignature: key.KeySignature
    timeSignature: Fraction

    def __init__(self, keySignature: key.KeySignature, timeSignature: tuple[int, int]) -> None:
        self.keySignature = keySignature
        self.timeSignature = timeSignature

def midi_to_staff_position(step: str, octave: int, is_treble: bool) -> float:
    """
    Convert step and octave to staff position.
    Position 0 is the bottom line of the staff.
    Negative values are below the staff, positive above.
    
    Args:
        step: Step name (e.g., 'A', 'B', 'C', 'D', 'E', 'F', 'G')
        octave: Octave number (integer)
        is_treble: True for treble clef, False for bass clef
    
    Returns:
        Position in staff line units (0 = bottom line, 0.5 = space, 1 = next line)
    """
    # Map step letters to their position relative to the reference note within the natural note sequence
    # Natural note sequence: C, D, E, F, G, A, B (repeating)
    # For treble: E4 is at position 0, so E=0, F=0.5, G=1.0, A=1.5, B=2.0, C=2.5 (but C4 is -1.0), D=3.0 (but D4 is -0.5)
    # For bass: G2 is at position 0, so G=0, A=0.5, B=1.0, C=1.5, D=2.0, E=2.5, F=3.0
    if is_treble:
        # Treble clef: E4 is on bottom line (position 0)
        # Middle C (C4) is one ledger line below (position -1)
        # Map steps to their position relative to E in the sequence E, F, G, A, B, C, D
        step_to_offset_from_reference = {
            'E': 0.0,   # E is the reference
            'F': 0.5,   # F is 0.5 above E
            'G': 1.0,   # G is 1.0 above E
            'A': 1.5,   # A is 1.5 above E
            'B': 2.0,   # B is 2.0 above E
            'C': -1.0,  # C is 1.0 below E (in same octave)
            'D': -0.5   # D is 0.5 below E (in same octave)
        }
        reference_step = 'E'
        reference_octave = 4
        reference_position = 0.0
    else:
        # Bass clef: G2 is on bottom line (position 0)
        # Middle C (C4) is one ledger line above (position 10)
        # Map steps to their position relative to G in the sequence G, A, B, C, D, E, F
        step_to_offset_from_reference = {
            'G': 0.0,   # G is the reference
            'A': 0.5,   # A is 0.5 above G
            'B': 1.0,   # B is 1.0 above G
            'C': -2.0,
            'D': -1.5,
            'E': -1.0,
            'F': -0.5
        }
        reference_step = 'G'
        reference_octave = 2
        reference_position = 0.0
    
    if step not in step_to_offset_from_reference:
        raise ValueError(f"Invalid step: {step}. Must be one of C, D, E, F, G, A, B")
    
    # Get the offset from reference within the same octave
    offset_from_reference = step_to_offset_from_reference[step]
    
    # Calculate octave difference
    octave_diff = octave - reference_octave
    
    # Each octave spans 3.5 staff positions (from reference note to same note in next octave)
    # For treble: E4 to E5 spans 3.5 positions
    # For bass: G2 to G3 spans 3.5 positions
    octave_span = 3.5
    
    # Calculate final position
    # Base position from reference, plus octave difference, plus offset within octave
    position = reference_position + octave_diff * octave_span + offset_from_reference
    
    return position

def draw_accidental(draw: ImageDraw.Draw, x: int, y: float, accidental, accidental_font: ImageFont.FreeTypeFont) -> None:
    """
    Draw an accidental symbol if it exists and should be displayed.
    
    Args:
        draw: PIL ImageDraw object
        x: X position for the note
        y: Y position for the note (center of notehead)
        accidental: music21 accidental object (or None)
        accidental_font: Font to use for drawing the accidental
    """
    if accidental and accidental.displayStatus == True:
        acc_char = {
            'sharp': MUSICAL_SYMBOLS['sharp'],
            'flat': MUSICAL_SYMBOLS['flat'],
            'natural': MUSICAL_SYMBOLS['natural']
        }.get(accidental.fullName, '')
        
        if acc_char:
            acc_bbox = draw.textbbox((0, 0), acc_char, font=accidental_font)
            acc_height = acc_bbox[3] - acc_bbox[1]
            acc_y_offset = {
                'sharp': -acc_height/2,
                'flat': -acc_height/2 + FLAT_OFFSET,
                'natural': -acc_height/2
            }.get(accidental.fullName, 0)
            acc_y = y + acc_y_offset
            draw.text((x - ACCIDENTAL_SPACE, acc_y), acc_char, fill=(0, 0, 0), 
                     font=accidental_font, anchor='lt')

def duration_to_rest_type(duration_quarters: float) -> str:
    """
    Convert a duration in quarter notes to a rest type string.
    
    Args:
        duration_quarters: Duration in quarter notes
        
    Returns:
        Rest type string (e.g., 'whole', 'half', 'quarter', etc.)
    """
    # Map durations to rest types (in quarter notes)
    if duration_quarters >= 4.0:
        return 'whole'
    elif duration_quarters >= 3.0:
        return 'dotted-half'
    elif duration_quarters >= 2.0:
        return 'half'
    elif duration_quarters >= 1.5:
        return 'dotted-quarter'
    elif duration_quarters >= 1.0:
        return 'quarter'
    elif duration_quarters >= 0.75:
        return 'dotted-eighth'
    elif duration_quarters >= 0.5:
        return 'eighth'
    elif duration_quarters >= 0.375:
        return 'dotted-sixteenth'
    elif duration_quarters >= 0.25:
        return 'sixteenth'
    else:
        return 'sixteenth'  # Default for very short durations

def draw_rest(draw: ImageDraw.Draw, x: int, staff_y: int, rest_type: str, is_treble: bool) -> None:
    """
    Draw a rest symbol on the staff.
    
    Args:
        draw: PIL ImageDraw object
        x: X position for the rest (center position)
        staff_y: Top Y position of the staff
        rest_type: Type of rest (e.g., 'whole', 'half', 'quarter', etc.)
        is_treble: True for treble clef, False for bass clef
    """
    rest_font = load_music_font(FONT_SIZES['rest'])
    
    # Get rest symbol
    rest_char = REST_MAP.get(rest_type, MUSICAL_SYMBOLS['quarter_rest'])
    
    # Get bounding box to position the rest
    bbox = draw.textbbox((0, 0), rest_char, font=rest_font)
    rest_width = bbox[2] - bbox[0]
    rest_height = bbox[3] - bbox[1]
    
    # Position rests on the middle line of the staff (position 2.0)
    # For treble: middle line is at staff_y + 2 * STAFF_LINE_SPACING
    # For bass: middle line is at staff_y + 2 * STAFF_LINE_SPACING
    middle_line_y = staff_y + 2 * STAFF_LINE_SPACING
    
    # Whole and half rests hang from the middle line
    # Quarter and shorter rests sit on the middle line
    if rest_type == 'whole' or rest_type == 'double-whole':
        # Whole rest hangs on the second top line
        rest_y = staff_y + STAFF_LINE_SPACING
    elif rest_type == 'half' or rest_type == 'dotted-half':
        # Half rest sits on the middle line
        rest_y = middle_line_y - rest_height
    else:
        # Quarter and shorter rests are centered around the middle line
        rest_y = middle_line_y - rest_height / 2
    
    # Center the rest horizontally at the given x position
    draw.text((x, rest_y), rest_char, fill=(0, 0, 0), font=rest_font, anchor='lt')
    
    # Draw augmentation dot if rest is dotted
    if is_dotted(rest_type):
        # Position dot to the right of the rest, in a space between staff lines
        dot_x = x + rest_width + 2  # Small spacing after rest
        # Position dot in the middle of a staff space (typically the space above the middle line)
        # Use the space above the middle line (position 2.5)
        dot_y = middle_line_y - STAFF_LINE_SPACING / 2
        draw_augmentation_dot(draw, dot_x, dot_y, rest_font)

def draw_ledger_lines(draw: ImageDraw.Draw, note_x: int, staff_y: int, note_width: int, staff_position: int) -> None:
    """
    Draw ledger lines for notes outside the staff.
    """
    # Draw ledger lines for notes outside the staff
    if staff_position <= -1:
        # Below staff - draw ledger lines above
        for i in range(1, int(abs(staff_position))+1):
            line_y = staff_y + STAFF_LINE_SPACING*(4 + i)
            draw.line([(note_x - 8, line_y), (note_x + note_width + 8, line_y)], 
                        fill=(0, 0, 0), width=LINE_THICKNESS)
    elif staff_position >= 5:
        # Above staff - draw ledger lines below
        for i in range(1, int(staff_position - 4)+1):
            line_y = staff_y - STAFF_LINE_SPACING*i
            draw.line([(note_x - 8, line_y), (note_x + note_width + 8, line_y)], 
                        fill=(0, 0, 0), width=LINE_THICKNESS)

def draw_tie(draw: ImageDraw.Draw, startAnchor: Tuple[int, int], endAnchor: Tuple[int, int], xLimitStart: int, xLimitEnd: int, curveDown: bool = False) -> None:
    """
    Draw a tie using a cubic Bezier curve with four points.
    By default, ties curve upward. If curveDown is True, the tie curves downward.
    
    Args:
        draw: PIL ImageDraw object
        startAnchor: Tuple[int, int] - Starting point
        endAnchor: Tuple[int, int] - Ending point
        xLimitStart: int - Absolute leftbound X limit. Before this point the tie curve is not drawn.
        xLimitEnd: int - Absolute rightbound X limit. After this point the tie curve is not drawn.
        curveDown: bool - If True, curve downward (for bottom note in chord). Default False (curves upward).
    """
    # Calculate four points
    P0 = startAnchor
    tieLengthX = endAnchor[0] - startAnchor[0]
    tieMinHeight = 10
    tieMaxHeight = 20
    proportionalHeight = min(max(tieLengthX / 4, tieMinHeight), tieMaxHeight)
    
    # Ties curve upward by default (negative height), downward if curveDown is True (positive height)
    if curveDown:
        height = proportionalHeight
    else:
        height = -proportionalHeight
    
    P1 = (startAnchor[0] + tieLengthX/3, startAnchor[1] + height)
    P2 = (endAnchor[0] - tieLengthX/3, endAnchor[1] + height)
    P3 = endAnchor

    # Draw the cubic Bezier curve with professional looking thickness and tapering.
    # Professional slurs are thicker in the middle and taper to thin ends.
    
    # Parameters for thickness tapering
    maxThickness = 4.0  # Maximum thickness in the middle
    minThickness = 1.0  # Minimum thickness at the ends
    numSamples = 100    # Number of points to sample along the curve
    
    # Sample points along the Bezier curve
    topPoints = []
    bottomPoints = []
    
    for i in range(numSamples + 1):
        t = i / numSamples
        
        # Cubic Bezier curve formula: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
        mt = 1 - t
        x = mt**3 * P0[0] + 3 * mt**2 * t * P1[0] + 3 * mt * t**2 * P2[0] + t**3 * P3[0]
        y = mt**3 * P0[1] + 3 * mt**2 * t * P1[1] + 3 * mt * t**2 * P2[1] + t**3 * P3[1]
        
        # Calculate tangent vector (derivative of Bezier curve)
        # B'(t) = 3(1-t)²(P1-P0) + 6(1-t)t(P2-P1) + 3t²(P3-P2)
        dx = 3 * mt**2 * (P1[0] - P0[0]) + 6 * mt * t * (P2[0] - P1[0]) + 3 * t**2 * (P3[0] - P2[0])
        dy = 3 * mt**2 * (P1[1] - P0[1]) + 6 * mt * t * (P2[1] - P1[1]) + 3 * t**2 * (P3[1] - P2[1])
        
        # Normalize tangent to get unit vector
        tangentLength = np.sqrt(dx**2 + dy**2)
        if tangentLength > 0:
            dx /= tangentLength
            dy /= tangentLength
        else:
            dx, dy = 0, 1
        
        # Calculate thickness using a smooth taper function
        # Thickness is maximum at t=0.5 and minimum at t=0 and t=1
        # Using a smooth curve: thickness = minThickness + (maxThickness - minThickness) * 4t(1-t)
        thickness = minThickness + (maxThickness - minThickness) * 4 * t * (1 - t)
        
        # Calculate perpendicular vector (rotate tangent 90 degrees)
        # Perpendicular to (dx, dy) is (-dy, dx)
        perpX = -dy
        perpY = dx
        
        # Calculate offset points above and below the curve
        offset = thickness / 2
        topX = x + perpX * offset
        topY = y + perpY * offset
        bottomX = x - perpX * offset
        bottomY = y - perpY * offset
        
        # Only add points within the xLimit bounds
        if xLimitStart <= x <= xLimitEnd:
            topPoints.append((int(topX), int(topY)))
            bottomPoints.append((int(bottomX), int(bottomY)))
    
    # Draw the tie by filling the area between top and bottom curves
    if len(topPoints) > 1 and len(bottomPoints) > 1:
        # Create polygon points: top curve + reversed bottom curve
        polygonPoints = topPoints + list(reversed(bottomPoints))
        draw.polygon(polygonPoints, fill=(0, 0, 0))


def draw_note(draw: ImageDraw.Draw, x: int, pitch: pitch.Pitch, note_type: str, 
              is_treble: bool, staff_y: int, keySignature: key.KeySignature) -> None:
    """
    Draw a single note on the staff using Noto Music font glyphs.
    Follows engraving rules for stem direction and positioning.
    
    Args:
        draw: PIL ImageDraw object
        x: X position
        midi_note: MIDI note number
        note_type: Type of note (quarter, half, whole, etc.)
        is_treble: True for treble clef, False for bass clef
        staff_y: Top Y position of the staff
        key_sig: Current key signature tuple or None
    """
    note_font = load_music_font(FONT_SIZES['noteWithStem'])
    accidental_font = load_music_font(FONT_SIZES['accidental'])
    
    # Calculate staff position
    staff_position = midi_to_staff_position(pitch.step, pitch.octave, is_treble)
    
    # Convert to actual Y coordinate
    # Staff has 5 lines (4 spaces), position 0 is bottom line
    actual_y = staff_y + (4 * STAFF_LINE_SPACING) - (staff_position * STAFF_LINE_SPACING)
    
    # Get note symbol (default to quarter note if not found)
    note_type_clean = note_type.strip() if note_type else 'quarter'
    note_char = NOTE_UP_MAP.get(note_type_clean, MUSICAL_SYMBOLS['quarter_note'])
    
    # Warn if note type not found
    if note_type_clean not in NOTE_UP_MAP:
        print(f"Warning: Unknown note type '{note_type_clean}' (original: '{note_type}'), using quarter note symbol", file=sys.stderr)
    
    # Get bounding box to center the note
    bbox = draw.textbbox((0, 0), note_char, font=note_font)
    note_width = bbox[2] - bbox[0]
    note_height = bbox[3] - bbox[1]
    
    # Check if accidental is needed
    accidental = pitch.accidental
    draw_accidental(draw, x, actual_y, accidental, accidental_font)
    
    # Determine stem direction based on staff position
    # Middle line is at position 2.0 (third line, counting from bottom as 0, 1, 2, 3, 4)
    middle_line_pos = 2.0
    stem_up = staff_position < middle_line_pos  # Stem up if below middle line
    
    # Draw note head
    note_x = x
    note_y = actual_y - note_height + STAFF_LINE_SPACING/2
    
    # For notes above middle line (stem down), use SMuFL downward stem symbols
    if note_type_clean == 'whole' or note_type_clean == 'double-whole':
        note_y = actual_y - note_height/2
        draw.text((note_x, note_y), note_char, fill=(0, 0, 0), font=note_font, anchor='lt')
    elif stem_up:
        note_y = actual_y - note_height + STAFF_LINE_SPACING/2
        draw.text((note_x, note_y), note_char, fill=(0, 0, 0), font=note_font, anchor='lt')
    else:
        note_char_down = NOTE_DOWN_MAP.get(note_type_clean, MUSICAL_SYMBOLS['quarter_note_down'])
        note_bbox = draw.textbbox((0, 0), note_char_down, font=note_font)
        note_width = note_bbox[2] - note_bbox[0]
        note_char_nonstem = NOTE_HEAD_MAP.get(note_type_clean, MUSICAL_SYMBOLS['notehead_black'])
        note_bbox = draw.textbbox((0, 0), note_char_nonstem, font=note_font)
        note_head_height = note_bbox[3] - note_bbox[1]
        note_y = actual_y - note_head_height/2
        draw.text((note_x, note_y), note_char_down, fill=(0, 0, 0), font=note_font, anchor='lt')

    # Draw augmentation dot if note is dotted
    if is_dotted(note_type_clean):
        # Position dot to the right of the note, in a space between staff lines
        dot_x = note_x + note_width + 2  # Small spacing after note
        # Position dot in the space above the notehead
        # If note is on a line (integer position), use space above (position + 0.5)
        # If note is in a space (half position), use the same space
        if staff_position % 1 == 0:  # On a line
            dot_position = staff_position + 0.5
        else:  # In a space
            dot_position = staff_position
        dot_y = staff_y + (4 * STAFF_LINE_SPACING) - (dot_position * STAFF_LINE_SPACING) - LINE_THICKNESS
        draw_augmentation_dot(draw, dot_x, dot_y, note_font)
    
    # Draw ledger lines for notes outside the staff
    draw_ledger_lines(draw, note_x, staff_y, note_width, staff_position)

    return actual_y

def draw_chord(draw: ImageDraw.Draw, x: int, pitches: List, is_treble: bool, 
               staff_y: int, keySignature: key.KeySignature, noteType: str, clef: clef.Clef) -> None:
    """
    Draw a chord (stacked notes) with proper engraving rules.
    
    Args:
        draw: PIL ImageDraw object
        x: X position for the chord
        pitches: List of MIDI note numbers (must be sorted by pitch, lowest to highest)
        is_treble: True for treble clef, False for bass clef
        staff_y: Top Y position of the staff
        keySignature: Current key signature
        clef: Current clef
    """
    if not pitches:
        return []
    
    if len(pitches) == 1:
        # Single note - use regular drawing
        actual_y = draw_note(draw, x, pitches[0], noteType, is_treble, staff_y, keySignature)
        return [actual_y]
    
    # Sort notes by pitch (lowest to highest)
    sorted_pitches = sorted(pitches, key=lambda n: n.midi)
    
    stemNoteFont = load_music_font(FONT_SIZES['noteWithStem'])
    noneStemNoteFont = load_music_font(FONT_SIZES['noteWithoutStem'])
    accidental_font = load_music_font(FONT_SIZES['accidental'])
    
    # Calculate staff positions for all notes
    noteList = []
    actual_ys = []
    for pitch in sorted_pitches:
        staff_pos = midi_to_staff_position(pitch.step, pitch.octave, is_treble)
        actual_y = staff_y + (4 * STAFF_LINE_SPACING) - (staff_pos * STAFF_LINE_SPACING)
        actual_ys.append(actual_y)
        accidental = pitch.accidental
        noteList.append({
            'midi': pitch.midi,
            'staff_pos': staff_pos,
            'y': actual_y,
            'accidental': accidental
        })
    
    # Determine stem direction based on average position
    # Middle line is at position 2.0 (third line)
    middle_line_pos = 2.0
    avg_position = sum(note['staff_pos'] for note in noteList) / len(noteList)
    stem_up = avg_position <= middle_line_pos  # Stem up if mostly below middle line
    
    # Choose note symbol based on stem direction
    noneStemNoteChar = NOTE_HEAD_MAP.get(noteType, MUSICAL_SYMBOLS['notehead_black'])
    if stem_up or noteType == 'whole' or noteType == 'double-whole':
        # For upward stems or whole notes, use standard symbols (which include stems)
        stemNoteChar = NOTE_UP_MAP.get(noteType, MUSICAL_SYMBOLS['quarter_note'])
    else:
        stemNoteChar = NOTE_DOWN_MAP.get(noteType,  MUSICAL_SYMBOLS['quarter_note'])
    
    # Warn if note type not found
    if noteType not in NOTE_UP_MAP:
        print(f"Warning: Unknown note type '{noteType}', using quarter note symbol", file=sys.stderr)
    
    # Get note dimensions
    bbox = draw.textbbox((0, 0), noneStemNoteChar, font=noneStemNoteFont)
    noneStemNoteW = bbox[2] - bbox[0]
    noneStemNoteH = bbox[3] - bbox[1]
    bbox = draw.textbbox((0, 0), stemNoteChar, font=stemNoteFont)
    stemNoteW = bbox[2] - bbox[0]
    stemNoteH = bbox[3] - bbox[1]

    # Calculate notehead offsets for seconds (notes a second apart)
    # Notes a second apart (1 semitone = 0.5 staff positions) need horizontal offset
    notehead_offsets = [0] * len(noteList)
    for i in range(len(noteList) - 1):
        pos_diff = abs(noteList[i+1]['staff_pos'] - noteList[i]['staff_pos'])
        if pos_diff <= 0.5:  # Second apart or less
            # Offset the higher note to the right
            notehead_offsets[i+1] = noneStemNoteW - 1 # Small horizontal offset
    
    # Draw accidentals first (left to right, staggered vertically)
    # Use consistent accidental width (same as single notes: 20 pixels)
    # This ensures noteheads align at the same position for uniform spacing
    has_accidentals = any(note_info['accidental'] for note_info in noteList)
    
    if has_accidentals:        
        for i, note_info in enumerate(noteList):
            draw_accidental(draw, x, note_info['y'], note_info['accidental'], accidental_font)
    
    baseNoteX = x
    # Draw the note/noteheads.
    for i, noteInfo in enumerate(noteList):
        noteX = baseNoteX + notehead_offsets[i]
        # Need to decide if this note is the stem or not.
        isStemNote = False
        if stem_up:# and i == len(noteList) - 1:
            if len(noteList) == 2 and i == 0 and noteList[1]['staff_pos'] - noteList[0]['staff_pos'] == 0.5:
                isStemNote = True
            elif i == len(noteList) - 1 and noteList[1]['staff_pos'] - noteList[0]['staff_pos'] > 0.5:
                isStemNote = True
        elif stem_up == False and i == 0:
            isStemNote = True
        
        if noteType == 'whole' or noteType == 'double-whole':
            # For upward stems or whole notes, use standard positioning
            noteY = noteInfo['y'] - stemNoteH/2
            noteFont = stemNoteFont
        elif isStemNote:
            noteFont = stemNoteFont
            if stem_up:
                noteY = noteInfo['y'] - stemNoteH + STAFF_LINE_SPACING/2
            else:
                noteY = noteInfo['y'] - noneStemNoteH/2
        else:
            noteFont = noneStemNoteFont
            noteY = noteInfo['y'] - noneStemNoteH/2
        noteChar = stemNoteChar if isStemNote else noneStemNoteChar
        draw.text((noteX, noteY), noteChar, fill=(0, 0, 0), 
                 font=noteFont, anchor='lt')
    
    # Draw shared stem (only for notes with stems, not whole notes)
    if noteType != 'whole' and noteType != 'double-whole':
        # Find the outermost notes for stem endpoints
        topNote = noteList[-1]
        bottomNote = noteList[0]
        stemStartY = bottomNote['y']
        stemEndY = topNote['y']
        if stem_up:
            # Place stem on the right side
            stemX = x + noneStemNoteW + CHORD_STEM_X_OFFSET
        else:
            # Place stem on the left side
            stemX = x - CHORD_STEM_X_OFFSET
        
        # Draw stem line
        draw.line([(stemX, stemStartY), (stemX, stemEndY)], 
                 fill=(0, 0, 0), width=2)
        
    
    # Draw augmentation dot if note is dotted
    if is_dotted(noteType):
        # Position dot to the right of the chord, in a space between staff lines
        # Use the middle note's position to determine dot placement
        middle_note_idx = len(noteList) // 2
        middle_staff_pos = noteList[middle_note_idx]['staff_pos']
        dot_x = x + max(noneStemNoteW, stemNoteW) + 2  # Small spacing after chord
        
        # Position dot in the space above the middle note
        # If note is on a line (integer position), use space above (position + 0.5)
        # If note is in a space (half position), use the same space
        if middle_staff_pos % 1 == 0:  # On a line
            dot_position = middle_staff_pos + 0.5
        else:  # In a space
            dot_position = middle_staff_pos
        dot_y = staff_y + (4 * STAFF_LINE_SPACING) - (dot_position * STAFF_LINE_SPACING) - LINE_THICKNESS
        draw_augmentation_dot(draw, dot_x, dot_y, stemNoteFont)
    
    # Draw ledger lines for notes outside the staff
    topNote = noteList[-1]
    bottomNote = noteList[0]
    draw_ledger_lines(draw, x, staff_y, noneStemNoteW, topNote['staff_pos'])
    draw_ledger_lines(draw, x, staff_y, noneStemNoteW, bottomNote['staff_pos'])

    return actual_ys

def get_note_type(note_type: str, duration: float) -> str:
    """
    Get the note type based on the note type and duration.
    """
    if note_type == 'half' and duration == 3.0:
        return 'dotted-half'
    elif note_type == 'quarter' and duration == 1.5:
        return 'dotted-quarter'
    elif note_type == 'eighth' and duration == 0.75:
        return 'dotted-eighth'
    elif note_type == 'sixteenth' and duration == 0.375:
        return 'dotted-sixteenth'
    elif note_type == 'thirty-second' and duration == 0.1875:
        return 'dotted-thirty-second'
    else:
        return note_type

def draw_measure(
    measureGroupInfo: MeasureGroupInfo,
    draw: ImageDraw.Draw,
    measure: stream.Measure,
    xCumulated: int,
    yBottomLine: int,
    widthPerMeasure: int,
    isFirstMeasureInGroup: bool = False,
    active_ties: dict = None,
    clefChanges: list[dict] = []) -> tuple[MeasureGroupInfo, List[dict], dict]:
    # Check if this measures's clef, key signature, or time signature has changed from measureGroupInfo.
    
    # If it has changed, draw the symbols that changed and draw the notes following the new symbols. 
    # If it has not changed, draw the notes following the measureGroupInfo.
    
    # Get current clef, key signature, and time signature from the measure
    currentClef = measure.clef if measure.clef else measure.getContextByClass(clef.Clef)
    if measure.keySignature != None:
        currentKeySignature = measure.keySignature
    else:
        currentKeySignature = measureGroupInfo.keySignature
    if measure.timeSignature != None:
        currentTimeSignature = measure.timeSignature
    else:
        currentTimeSignature = measureGroupInfo.timeSignature
    
    # Check for changes
    keySignatureChanged = (currentKeySignature is not None and measureGroupInfo.keySignature is not None and
                           currentKeySignature.sharps != measureGroupInfo.keySignature.sharps)
    # measureGroupInfo.timeSignature is a tuple (numerator, denominator)
    timeSignatureChanged = (currentTimeSignature.numerator != measureGroupInfo.timeSignature.numerator or
                            currentTimeSignature.denominator != measureGroupInfo.timeSignature.denominator)
    
    # Draw changed symbols
    trebleChar = MUSICAL_SYMBOLS['treble_clef']
    bassChar = MUSICAL_SYMBOLS['bass_clef']
    changedSymbolsWidth = 0
    if currentClef.name == 'treble':
        clefChar = trebleChar
    else:
        clefChar = bassChar
    if keySignatureChanged and currentKeySignature.sharps != 0:
        # Draw new key signature
        keySignatureWidth = draw_key_signature(draw, currentKeySignature.sharps, xCumulated, yBottomLine, clefChar)
        xCumulated += keySignatureWidth
        changedSymbolsWidth += keySignatureWidth    
    if timeSignatureChanged:
        # Draw new time signature
        xCumulated += X_TIME_SIG_OFFSET
        changedSymbolsWidth += X_TIME_SIG_OFFSET
        timeSignatureFont = load_music_font(FONT_SIZES['time_sig'])
        yDenom = yBottomLine - STAFF_LINE_SPACING*4
        yNom = yBottomLine - STAFF_LINE_SPACING*2
        bboxDenom = draw.textbbox((0, 0), MUSICAL_SYMBOLS[str(currentTimeSignature.numerator)], font=timeSignatureFont)
        bboxNom = draw.textbbox((0, 0), MUSICAL_SYMBOLS[str(currentTimeSignature.denominator)], font=timeSignatureFont)
        if bboxDenom[2] - bboxDenom[0] > bboxNom[2] - bboxNom[0]:
            draw.text((xCumulated, yDenom), MUSICAL_SYMBOLS[str(currentTimeSignature.numerator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt')
            draw.text((xCumulated + (bboxDenom[2] - bboxDenom[0])//2 - (bboxNom[2] - bboxNom[0])//2, yNom), MUSICAL_SYMBOLS[str(currentTimeSignature.denominator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt')
            xCumulated += bboxDenom[2] - bboxDenom[0]
            changedSymbolsWidth += bboxDenom[2] - bboxDenom[0]
        elif bboxNom[2] - bboxNom[0] > bboxDenom[2] - bboxDenom[0]:
            draw.text((xCumulated + (bboxNom[2] - bboxNom[0])//2 - (bboxDenom[2] - bboxDenom[0])//2, yDenom), MUSICAL_SYMBOLS[str(currentTimeSignature.numerator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt')
            draw.text((xCumulated, yNom), MUSICAL_SYMBOLS[str(currentTimeSignature.denominator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt')
            xCumulated += bboxNom[2] - bboxNom[0]
            changedSymbolsWidth += bboxNom[2] - bboxNom[0]
        else:
            draw.text((xCumulated, yDenom), MUSICAL_SYMBOLS[str(currentTimeSignature.numerator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt')
            draw.text((xCumulated, yNom), MUSICAL_SYMBOLS[str(currentTimeSignature.denominator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt')
            xCumulated += bboxNom[2] - bboxNom[0]
            changedSymbolsWidth += bboxNom[2] - bboxNom[0]
    
    # Draw notes in the measure
    # Use the current (possibly updated) clef and key signature for drawing notes
    activeClef = currentClef
    activeKeySignature = currentKeySignature if keySignatureChanged else measureGroupInfo.keySignature
    
    # Calculate measure duration in quarter notes
    measureDuration = measure.duration.quarterLength
    
    # Collect all notes with their offsets and durations
    note_events = []
    for note in measure.notes:
        # Handle single notes and chords
        if hasattr(note, 'pitch'):
            # Single note
            pitches = [note.pitch]
            # For single notes, get tie from the note itself
            pitch_ties = {note.pitch.midi: note.tie if note.tie else None}
        elif hasattr(note, 'pitches'):
            # Chord - draw all pitches
            pitches = note.pitches
            # For chords, check individual note ties if available, otherwise use chord tie
            pitch_ties = {}
            if hasattr(note, 'notes') and note.notes:
                # Individual notes in chord may have their own ties
                for inner_note in note.notes:
                    if hasattr(inner_note, 'pitch') and hasattr(inner_note, 'tie'):
                        pitch_ties[inner_note.pitch.midi] = inner_note.tie if inner_note.tie else None
            # If no individual ties found, use chord tie for all pitches
            if not pitch_ties:
                chord_tie = note.tie if note.tie else None
                for pitch in pitches:
                    pitch_ties[pitch.midi] = chord_tie
        else:
            # Skip if no pitch information
            continue
        note_events.append({
            'offset': note.offset,
            'duration': note.duration.quarterLength,
            'pitches': pitches,
            'note_type': note.duration.type,
            'pitch_ties': pitch_ties,  # Dictionary mapping pitch.midi to tie object
            'clef': note.getContextByClass(clef.Clef)
        })
    
    # Sort notes by offset
    note_events.sort(key=lambda n: n['offset'])
    
    # Draw notes and rests
    current_position = 0.0
    is_treble = currentClef.name == 'treble'
    staff_y = yBottomLine - STAFF_LINE_SPACING*4
    
    # Track active ties by pitch (midi number) - stores (start_x, start_y, is_bottom_note)
    # Use provided active_ties or initialize empty dict
    if active_ties is None:
        active_ties = {}
    else:
        # Make a copy to avoid modifying the original
        active_ties = active_ties.copy()
    tie_curves = []  # List of tie curves to draw: {'start': (x, y), 'end': (x, y), 'curveDown': bool, 'xLimitStart': int, 'xLimitEnd': int}
    
    totalClefWidthChanges = 0
    for clefChange in clefChanges:
        totalClefWidthChanges += clefChange['width'] + X_CLEF_OFFSET
    availableWidth = widthPerMeasure - changedSymbolsWidth - totalClefWidthChanges
    noteKeyFrames = []
    for note_event in note_events:
        note_offset = note_event['offset']
        note_duration = note_event['duration']
        note_clef = note_event['clef']
        
        # Draw the note        
        note_type = get_note_type(note_event['note_type'], note_event['duration'])
        xPosition = xCumulated + ACCIDENTAL_SPACE + (note_offset / measureDuration) * (availableWidth - ACCIDENTAL_SPACE)
        # Before drawing the note, check for the clef change. 
        # If there is a clef change, draw the clef change symbol and adjust the x position accordingly.
        for clefChange in clefChanges:
            if note_offset == clefChange['startQuarter']:
                xCumulated += X_CLEF_OFFSET
                # Draw the clef is the clef change occurred by this hand. 
                if clefChange['newClef']:
                    clefChar = MUSICAL_SYMBOLS[clefChange['newClef'].name + '_clef']
                    bbox = draw.textbbox((0, 0), clefChar, font=load_music_font(FONT_SIZES['clef']))
                    clefY = yBottomLine - STAFF_LINE_SPACING*2 - (bbox[3] - bbox[1])/2
                    draw.text((xPosition, clefY), clefChar, fill=(0, 0, 0), font=load_music_font(FONT_SIZES['clef']), anchor='lt')
                    currentClef = clefChange['newClef']
                xCumulated += clefChange['width']
                xPosition = xCumulated + ACCIDENTAL_SPACE + (note_offset / measureDuration) * (availableWidth - ACCIDENTAL_SPACE)
        
        # Check if there's a gap before this note
        if note_offset > current_position:
            # There's a gap - draw a rest
            gap_duration = note_offset - current_position
            rest_type = duration_to_rest_type(gap_duration)
            
            # Calculate x position for the rest (centered in the gap)
            rest_x = xCumulated + ACCIDENTAL_SPACE + (current_position / measureDuration) * (availableWidth - ACCIDENTAL_SPACE)
            
            draw_rest(draw, rest_x, staff_y, rest_type, is_treble)
        actual_ys = draw_chord(draw, xPosition, note_event['pitches'], is_treble, staff_y, activeKeySignature, note_type, currentClef)

        # Store absolute score time (in quarterLength/beats). This enables optional
        # score→performance timing transfer later.
        noteKeyFrames.append({
            "t_ratio": note_offset / measureDuration,
            "x_absolute": xPosition,
            "y_middles": actual_ys,
            "score_quarter": float(measure.offset + note_offset),
        })
        
        # Handle ties for each pitch in the note/chord
        sorted_pitches = sorted(note_event['pitches'], key=lambda p: p.midi)
        pitch_ties = note_event.get('pitch_ties', {})
        
        for pitch_idx, pitch in enumerate(sorted_pitches):
            # Determine if this is the bottom note in a chord
            is_bottom_note = (len(sorted_pitches) > 1 and pitch_idx == 0)
            
            # Calculate Y position for this pitch
            staff_pos = midi_to_staff_position(pitch.step, pitch.octave, is_treble)
            pitch_y = staff_y + (4 * STAFF_LINE_SPACING) - (staff_pos * STAFF_LINE_SPACING)
            
            # Get tie information for this pitch
            tie = pitch_ties.get(pitch.midi)
            tie_type = None
            if tie:
                tie_type = tie.type
            
            # Handle tie tracking
            pitch_key = pitch.midi
            
            if tie_type == 'start':
                # Start a new tie
                active_ties[pitch_key] = {
                    'start_x': xPosition,
                    'start_y': pitch_y,
                    'is_bottom_note': is_bottom_note
                }
            elif tie_type == 'continue':
                # Continue an existing tie
                if pitch_key in active_ties:
                    # End the current tie segment and start a new one
                    # This tie spans from a previous measure to this measure
                    tie_info = active_ties[pitch_key]
                    # Calculate xLimitStart to include the previous measure (where tie started)
                    # The start_x might be in a previous measure, so we need to find the measure start
                    # For now, use the current measure start, but we'll adjust this when drawing
                    tie_curves.append({
                        'start': (tie_info['start_x'], tie_info['start_y']),
                        'end': (xPosition, pitch_y),
                        'curveDown': tie_info['is_bottom_note'],
                        'xLimitStart': min(tie_info['start_x'], xCumulated),  # Include previous measure if needed
                        'xLimitEnd': xCumulated + availableWidth
                    })
                    # Start new segment
                    active_ties[pitch_key] = {
                        'start_x': xPosition,
                        'start_y': pitch_y,
                        'is_bottom_note': is_bottom_note
                    }
                else:
                    # Tie continues from previous measure - start tracking from measure beginning
                    if isFirstMeasureInGroup:
                        # Draw tie from measure start to this note
                        tie_curves.append({
                            'start': (xCumulated, pitch_y),
                            'end': (xPosition, pitch_y),
                            'curveDown': is_bottom_note,
                            'xLimitStart': xCumulated,
                            'xLimitEnd': xCumulated + availableWidth
                        })
                    # Start tracking for continuation
                    active_ties[pitch_key] = {
                        'start_x': xPosition,
                        'start_y': pitch_y,
                        'is_bottom_note': is_bottom_note
                    }
            elif tie_type == 'stop' or tie_type == 'end':
                # End an existing tie
                if pitch_key in active_ties:
                    tie_info = active_ties[pitch_key]
                    tie_curves.append({
                        'start': (tie_info['start_x'], tie_info['start_y']),
                        'end': (xPosition, pitch_y),
                        'curveDown': tie_info['is_bottom_note'],
                        'xLimitStart': xCumulated,
                        'xLimitEnd': xCumulated + availableWidth
                    })
                    del active_ties[pitch_key]
                else:
                    # Tie ends at measure start (tie from previous measure)
                    if isFirstMeasureInGroup:
                        # Draw short tie from measure start to this note
                        tie_curves.append({
                            'start': (xCumulated, pitch_y),
                            'end': (xPosition, pitch_y),
                            'curveDown': is_bottom_note,
                            'xLimitStart': xCumulated,
                            'xLimitEnd': xCumulated + availableWidth
                        })
        
        # Update current position to end of this note
        current_position = note_offset + note_duration
    
    # Check if there's a gap at the end of the measure
    if current_position < measureDuration:
        # There's a gap at the end - draw a rest
        gap_duration = measureDuration - current_position
        rest_type = duration_to_rest_type(gap_duration)
        
        # Calculate x position for the rest (centered in the gap)
        if current_position == 0:
            bbox = draw.textbbox((0, 0), REST_MAP[rest_type], font=load_music_font(FONT_SIZES['rest']))
            rest_width = bbox[2] - bbox[0]
            rest_x = xCumulated + (availableWidth - rest_width)/2
        else:
            rest_x = xCumulated + ACCIDENTAL_SPACE + (current_position / measureDuration) * (availableWidth - ACCIDENTAL_SPACE)
        
        draw_rest(draw, rest_x, staff_y, rest_type, is_treble)
    
    updatedKeySignature = currentKeySignature if currentKeySignature is not None else measureGroupInfo.keySignature
    updatedTimeSignature = currentTimeSignature if timeSignatureChanged else measureGroupInfo.timeSignature
    
    return MeasureGroupInfo(updatedKeySignature, updatedTimeSignature), tie_curves, active_ties, noteKeyFrames

def draw_key_signature(draw: ImageDraw.Draw, keySignatureSharps: int, xCumulated: int, yBottomLine: int, clefChar: str) -> int:
    font = load_music_font(FONT_SIZES['accidental'])

    symbolChar = '\u266F' if keySignatureSharps > 0 else '\u266D'
    cumulatedWidth = 0

    if keySignatureSharps > 0:
        offset = 0
        if clefChar == MUSICAL_SYMBOLS['treble_clef']:
            positions = [4, 2.5, 4.5, 3, 1.5, 3.5, 2] # fa do sol re la mi si
        else:
            positions = [3, 1.5, 3.5, 2, 0.5, 2.5, 1]
    else:
        offset = FLAT_OFFSET
        if clefChar == MUSICAL_SYMBOLS['treble_clef']:
            positions = [2, 3.5, 1.5, 3, 1, 2.5, 0.5]
        else:
            positions = [1, 2.5, 0.5, 2, 0, 1.5, -0.5]
    
    xCumulated += X_SIGN_OFFSET
    cumulatedWidth += X_SIGN_OFFSET
    bbox = draw.textbbox((0, 0), symbolChar, font=font)
    for i in range(abs(keySignatureSharps)):
        pos = positions[i]
        
        y = yBottomLine - pos*STAFF_LINE_SPACING - (bbox[3] - bbox[1])/2 + offset
        draw.text((xCumulated, y), symbolChar, fill=(0,0,0), font=font, anchor='lt')
        xCumulated += bbox[2] - bbox[0]
        cumulatedWidth += bbox[2] - bbox[0]
        xCumulated += KEY_SIGNATURE_ACCIDENTAL_SPACING
        cumulatedWidth += KEY_SIGNATURE_ACCIDENTAL_SPACING
    
    return cumulatedWidth

def create_score_frame(
    measureGroupInfo: MeasureGroupInfo,
    rightHandMeasures: list[stream.Measure],
    leftHandMeasures: list[stream.Measure],
    measureGroup: dict,
    endMeasureIndex:int,
    endMeasureTime: float) -> tuple[Image.Image, MeasureGroupInfo]:

    """
    Create a score frame image.
    """
    startMeasureIndex = measureGroup["startMeasureIndex"]
    startMeasureTime = measureGroup["absolutePlayTime"]
    measureGroupKeyFrames = measureGroup["keyframes"]
    relativePlayTimes = measureGroup["relativePlayTimes"]
    img = Image.new('RGBA', (VIDEO_WIDTH, VIDEO_HEIGHT//2), color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    # treble clef width
    drawbbox = draw.textbbox((0,0), MUSICAL_SYMBOLS['treble_clef'], font=load_music_font(FONT_SIZES['clef']))
    trebleClefWidth = drawbbox[2] - drawbbox[0]
    # bass clef width
    drawbbox = draw.textbbox((0,0), MUSICAL_SYMBOLS['bass_clef'], font=load_music_font(FONT_SIZES['clef']))
    bassClefWidth = drawbbox[2] - drawbbox[0]
    clefWidthDict = {"treble": trebleClefWidth, "bass": bassClefWidth}

    # Draw two sets of five lines for the staff
    # height 540
    middlePoint = VIDEO_HEIGHT//4
    HANDS_SPACING_OFFSET = STAFF_LINE_SPACING*3

    xCumulated = X_MARGIN

    # Right and Left hand lines
    for i in range(5):
        lineYR = middlePoint - HANDS_SPACING_OFFSET - i*STAFF_LINE_SPACING - LINE_THICKNESS
        lineYL = middlePoint + HANDS_SPACING_OFFSET + i*STAFF_LINE_SPACING
        draw.line([(X_MARGIN, lineYR), (VIDEO_WIDTH - X_MARGIN, lineYR)], fill=(0, 0, 0), width=LINE_THICKNESS)
        draw.line([(X_MARGIN, lineYL), (VIDEO_WIDTH - X_MARGIN, lineYL)], fill=(0, 0, 0), width=LINE_THICKNESS)

    
    # Draw leftmost line
    lineYR0 = middlePoint - HANDS_SPACING_OFFSET - 4*STAFF_LINE_SPACING - LINE_THICKNESS
    lineYL4 = middlePoint + HANDS_SPACING_OFFSET + 4*STAFF_LINE_SPACING
    draw.line([(X_MARGIN, lineYR0), (X_MARGIN, lineYL4)], fill=(0,0,0), width=LINE_THICKNESS)

    # Draw brace
    draw_brace(draw, X_MARGIN - 2, lineYR0, load_music_font(lineYL4 - lineYR0))

    # Draw rightmost line
    draw.line([(VIDEO_WIDTH - X_MARGIN, lineYR0), (VIDEO_WIDTH - X_MARGIN, lineYL4)], fill=(0,0,0), width=LINE_THICKNESS)
    
    # Draw clef
    # Determine right hand clef.
    rightHandClef = rightHandMeasures[startMeasureIndex].clef if rightHandMeasures[startMeasureIndex].clef else rightHandMeasures[startMeasureIndex].getContextByClass(clef.Clef)
    if rightHandClef.name == 'treble':
        rightHandClefChar = MUSICAL_SYMBOLS['treble_clef']
    else:
        rightHandClefChar = MUSICAL_SYMBOLS['bass_clef']
    bbox = draw.textbbox((0,0), rightHandClefChar, font=load_music_font(FONT_SIZES['clef']))
    xCumulated += X_CLEF_OFFSET
    trebleY = middlePoint - HANDS_SPACING_OFFSET - STAFF_LINE_SPACING*2 - (bbox[3] - bbox[1])/2
    draw.text((xCumulated, trebleY), rightHandClefChar, fill=(0, 0, 0), font=load_music_font(FONT_SIZES['clef']), anchor='lt')
    # Determine left hand clef.
    leftHandClef = leftHandMeasures[startMeasureIndex].clef if leftHandMeasures[startMeasureIndex].clef else leftHandMeasures[startMeasureIndex].getContextByClass(clef.Clef)
    if leftHandClef.name == 'treble':
        leftHandClefChar = MUSICAL_SYMBOLS['treble_clef']
    else:
        leftHandClefChar = MUSICAL_SYMBOLS['bass_clef']
    bbox = draw.textbbox((0,0), leftHandClefChar, font=load_music_font(FONT_SIZES['clef']))
    bassY = middlePoint + HANDS_SPACING_OFFSET + STAFF_LINE_SPACING*2 - (bbox[3] - bbox[1])/2
    draw.text((xCumulated, bassY), leftHandClefChar, fill=(0, 0, 0), font=load_music_font(FONT_SIZES['clef']), anchor='lt')
    xCumulated += bbox[2] - bbox[0]

    # Draw key signature
    if rightHandMeasures[startMeasureIndex].keySignature != None:
        keySignature = rightHandMeasures[startMeasureIndex].keySignature
    else:
        keySignature = measureGroupInfo.keySignature
    if keySignature.sharps != 0:
        draw_key_signature(draw, keySignature.sharps, xCumulated, middlePoint - HANDS_SPACING_OFFSET - LINE_THICKNESS, rightHandClefChar)
        xCumulated += draw_key_signature(draw, keySignature.sharps, xCumulated, middlePoint + HANDS_SPACING_OFFSET + STAFF_LINE_SPACING*4, leftHandClefChar)

    # Draw time signature
    if rightHandMeasures[startMeasureIndex].timeSignature != None:
        timeSignature = rightHandMeasures[startMeasureIndex].timeSignature
    else:
        timeSignature = measureGroupInfo.timeSignature
    xCumulated += X_TIME_SIG_OFFSET
    timeSignatureFont = load_music_font(FONT_SIZES['time_sig'])
    yDenomRight = middlePoint - HANDS_SPACING_OFFSET - STAFF_LINE_SPACING*4 - LINE_THICKNESS
    yNomRight = middlePoint - HANDS_SPACING_OFFSET - STAFF_LINE_SPACING*2 - LINE_THICKNESS
    yDenomLeft = middlePoint + HANDS_SPACING_OFFSET + LINE_THICKNESS
    yNomLeft = middlePoint + HANDS_SPACING_OFFSET + STAFF_LINE_SPACING*2 + LINE_THICKNESS
    bboxDenom = draw.textbbox((0,0), MUSICAL_SYMBOLS[str(timeSignature.numerator)], font=timeSignatureFont)
    bboxNom = draw.textbbox((0,0), MUSICAL_SYMBOLS[str(timeSignature.denominator)], font=timeSignatureFont)
    if bboxDenom[2] - bboxDenom[0] > bboxNom[2] - bboxNom[0]:
        draw.text((xCumulated, yDenomRight), MUSICAL_SYMBOLS[str(timeSignature.numerator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt') # right hand
        draw.text((xCumulated, yDenomLeft), MUSICAL_SYMBOLS[str(timeSignature.numerator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt') # left hand
        draw.text((xCumulated + (bboxDenom[2] - bboxDenom[0])//2 - (bboxNom[2] - bboxNom[0])//2, yNomRight), MUSICAL_SYMBOLS[str(timeSignature.denominator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt')
        draw.text((xCumulated + (bboxDenom[2] - bboxDenom[0])//2 - (bboxNom[2] - bboxNom[0])//2, yNomLeft), MUSICAL_SYMBOLS[str(timeSignature.denominator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt')
        xCumulated += bboxDenom[2] - bboxDenom[0]
    elif bboxNom[2] - bboxNom[0] > bboxDenom[2] - bboxDenom[0]:
        draw.text((xCumulated + (bboxNom[2] - bboxNom[0])//2 - (bboxDenom[2] - bboxDenom[0])//2, yDenomRight), MUSICAL_SYMBOLS[str(timeSignature.numerator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt')
        draw.text((xCumulated + (bboxNom[2] - bboxNom[0])//2 - (bboxDenom[2] - bboxDenom[0])//2, yDenomLeft), MUSICAL_SYMBOLS[str(timeSignature.numerator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt')
        draw.text((xCumulated, yNomRight), MUSICAL_SYMBOLS[str(timeSignature.denominator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt')
        draw.text((xCumulated, yNomLeft), MUSICAL_SYMBOLS[str(timeSignature.denominator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt')
        xCumulated += bboxNom[2] - bboxNom[0]
    else:
        draw.text((xCumulated, yDenomRight), MUSICAL_SYMBOLS[str(timeSignature.numerator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt')
        draw.text((xCumulated, yDenomLeft), MUSICAL_SYMBOLS[str(timeSignature.numerator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt')
        draw.text((xCumulated, yNomRight), MUSICAL_SYMBOLS[str(timeSignature.denominator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt')
        draw.text((xCumulated, yNomLeft), MUSICAL_SYMBOLS[str(timeSignature.denominator)], fill=(0,0,0), font=timeSignatureFont, anchor='lt')
        xCumulated += bboxNom[2] - bboxNom[0]

    # Now time to draw the notes.
    currentMeasureGroupInfo = MeasureGroupInfo(keySignature, timeSignature)
    availableWidth = VIDEO_WIDTH - X_MARGIN - xCumulated - LINE_THICKNESS
    widthPerMeasure = (availableWidth - (endMeasureIndex - startMeasureIndex)*LINE_THICKNESS) / (endMeasureIndex - startMeasureIndex + 1)

    # Collect all tie curves from all measures
    all_tie_curves = []
    
    # Track active ties separately for right and left hand across measures in the same group
    rightHandActiveTies = {}
    leftHandActiveTies = {}
    
    # Track the start X of the first measure for proper tie clipping
    firstMeasureStartX = xCumulated
    currentRightHandClef = rightHandMeasures[startMeasureIndex].clef if rightHandMeasures[startMeasureIndex].clef else rightHandMeasures[startMeasureIndex].getContextByClass(clef.Clef)
    currentLeftHandClef = leftHandMeasures[startMeasureIndex].clef if leftHandMeasures[startMeasureIndex].clef else leftHandMeasures[startMeasureIndex].getContextByClass(clef.Clef)
    for measureIndex in range(startMeasureIndex, endMeasureIndex+1):
        isFirstMeasure = (measureIndex == startMeasureIndex)
        measureTimeSpan = endMeasureTime - (startMeasureTime + relativePlayTimes[endMeasureIndex - startMeasureIndex])
        if measureIndex < endMeasureIndex:
            measureTimeSpan = relativePlayTimes[measureIndex - startMeasureIndex + 1] - relativePlayTimes[measureIndex - startMeasureIndex]
        # Before drawing the measure, check if the clef has changed in either hand.
        # For every clef change, mark x position where the clef change occurs, and the width of the clef. 
        # This information should pass on to draw_measure function and notes area should avoid the space occupied by clef. 
        rightHandClefChanges = []
        leftHandClefChanges = []
        rightHandMeasure = rightHandMeasures[measureIndex]
        leftHandMeasure = leftHandMeasures[measureIndex]
        for note in rightHandMeasure.notes:
            noteClef = note.getContextByClass(clef.Clef)
            if noteClef != currentRightHandClef:
                rightHandClefChanges.append({
                    'startQuarter': note.offset,
                    'width': clefWidthDict[noteClef.name],
                    "newClef": noteClef
                })
                leftHandClefChanges.append({
                    'startQuarter': note.offset,
                    'width': clefWidthDict[noteClef.name],
                    "newClef": None
                })
                currentRightHandClef = noteClef
        for note in leftHandMeasure.notes:
            noteClef = note.getContextByClass(clef.Clef)
            if noteClef != currentLeftHandClef:
                leftHandClefChanges.append({
                    'startQuarter': note.offset,
                    'width': clefWidthDict[noteClef.name],
                    "newClef": noteClef
                })
                rightHandClefChanges.append({
                    'startQuarter': note.offset,
                    'width': clefWidthDict[noteClef.name],
                    "newClef": None
                })
                currentLeftHandClef = noteClef
        rightHandClefChanges.sort(key=lambda x: x['startQuarter'])
        leftHandClefChanges.sort(key=lambda x: x['startQuarter'])
        rightMeasureGroupInfo, rightTieCurves, rightHandActiveTies, rightMeasureKeyFrames = draw_measure(currentMeasureGroupInfo, draw, rightHandMeasures[measureIndex], xCumulated, middlePoint - HANDS_SPACING_OFFSET - LINE_THICKNESS, widthPerMeasure, isFirstMeasure, rightHandActiveTies, rightHandClefChanges)
        leftMeasureGroupInfo, leftTieCurves, leftHandActiveTies, leftMeasureKeyFrames = draw_measure(currentMeasureGroupInfo, draw, leftHandMeasures[measureIndex], xCumulated, middlePoint + HANDS_SPACING_OFFSET + STAFF_LINE_SPACING*4, widthPerMeasure, isFirstMeasure, leftHandActiveTies, leftHandClefChanges)
        # Before merging key frames remove duplicates between right and left key frames.
        i = 0; j = 0;
        while i < len(rightMeasureKeyFrames) and j < len(leftMeasureKeyFrames):
            if rightMeasureKeyFrames[i]["x_absolute"] == leftMeasureKeyFrames[j]["x_absolute"]:
                rightMeasureKeyFrames[i]["y_middles"].extend(leftMeasureKeyFrames[j]["y_middles"])
                leftMeasureKeyFrames.pop(j)
            elif rightMeasureKeyFrames[i]["x_absolute"] < leftMeasureKeyFrames[j]["x_absolute"]:
                i += 1
            else:
                j += 1
        # Merge key frames
        while len(rightMeasureKeyFrames) > 0 or len(leftMeasureKeyFrames) > 0:
            rightKeyFrame = {"x_absolute": 10000} # large enough value
            if len(rightMeasureKeyFrames) > 0:
                rightKeyFrame = rightMeasureKeyFrames[0]
            leftKeyFrame = {"x_absolute": 10000} # large enough value
            if len(leftMeasureKeyFrames) > 0:
                leftKeyFrame = leftMeasureKeyFrames[0]
            if rightKeyFrame["x_absolute"] < leftKeyFrame["x_absolute"]:
                rightMeasureKeyFrames.pop(0)
                measureGroupKeyFrames.append({
                    "t": measureTimeSpan * rightKeyFrame["t_ratio"] + startMeasureTime + relativePlayTimes[measureIndex - startMeasureIndex],
                    "x": rightKeyFrame["x_absolute"],
                    "y_middles": rightKeyFrame["y_middles"],
                    "score_quarter": rightKeyFrame.get("score_quarter"),
                })
            else:
                leftMeasureKeyFrames.pop(0)
                measureGroupKeyFrames.append({
                    "t": measureTimeSpan * leftKeyFrame["t_ratio"] + startMeasureTime + relativePlayTimes[measureIndex - startMeasureIndex],
                    "x": leftKeyFrame["x_absolute"],
                    "y_middles": leftKeyFrame["y_middles"],
                    "score_quarter": leftKeyFrame.get("score_quarter"),
                })
        
        # Collect tie curves
        all_tie_curves.extend(rightTieCurves)
        all_tie_curves.extend(leftTieCurves)
        
        xCumulated += widthPerMeasure
        # draw bar lines
        if measureIndex != endMeasureIndex:
            draw.line([(xCumulated, lineYR0), (xCumulated, lineYL4)], fill=(0,0,0), width=LINE_THICKNESS)
        xCumulated += LINE_THICKNESS
        currentMeasureGroupInfo = rightMeasureGroupInfo

    # Add last point to the key frames. 
    endScoreQuarter = float(rightHandMeasures[endMeasureIndex].offset + rightHandMeasures[endMeasureIndex].duration.quarterLength)
    measureGroupKeyFrames.append({
        "t": endMeasureTime,
        "x": VIDEO_WIDTH - X_MARGIN,
        "y_middles": [],
        "score_quarter": endScoreQuarter,
    })
    
    # Calculate the end X of the last measure for proper tie clipping
    lastMeasureEndX = xCumulated - LINE_THICKNESS  # Subtract the last barline thickness
    
    # Add tie curves for any active ties that haven't been closed (continue beyond the measure group)
    # The end point should be slightly before the measure end (with margin)
    tieEndX = lastMeasureEndX - TIE_END_MARGIN
    
    # Handle right hand active ties
    for pitch_key, tie_info in rightHandActiveTies.items():
        all_tie_curves.append({
            'start': (tie_info['start_x'], tie_info['start_y']),
            'end': (tieEndX, tie_info['start_y']),  # Use same Y as start (same pitch)
            'curveDown': tie_info['is_bottom_note'],
            'xLimitStart': firstMeasureStartX,
            'xLimitEnd': lastMeasureEndX
        })
    
    # Handle left hand active ties
    for pitch_key, tie_info in leftHandActiveTies.items():
        all_tie_curves.append({
            'start': (tie_info['start_x'], tie_info['start_y']),
            'end': (tieEndX, tie_info['start_y']),  # Use same Y as start (same pitch)
            'curveDown': tie_info['is_bottom_note'],
            'xLimitStart': firstMeasureStartX,
            'xLimitEnd': lastMeasureEndX
        })

    # Draw all tie curves at the end
    for tie_curve in all_tie_curves:
        # For ties that span measures, use the full measure group boundaries
        # This ensures ties spanning multiple measures are drawn correctly
        xLimitStart = firstMeasureStartX
        xLimitEnd = lastMeasureEndX
        draw_tie(draw, tie_curve['start'], tie_curve['end'], 
                xLimitStart, xLimitEnd, 
                tie_curve['curveDown'])

    return img, currentMeasureGroupInfo

def group_measures_for_frame(
    rightHandMeasures: list[stream.Measure],
    leftHandMeasures: list[stream.Measure],
    score: stream.Score,
) -> list[dict]:

    # output is a list of dictionaries, each dictionary contains the start measure index, the absolute play time entering the start measure, and keyframes.
    measureNumberPairs = []
    
    # Calculate tempo information for converting quarter notes to seconds
    flat = score.flat
    tempos = list(flat.getElementsByClass(tempo.MetronomeMark))
    tempos.sort(key=lambda m: m.offset)
    
    def quarter_notes_to_seconds(quarter_offset: float) -> float:
        """Convert quarter note offset to absolute play time in seconds."""
        if not tempos:
            # Default to 120 BPM if no tempo found
            return quarter_offset * 60.0 / 120.0
        
        total_seconds = 0.0
        first_tempo_offset = tempos[0].offset
        
        # Handle case where offset is before first tempo mark
        if quarter_offset < first_tempo_offset:
            # Use first tempo for the entire duration
            bpm = tempos[0].getQuarterBPM()
            return quarter_offset * 60.0 / bpm
        
        # Process tempo segments up to the target offset
        for i, mm in enumerate(tempos):
            start = mm.offset
            end = tempos[i+1].offset if i+1 < len(tempos) else quarter_offset
            bpm = mm.getQuarterBPM()
            
            # If we've reached or passed the target offset, calculate the remaining time
            if quarter_offset <= end:
                segment_quarters = quarter_offset - start
                if segment_quarters > 0:
                    total_seconds += segment_quarters * 60.0 / bpm
                break
            else:
                # Process the entire segment
                segment_quarters = end - start
                if segment_quarters > 0:
                    total_seconds += segment_quarters * 60.0 / bpm
        
        return total_seconds

    img = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT//2), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # How to determine the number of measures per frame
    # The width between two notes should be at least 
    trebleSignWidthInPixels = draw.textbbox((0,0), '\U0001D11E', font=load_music_font(FONT_SIZES['clef']))[2] - draw.textbbox((0,0), '\U0001D11E', font=load_music_font(FONT_SIZES['clef']))[0]
    bassSignWidthInPixels = draw.textbbox((0,0), '\U0001D122', font=load_music_font(FONT_SIZES['clef']))[2] - draw.textbbox((0,0), '\U0001D122', font=load_music_font(FONT_SIZES['clef']))[0]
    clefWidth = max(trebleSignWidthInPixels, bassSignWidthInPixels)
    keySignatureWidth = calculate_key_signature_width(draw, rightHandMeasures[0].keySignature)
    timeSignatureFont = ImageFont.truetype("AppleMyungjo.ttf", FONT_SIZES['time_sig'])
    timeSignatureWidth = draw.textbbox((0,0), "4", font=timeSignatureFont)[2] - draw.textbbox((0,0), "4", font=timeSignatureFont)[0]
    availableWidthBase = VIDEO_WIDTH - X_MARGIN*2 - X_CLEF_OFFSET - clefWidth - X_SIGN_OFFSET - timeSignatureWidth
    
    currentMeasureIndex = 0
    keySignature = None
    currentNumerator = 0
    currentDenominator = 0
    while currentMeasureIndex < len(rightHandMeasures):
        # Start new group. 
        availableWidth = availableWidthBase
        
        # Update the available width with the key signature width. 
        if rightHandMeasures[currentMeasureIndex].keySignature != None:
            keySignature = rightHandMeasures[currentMeasureIndex].keySignature
        keySignatureWidth = calculate_key_signature_width(draw, keySignature)
        if keySignatureWidth > 0:
            availableWidth -= keySignatureWidth
            availableWidth -= X_SIGN_OFFSET
        
        # Check if measures can fit into default number of measures per group.
        numberOfMeasures = DEFAULT_MEASURE_COUNT_PER_GROUP
        finished = False
        
        while numberOfMeasures > 1 and not finished:
            startingRightHandClef = rightHandMeasures[currentMeasureIndex].clef if rightHandMeasures[currentMeasureIndex].clef else rightHandMeasures[currentMeasureIndex].getContextByClass(clef.Clef)
            startingLeftHandClef = leftHandMeasures[currentMeasureIndex].clef if leftHandMeasures[currentMeasureIndex].clef else leftHandMeasures[currentMeasureIndex].getContextByClass(clef.Clef)
            targetMeasureWidth = availableWidth / numberOfMeasures - LINE_THICKNESS*(numberOfMeasures - 1)
            numerator = currentNumerator
            denominator = currentDenominator
            currentKeySignature = keySignature
            numberOfMeasuresAdjusted = False
            for i in range(numberOfMeasures):
                if currentMeasureIndex + i >= len(rightHandMeasures):
                    numberOfMeasures = i
                    finished = True
                    break
                if rightHandMeasures[currentMeasureIndex + i].timeSignature != None:
                    currentNumerator = rightHandMeasures[currentMeasureIndex + i].timeSignature.numerator
                    currentDenominator = rightHandMeasures[currentMeasureIndex + i].timeSignature.denominator
                    numerator = currentNumerator
                    denominator = currentDenominator
                
                # chek for key signature change.
                extraKeySignatureWidth = 0
                if rightHandMeasures[currentMeasureIndex + i].keySignature != None and currentKeySignature != rightHandMeasures[currentMeasureIndex + i].keySignature:
                    currentKeySignature = rightHandMeasures[currentMeasureIndex + i].keySignature
                    extraKeySignatureWidth = calculate_key_signature_width(draw, currentKeySignature)

                # Find the smallest length note in the measure. 
                # Also, account for the width for the clef change. 
                biggestDenominator = denominator
                clefChanges = 0
                for note in rightHandMeasures[currentMeasureIndex + i].notes:
                    noteDenominator = Fraction(note.beat*0.25).denominator
                    if noteDenominator > biggestDenominator:
                        biggestDenominator = noteDenominator
                    if note.getContextByClass(clef.Clef) != startingRightHandClef:
                        clefChanges += 1
                        startingRightHandClef = note.getContextByClass(clef.Clef)
                for note in leftHandMeasures[currentMeasureIndex + i].notes:
                    noteDenominator = Fraction(note.beat*0.25).denominator
                    if noteDenominator > biggestDenominator:
                        biggestDenominator = noteDenominator
                    if note.getContextByClass(clef.Clef) != startingLeftHandClef:
                        clefChanges += 1
                        startingLeftHandClef = note.getContextByClass(clef.Clef)
                # Calculate the smallest distance counts in the measure. 
                while biggestDenominator > denominator:
                    numerator *= 2
                    denominator *= 2
                numberOfSpacesRequired = numerator + 1;
                musicalNotationWidth = extraKeySignatureWidth + clefChanges*(clefWidth + X_CLEF_OFFSET);
                if ((targetMeasureWidth - musicalNotationWidth - ACCIDENTAL_SPACE) / numberOfSpacesRequired < MINIMUM_DISTANCE_BETWEEN_NOTES):
                    # This measure cannot fit, reduce the number of measures per group and start over. 
                    numberOfMeasures -= 1
                    numberOfMeasuresAdjusted = True
                    break

            if not numberOfMeasuresAdjusted:
                break

        
        # Calculate absolute play time for the start measure
        startMeasure = rightHandMeasures[currentMeasureIndex]
        measureOffsetInQuarters = startMeasure.offset
        absolutePlayTime = quarter_notes_to_seconds(measureOffsetInQuarters)

        relativePlayTimes = []
        for i in range(numberOfMeasures):
            relativePlayTimes.append(quarter_notes_to_seconds(rightHandMeasures[currentMeasureIndex + i].offset) - absolutePlayTime)
        
        measureNumberPairs.append({
            "startMeasureIndex": currentMeasureIndex,
            "absolutePlayTime": absolutePlayTime,
            "startScoreQuarter": float(measureOffsetInQuarters),
            "relativePlayTimes": relativePlayTimes,
            "keyframes": [],
        })
        currentMeasureIndex += numberOfMeasures
        

    return measureNumberPairs

def draw_playhead(draw: ImageDraw.Draw, x: int, y: int, opacity: int) -> None:
    playHeadHeight = 2*HANDS_SPACING_OFFSET + 8*STAFF_LINE_SPACING
    draw.rectangle([(x, y), (x + PLAYHEAD_WIDTH, y + playHeadHeight)], fill=(PLAYHEAD_COLOR[0], PLAYHEAD_COLOR[1], PLAYHEAD_COLOR[2], opacity))

def generate_movie(
    score: stream.Score,
    outputPath: str,
    *,
    performance_midi_path: str | None = None,
    score_path: str | None = None,
    score_midi_path: str | None = None,
) -> None:
    """
    Generate a movie from a music21 Score object.
    
    Args:
        score: music21 Score object to convert to video
        performance_midi_path: Optional performance MIDI to align timing to
        score_path: Path to the score file (MIDI or MusicXML) used for partitura alignment
        score_midi_path: Optional absolute path to a score MIDI file for score–performance alignment
    """
    metadata = score.metadata
    righthandPart = score.parts[0]
    lefthandPart = score.parts[1]
    rightHandMeasures = righthandPart.getElementsByClass(stream.Measure)
    leftHandMeasures = lefthandPart.getElementsByClass(stream.Measure)

    # Determine the total seconds of the score (symbolic timing).
    flat = score.flat
    tempos = list(flat.getElementsByClass(tempo.MetronomeMark))
    tempos.sort(key=lambda m: m.offset)
    total_quarters = score.duration.quarterLength
    total_seconds = 0.0
    for i, mm in enumerate(tempos):
        start = mm.offset
        end = tempos[i+1].offset if i+1 < len(tempos) else total_quarters
        bpm = mm.getQuarterBPM()

        total_seconds += (end - start) * 60.0 / bpm

    # Determine the set of measures to display per frame before rendering. 
    # There will be top frame and bottom frame. 
    measureGroups = group_measures_for_frame(rightHandMeasures, leftHandMeasures, score)
    # measureGroups = [{"startMeasureIndex": int, "absolutePlayTime": float, "keyframes": list}, ...]

    currentGroupIndex = [0]
    currentGroupSide = ['top']
    frameImages = []

    # Generate all frame images before rendering. 
    measureGroupInfo = None
    for i in range(len(measureGroups)):
        if i == 0:
            print(":::")
        endMeasureIndex = len(rightHandMeasures)-1 if i == len(measureGroups) - 1 else measureGroups[i+1]["startMeasureIndex"] - 1
        endMeasureTime = measureGroups[i+1]["absolutePlayTime"] if i+1 < len(measureGroups) else total_seconds
        frameImage, measureGroupInfo = create_score_frame(measureGroupInfo, rightHandMeasures, leftHandMeasures, measureGroups[i], endMeasureIndex, endMeasureTime)
        frameImages.append(frameImage)

    # Optional: replace symbolic timing with performance timing via partitura-based alignment.
    if performance_midi_path is not None:
        if score_path is None:
            raise ValueError("score_path must be provided when performance_midi_path is set")

        try:
            from partitura_alignment import compute_score_to_performance_time_warp

            warp, diag = compute_score_to_performance_time_warp(
                score_path=score_path,
                performance_midi_path=performance_midi_path,
                score_midi_path=score_midi_path,
            )
            print(
                f"[alignment] score_groups={diag.n_score_groups} perf_groups={diag.n_perf_groups} "
                f"anchors={diag.n_anchor_points} perf_shift={diag.perf_time_shift:.3f}s"
            )

            # Update measure start times (for group switching/fades) and keyframe times.
            last_t = 0.0
            for g in measureGroups:
                if "startScoreQuarter" in g:
                    g["absolutePlayTime"] = float(warp(float(g["startScoreQuarter"])))
                for kf in g.get("keyframes", []):
                    if "score_quarter" in kf and kf["score_quarter"] is not None:
                        kf["t"] = float(warp(float(kf["score_quarter"])))
                if g.get("keyframes"):
                    last_t = max(last_t, float(g["keyframes"][-1].get("t", 0.0)))

            total_seconds = max(total_seconds, last_t)
        except Exception as e:
            print(f"[alignment] WARNING: alignment failed ({type(e).__name__}: {e}); using score timing instead.")

    def make_frame(t):
        # Determine current group.
        if currentGroupIndex[0]+1 < len(measureGroups) and t >= measureGroups[currentGroupIndex[0]+1]["absolutePlayTime"]:
            currentGroupIndex[0] += 1
            currentGroupSide[0] = 'bottom' if currentGroupSide[0] == 'top' else 'top'

        # Determine playhead position with interpolation.
        targetKeyFrames = measureGroups[currentGroupIndex[0]]["keyframes"]
        playHeadX0 = targetKeyFrames[0]["x"]
        playHeadT0 = targetKeyFrames[0]["t"]
        playHeadX1 = VIDEO_WIDTH - X_MARGIN
        playHeadT1 = measureGroups[currentGroupIndex[0]+1]["absolutePlayTime"] if currentGroupIndex[0]+1 < len(measureGroups) else total_seconds
        keyFrame0 = targetKeyFrames[-1]
        for i in range(len(targetKeyFrames) - 1):
            if targetKeyFrames[i]["t"] <= t < targetKeyFrames[i+1]["t"]:
                playHeadX0 = targetKeyFrames[i]["x"]
                playHeadT0 = targetKeyFrames[i]["t"]
                playHeadX1 = targetKeyFrames[i+1]["x"]
                playHeadT1 = targetKeyFrames[i+1]["t"]
                keyFrame0 = targetKeyFrames[i]
                break
        
        # Figure out the notes that note sparkle animation applies at time t. 
        sparkleAnimationList = []
        if t - keyFrame0["t"] < SPARKLE_ANIMATION_DURATION:
            yMiddles = keyFrame0["y_middles"]
            x = keyFrame0["x"]
            for yMiddle in yMiddles:
                sparkleAnimationList.append({"x": x, "y": yMiddle, "t": t - keyFrame0["t"]})
        
        playHeadX = playHeadX0 + (playHeadX1 - playHeadX0) * (t - playHeadT0) / (playHeadT1 - playHeadT0)
        middlePoint = VIDEO_HEIGHT//4
        playHeadY0 = middlePoint - HANDS_SPACING_OFFSET - 4*STAFF_LINE_SPACING - LINE_THICKNESS
        
        # The other side should display either fading out previous group or fading in next group.
        otherGroupIndex = 0
        otherGroupOpacity = 0.0
        if currentGroupIndex[0] == 0:
            otherGroupIndex = 1
            otherGroupOpacity = 1.0
        elif currentGroupIndex[0] == len(measureGroups) - 1:
            if t - measureGroups[currentGroupIndex[0]]["absolutePlayTime"] <= GROUP_FADE_DURATION:
                otherGroupOpacity = 1.0 - (t - measureGroups[currentGroupIndex[0]]["absolutePlayTime"]) / GROUP_FADE_DURATION
                otherGroupIndex = currentGroupIndex[0] - 1
            else:
                otherGroupIndex = -1 # Indication of not to draw. 
        elif t - measureGroups[currentGroupIndex[0]]["absolutePlayTime"] <= GROUP_FADE_DURATION:
            otherGroupOpacity = 1.0 - (t - measureGroups[currentGroupIndex[0]]["absolutePlayTime"]) / GROUP_FADE_DURATION
            otherGroupIndex = currentGroupIndex[0] - 1
        else:
            otherGroupIndex = currentGroupIndex[0] + 1
            otherGroupOpacity = (t - measureGroups[currentGroupIndex[0]]["absolutePlayTime"] - GROUP_FADE_DURATION) / GROUP_FADE_DURATION
            if otherGroupOpacity > 1.0:
                otherGroupOpacity = 1.0
        
        # Prepare image.
        combined = Image.new('RGBA', (VIDEO_WIDTH, VIDEO_HEIGHT), color=(255, 255, 255, 255))

        # copy current section from frameImages
        currentGroupImage = frameImages[currentGroupIndex[0]].convert("RGBA")
        draw = ImageDraw.Draw(currentGroupImage)
        draw_playhead(draw, playHeadX, playHeadY0, 255)
        draw.text((30, 30), f"{str(currentGroupIndex[0])}", fill=(0, 0, 0, 255), font=ImageFont.truetype("Helvetica", 50))
        # Draw sparkle animations.
        sparkleOverlay = Image.new("RGBA", currentGroupImage.size, color=(0, 0, 0, 0))
        sparkleDraw = ImageDraw.Draw(sparkleOverlay)
        for sparkleAnimation in sparkleAnimationList:
            draw_note_sparkle(sparkleDraw, sparkleAnimation["x"], sparkleAnimation["y"], sparkleAnimation["t"])
        # Alpha composting
        currentGroupImage = Image.alpha_composite(currentGroupImage, sparkleOverlay)

        if currentGroupSide[0] == 'top':
            combined.paste(currentGroupImage, (0, 0), currentGroupImage)
        else:
            combined.paste(currentGroupImage, (0, VIDEO_HEIGHT // 2), currentGroupImage)

        # draw the other section
        if otherGroupIndex != -1:
            otherGroupImage = frameImages[otherGroupIndex].convert('RGBA')
            draw = ImageDraw.Draw(otherGroupImage)
            draw.text((30, 30), f"{str(otherGroupIndex)}", fill=(0, 0, 0, 255), font=ImageFont.truetype("Helvetica", 50))
            otherGroupImage.putalpha(int(otherGroupOpacity*255))
            
            if currentGroupSide[0] == 'top':
                combined.paste(otherGroupImage, (0, VIDEO_HEIGHT // 2), otherGroupImage)
            else:
                combined.paste(otherGroupImage, (0, 0), otherGroupImage)

        # Convert RGBA to RGB for video output
        combined_rgb = combined.convert('RGB')
        frame_array = np.array(combined_rgb)

        # Determine playhead position

        return frame_array
        

    video = VideoClip(make_frame, duration=total_seconds)

    video.write_videofile(outputPath, fps=FPS, codec='libx264', audio=False)


