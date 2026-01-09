"""
Score–performance alignment utilities using `partitura`.

This module estimates a monotonic time-warp mapping from score time (in beats /
quarterLength) to performance time (seconds). It is designed for transferring
performance timing onto score-derived keyframes (notehead positions) used by
`score2movie.py`.

Notes:
- We intentionally keep this dependency-light: `partitura` for MIDI parsing and
  `numpy` for interpolation. If you later install `parangonar`, you can swap in
  its more sophisticated alignment.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class AlignmentDiagnostics:
    n_score_groups: int
    n_perf_groups: int
    n_anchor_points: int
    perf_time_shift: float


def _group_note_array(
    note_array,
    onset_field: str,
    pitch_field: str = "pitch",
    onset_tolerance: float = 0.0,
) -> List[dict]:
    """
    Group a note_array into onset "chords" within onset_tolerance.

    Returns list of dicts: {"onset": float, "pitches": [int, ...], "onsets": [float, ...]}
    """
    if len(note_array) == 0:
        return []

    # structured array: sort by onset
    order = np.argsort(note_array[onset_field])
    na = note_array[order]

    groups: List[dict] = []
    cur_onset = float(na[0][onset_field])
    cur_pitches: List[int] = [int(na[0][pitch_field])]
    cur_onsets: List[float] = [cur_onset]

    for row in na[1:]:
        onset = float(row[onset_field])
        pitch = int(row[pitch_field])
        if onset - cur_onset <= onset_tolerance:
            cur_pitches.append(pitch)
            cur_onsets.append(onset)
        else:
            groups.append(
                {"onset": float(np.median(cur_onsets)), "pitches": cur_pitches, "onsets": cur_onsets}
            )
            cur_onset = onset
            cur_pitches = [pitch]
            cur_onsets = [onset]

    groups.append({"onset": float(np.median(cur_onsets)), "pitches": cur_pitches, "onsets": cur_onsets})
    return groups


def _multiset_similarity(a: Sequence[int], b: Sequence[int]) -> float:
    """Return similarity in [0,1] based on multiset pitch intersection."""
    if not a or not b:
        return 0.0
    ca = Counter(a)
    cb = Counter(b)
    inter = sum((ca & cb).values())
    denom = max(len(a), len(b))
    return float(inter) / float(denom) if denom else 0.0


def _align_groups_dp(
    score_groups: Sequence[dict],
    perf_groups: Sequence[dict],
    *,
    gap_penalty: float = 0.9,
    min_similarity_for_anchor: float = 0.25,
) -> List[Tuple[int, int, float]]:
    """
    Global alignment (Needleman–Wunsch style) over onset-groups.

    Returns list of matched group indices: (i_score, j_perf, similarity).
    """
    n = len(score_groups)
    m = len(perf_groups)
    if n == 0 or m == 0:
        return []

    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up, 2 left
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + gap_penalty
        bt[i, 0] = 1
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + gap_penalty
        bt[0, j] = 2

    sim_cache = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            sim_cache[i, j] = _multiset_similarity(score_groups[i]["pitches"], perf_groups[j]["pitches"])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = sim_cache[i - 1, j - 1]
            match_cost = 1.0 - sim
            diag = dp[i - 1, j - 1] + match_cost
            up = dp[i - 1, j] + gap_penalty
            left = dp[i, j - 1] + gap_penalty
            best = min(diag, up, left)
            dp[i, j] = best
            bt[i, j] = 0 if best == diag else (1 if best == up else 2)

    # backtrace
    i, j = n, m
    matches_rev: List[Tuple[int, int, float]] = []
    while i > 0 or j > 0:
        step = bt[i, j]
        if step == 0:
            si = i - 1
            pj = j - 1
            sim = float(sim_cache[si, pj])
            if sim >= min_similarity_for_anchor:
                matches_rev.append((si, pj, sim))
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1

    matches_rev.reverse()
    return matches_rev


def compute_score_to_performance_time_warp(
    score_path: str,
    performance_midi_path: str,
    score_midi_path: str | None = None,
    *,
    performance_grouping_ms: float = 35.0,
    score_grouping_beats: float = 1e-6,
    gap_penalty: float = 0.9,
    min_similarity_for_anchor: float = 0.25,
    start_at_zero: bool = True,
) -> Tuple[Callable[[float], float], AlignmentDiagnostics]:
    """
    Estimate a monotonic mapping f(score_beats) -> performance_seconds.

    score_beats is expected to be in quarter notes / beats (compatible with music21 quarterLength).
    """
    import partitura as pt
    from partitura.utils import music

    # Load score from either MusicXML/MXL or score-MIDI.
    if score_midi_path is not None:
        score = pt.load_score_midi(score_midi_path)
    else:
        score = pt.load_score(score_path)
    perf = pt.load_performance_midi(performance_midi_path)

    # Score can be a Part, PartGroup, or list-like. Normalize to a note array.
    if hasattr(score, "parts"):
        # partitura.load_score_midi returns a partitura.score.Score which exposes .parts
        score_note_array = music.note_array_from_part_list(list(score.parts))
    elif isinstance(score, (list, tuple)):
        score_note_array = music.note_array_from_part_list(list(score))
    else:
        score_note_array = music.note_array_from_part(score)

    perf_note_array = perf.note_array()

    # Prefer beat-based time on score; seconds on performance.
    if "onset_beat" not in score_note_array.dtype.names:
        raise RuntimeError("partitura score note array missing 'onset_beat' field")
    if "onset_sec" not in perf_note_array.dtype.names:
        raise RuntimeError("partitura performance note array missing 'onset_sec' field")
    if "pitch" not in score_note_array.dtype.names or "pitch" not in perf_note_array.dtype.names:
        raise RuntimeError("partitura note arrays missing 'pitch' field")

    score_groups = _group_note_array(
        score_note_array,
        onset_field="onset_beat",
        pitch_field="pitch",
        onset_tolerance=score_grouping_beats,
    )
    perf_groups = _group_note_array(
        perf_note_array,
        onset_field="onset_sec",
        pitch_field="pitch",
        onset_tolerance=float(performance_grouping_ms) / 1000.0,
    )

    matches = _align_groups_dp(
        score_groups,
        perf_groups,
        gap_penalty=gap_penalty,
        min_similarity_for_anchor=min_similarity_for_anchor,
    )

    # Build anchor points and enforce monotonicity.
    anchor_x: List[float] = []
    anchor_y: List[float] = []
    for si, pj, sim in matches:
        sx = float(score_groups[si]["onset"])
        py = float(min(perf_groups[pj]["onsets"]))  # earliest onset in the performance chord
        anchor_x.append(sx)
        anchor_y.append(py)

    if not anchor_x:
        # Fallback: use a simple linear scaling based on durations.
        score_end = float(np.max(score_note_array["onset_beat"])) if len(score_note_array) else 0.0
        perf_end = float(np.max(perf_note_array["onset_sec"])) if len(perf_note_array) else 0.0
        scale = (perf_end / score_end) if score_end > 1e-9 else 1.0

        def warp(x: float) -> float:
            return max(0.0, float(x) * scale)

        return warp, AlignmentDiagnostics(
            n_score_groups=len(score_groups),
            n_perf_groups=len(perf_groups),
            n_anchor_points=0,
            perf_time_shift=0.0,
        )

    # Sort by score time (just in case)
    order = np.argsort(np.asarray(anchor_x))
    xs = np.asarray(anchor_x, dtype=float)[order]
    ys = np.asarray(anchor_y, dtype=float)[order]

    # Drop non-monotonic (in performance time) anchors.
    keep_xs: List[float] = []
    keep_ys: List[float] = []
    last_y = -np.inf
    for x, y in zip(xs.tolist(), ys.tolist()):
        if y >= last_y - 1e-6:
            keep_xs.append(float(x))
            keep_ys.append(float(y))
            last_y = y

    xs = np.asarray(keep_xs, dtype=float)
    ys = np.asarray(keep_ys, dtype=float)

    perf_shift = float(ys[0]) if start_at_zero else 0.0
    ys = ys - perf_shift

    # Make sure we have at least 2 points for interpolation.
    if len(xs) == 1:
        x0 = float(xs[0])
        y0 = float(ys[0])
        # Use performance duration ratio as slope.
        score_end = float(np.max(score_note_array["onset_beat"])) if len(score_note_array) else x0
        perf_end = float(np.max(perf_note_array["onset_sec"])) - perf_shift if len(perf_note_array) else y0
        slope = (perf_end - y0) / (score_end - x0) if abs(score_end - x0) > 1e-9 else 1.0

        def warp(x: float) -> float:
            return max(0.0, y0 + (float(x) - x0) * slope)

        return warp, AlignmentDiagnostics(
            n_score_groups=len(score_groups),
            n_perf_groups=len(perf_groups),
            n_anchor_points=int(len(xs)),
            perf_time_shift=perf_shift,
        )

    # Slopes for extrapolation.
    s0 = (ys[1] - ys[0]) / (xs[1] - xs[0]) if abs(xs[1] - xs[0]) > 1e-9 else 1.0
    s1 = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) if abs(xs[-1] - xs[-2]) > 1e-9 else 1.0

    def warp(x: float) -> float:
        xf = float(x)
        if xf <= xs[0]:
            return max(0.0, float(ys[0] + (xf - xs[0]) * s0))
        if xf >= xs[-1]:
            return max(0.0, float(ys[-1] + (xf - xs[-1]) * s1))
        return max(0.0, float(np.interp(xf, xs, ys)))

    return warp, AlignmentDiagnostics(
        n_score_groups=len(score_groups),
        n_perf_groups=len(perf_groups),
        n_anchor_points=int(len(xs)),
        perf_time_shift=perf_shift,
    )


