"""
Diarization utilities (Pyannote)
- Loads pyannote pipeline with HF token
- Runs diarization and returns normalized segments: [{start, end, speaker}]
- Saves RTTM and JSON artifacts
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pathlib import Path

from config import (
    HF_TOKEN,
    PYANNOTE_DIARIZATION_MODEL,
    SAVE_RTTM,
    DIARIZATION_DIR,
)
from utils import setup_logger, save_json, format_duration

logger = setup_logger("diarization")

# Global cache
_DIARIZATION_PIPELINE = None


@dataclass
class DiarSeg:
    start: float
    end: float
    speaker: str


def load_diarization_pipeline():
    """
    Load and cache pyannote speaker diarization pipeline.
    """
    global _DIARIZATION_PIPELINE
    if _DIARIZATION_PIPELINE is not None:
        return _DIARIZATION_PIPELINE

    try:
        from pyannote.audio import Pipeline  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pyannote.audio not installed. Please `pip install pyannote.audio`."
        ) from e

    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set — pyannote may prompt for auth in some environments.")

    logger.info(f"Loading diarization pipeline: {PYANNOTE_DIARIZATION_MODEL}")
    _DIARIZATION_PIPELINE = Pipeline.from_pretrained(
        PYANNOTE_DIARIZATION_MODEL,
        use_auth_token=HF_TOKEN
    )

    # Disable overlapping speakers in the output for a simpler Phase-1 pipeline
    try:
        _DIARIZATION_PIPELINE.segmentation.onset = 0.5  # mild denoising; optional
    except Exception:
        pass

    logger.info("✅ Pyannote pipeline loaded.")
    return _DIARIZATION_PIPELINE


def unload_diarization_pipeline():
    """
    Release cached pipeline (mainly to free GPU RAM in long sessions).
    """
    global _DIARIZATION_PIPELINE
    _DIARIZATION_PIPELINE = None


def run_diarization(audio_path: str, pipeline=None) -> Any:
    """
    Run diarization on an audio file and return pyannote Annotation.
    """
    if pipeline is None:
        pipeline = load_diarization_pipeline()
    logger.info(f"Diarizing: {Path(audio_path).name}")
    annotation = pipeline(audio_path)
    logger.info("✅ Diarization complete.")
    return annotation


def diarization_to_segments(annotation) -> List[Dict]:
    """
    Convert pyannote Annotation to a sorted list of dicts:
    [{start: float, end: float, speaker: 'SPEAKER_00'}]
    """
    segments: List[DiarSeg] = []

    # Each labeled track (segment) has a time interval and a speaker label
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        seg = DiarSeg(
            start=float(turn.start),
            end=float(turn.end),
            speaker=str(speaker),
        )
        # Guard against degenerate/negative spans
        if seg.end > seg.start:
            segments.append(seg)

    # Sort by time
    segments.sort(key=lambda s: (s.start, s.end))

    # Normalize speaker labels to SPEAKER_00 style if needed
    mapping: Dict[str, str] = {}
    normalized: List[Dict] = []
    next_id = 0
    for s in segments:
        if s.speaker not in mapping:
            mapping[s.speaker] = f"SPEAKER_{next_id:02d}"
            next_id += 1
        normalized.append({"start": s.start, "end": s.end, "speaker": mapping[s.speaker]})

    return normalized


def save_diarization(annotation, base_path: str) -> Dict[str, str]:
    """
    Save diarization artifacts:
      - <base_path>.rttm (if SAVE_RTTM=True)
      - <base_path>.json (normalized segments)
    Returns dict of file paths that were written.
    """
    out: Dict[str, str] = {}
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    # Normalized JSON segments
    segments = diarization_to_segments(annotation)
    json_path = base.with_suffix(".json")
    save_json(segments, json_path)
    out["json"] = str(json_path)

    # RTTM export (optional)
    if SAVE_RTTM:
        try:
            rttm_path = base.with_suffix(".rttm")
            with rttm_path.open("w", encoding="utf-8") as f:
                annotation.write_rttm(f)
            out["rttm"] = str(rttm_path)
        except Exception as e:
            logger.warning(f"Could not save RTTM: {e}")

    logger.info(f"💾 Diarization saved: {out}")
    return out


def print_speaker_stats(diarization_segments: List[Dict]) -> None:
    """
    Print simple stats by speaker from normalized segments list.
    """
    if not diarization_segments:
        logger.info("No diarization segments to summarize.")
        return

    # Aggregate durations
    from collections import defaultdict
    dur_by_spk = defaultdict(float)
    total = 0.0
    for s in diarization_segments:
        d = float(s["end"]) - float(s["start"])
        if d > 0:
            dur_by_spk[s["speaker"]] += d
            total += d

    uniq = len(dur_by_spk)
    logger.info("\n👥 Diarization — Speaker Statistics")
    logger.info(f"   Unique speakers: {uniq}")
    logger.info(f"   Total labeled duration: {format_duration(total)}")
    logger.info("   Breakdown:")
    for spk in sorted(dur_by_spk.keys()):
        d = dur_by_spk[spk]
        pct = (d / total * 100.0) if total > 0 else 0.0
        logger.info(f"      {spk}: {format_duration(d)} ({pct:.1f}%)")
