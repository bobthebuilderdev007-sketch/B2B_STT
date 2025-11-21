"""
Audio processing helpers for Phase-1
- Validation for supported media (audio & video)
- Video → audio extraction
- Optional format conversion & chunking
- Basic audio info (duration, sample rate, channels)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import librosa
import soundfile as sf

from config import (
    SUPPORTED_EXTENSIONS,
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    SAMPLE_RATE,
    CHANNELS,
    MAX_CHUNK_DURATION,
    CHUNK_OVERLAP,
    CHUNKS_DIR,
    TEMP_DIR,
)
from utils import setup_logger, format_duration

logger = setup_logger("audio_processing")


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def validate_audio(path: str) -> Tuple[bool, str]:
    """
    Quick checks + attempt to read a short snippet to ensure the file is decodable.
    Accepts either audio or video (video will be handled by the caller to extract audio).
    Returns (is_valid, error_message_if_any)
    """
    p = Path(path)
    if not p.exists():
        return False, f"File not found: {path}"

    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file extension: {ext}"

    # Fast sanity: size > 0
    if os.path.getsize(path) == 0:
        return False, "File is empty"

    # If already an audio file, try decoding a tiny slice
    if ext in AUDIO_EXTENSIONS:
        try:
            _y, _sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, offset=0.0, duration=0.5)
        except Exception as e:
            return False, f"Failed to decode audio: {e}"

    # For video: decoding is handled after extraction
    return True, ""


# -----------------------------------------------------------------------------
# Video → Audio extraction
# -----------------------------------------------------------------------------
def extract_audio_from_video(
    video_path: str,
    output_path: str,
    sample_rate: int = SAMPLE_RATE
) -> Tuple[bool, str]:
    """
    Extract audio from video file and write mono WAV (pcm_s16le).
    Returns (success, output_path_or_error)
    """
    logger.info(f"Extracting audio from video: {Path(video_path).name}")
    try:
        from moviepy.editor import VideoFileClip  # lazy import
    except Exception as e:
        return False, f"moviepy not installed: {e}"

    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio is None:
            video.close()
            return False, "No audio track found in video"

        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        temp_audio = str(TEMP_DIR / f"temp_{Path(video_path).stem}.wav")

        # Write 16-bit PCM WAV with desired samplerate
        audio.write_audiofile(
            temp_audio,
            fps=sample_rate,
            nbytes=2,
            codec="pcm_s16le",
            verbose=False,
            logger=None
        )
        video.close()

        # Load and ensure mono
        y, sr = librosa.load(temp_audio, sr=sample_rate, mono=True)
        sf.write(output_path, y, sample_rate)

        # Cleanup
        try:
            os.remove(temp_audio)
        except Exception:
            pass

        duration = len(y) / sample_rate
        logger.info(f"✅ Audio extracted: {format_duration(duration)} → {output_path}")
        return True, output_path

    except Exception as e:
        logger.error(f"Failed to extract audio: {e}")
        return False, str(e)


# -----------------------------------------------------------------------------
# Audio format conversion
# -----------------------------------------------------------------------------
def convert_audio_format(
    input_path: str,
    output_path: str,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS
) -> Tuple[bool, str]:
    """
    Convert audio to standardized WAV, mono, specified sample rate.
    Returns (success, output_path_or_error)
    """
    logger.info(f"Converting audio format: {Path(input_path).name}")
    try:
        y, sr = librosa.load(input_path, sr=sample_rate, mono=(channels == 1))
        sf.write(output_path, y, sample_rate)
        duration = len(y) / sample_rate
        logger.info(f"✅ Converted ({format_duration(duration)}): {output_path}")
        return True, output_path
    except Exception as e:
        logger.error(f"Failed to convert audio: {e}")
        return False, str(e)


# -----------------------------------------------------------------------------
# Chunking (optional, for very long files)
# -----------------------------------------------------------------------------
def get_audio_duration(audio_path: str) -> float:
    """Return audio duration in seconds."""
    return float(librosa.get_duration(path=audio_path))


def chunk_audio_file(
    audio_path: str,
    max_duration: int = MAX_CHUNK_DURATION,
    overlap: int = CHUNK_OVERLAP,
    output_dir: Optional[Path] = None
) -> Tuple[List[str], List[float]]:
    """
    Split audio into overlapping chunks if longer than max_duration.
    Returns (chunk_paths, chunk_offsets)
    """
    duration = get_audio_duration(audio_path)
    if duration <= max_duration:
        logger.info(f"Audio duration ({duration:.1f}s) <= max ({max_duration}s), no chunking needed")
        return [audio_path], [0.0]

    logger.info(f"Audio duration: {format_duration(duration)}")
    logger.info(f"Splitting into chunks (max: {max_duration}s, overlap: {overlap}s)")

    if output_dir is None:
        output_dir = CHUNKS_DIR

    base_name = Path(audio_path).stem
    chunk_dir = output_dir / base_name
    chunk_dir.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    chunk_paths: List[str] = []
    chunk_offsets: List[float] = []

    chunk_samples = int(max_duration * sr)
    overlap_samples = int(overlap * sr)
    step_samples = max(1, chunk_samples - overlap_samples)

    idx = 0
    for start in range(0, len(y), step_samples):
        end = min(start + chunk_samples, len(y))
        chunk = y[start:end]
        cpath = chunk_dir / f"chunk_{idx:03d}.wav"
        sf.write(str(cpath), chunk, sr)

        chunk_paths.append(str(cpath))
        chunk_offsets.append(start / sr)
        idx += 1
        if end >= len(y):
            break

    logger.info(f"Created {len(chunk_paths)} chunks")
    return chunk_paths, chunk_offsets


# -----------------------------------------------------------------------------
# Audio info
# -----------------------------------------------------------------------------
def get_audio_info(path: str) -> Dict:
    """
    Return basic info for an audio file: duration, sample_rate, channels.
    Falls back gracefully if something can’t be probed.
    """
    info = {
        "path": str(path),
        "filename": Path(path).name,
        "duration": 0.0,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2) if os.path.exists(path) else 0.0,
        "extension": Path(path).suffix.lower(),
    }

    try:
        # Fast header probe when supported
        import soundfile as _sf
        with _sf.SoundFile(path) as f:
            info["sample_rate"] = int(f.samplerate)
            info["channels"] = int(f.channels)
            info["duration"] = float(len(f) / f.samplerate)
            return info
    except Exception:
        try:
            y, sr = librosa.load(path, sr=None, mono=False, duration=1.0)
            info["sample_rate"] = int(sr)
            if y.ndim == 1:
                info["channels"] = 1
            else:
                info["channels"] = y.shape[0]
            # duration via librosa (slower if full file)
            info["duration"] = float(librosa.get_duration(path=path))
        except Exception:
            pass

    return info


def print_audio_info(path: str) -> None:
    """Pretty-print audio info to console."""
    info = get_audio_info(path)
    logger.info("\n🎵 Audio Info")
    logger.info(f"   File: {Path(path).name}")
    logger.info(f"   Size: {info['size_mb']} MB")
    logger.info(f"   Duration: {format_duration(info['duration'])}")
    logger.info(f"   Sample rate: {info['sample_rate']} Hz")
    logger.info(f"   Channels: {info['channels']}")
    logger.info(f"   Extension: {info['extension']}")
