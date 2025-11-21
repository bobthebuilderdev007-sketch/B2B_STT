"""
Entry point for Phase-1 ASR (multilingual + diarization)
- Supports single file or directory batch
- Auto-extracts audio if an input is a video file
- Uses per-segment transcription by default (config-driven)
"""

import argparse
import sys
from pathlib import Path
from typing import List

from config import (
    AUDIO_DIR,
    OUTPUTS_DIR,
    VIDEO_EXTENSIONS,
    PER_SEGMENT_MODE,
    LANGUAGE_MODE,
    MIN_SEG_DUR,
    PAD_SEC,
    print_config,
    validate_config,
)
from utils import setup_logger, scan_media_files, print_file_summary
from pipeline import process_audio_file, process_batch
from audio_processing import extract_audio_from_video, print_audio_info, validate_audio

logger = setup_logger("main")


def _gather_inputs(input_path: Path) -> List[str]:
    """Return a list of audio/video files from a file or directory."""
    if input_path.is_file():
        return [str(input_path)]
    if input_path.is_dir():
        audio_files, video_files = scan_media_files(input_path)
        # Process videos too (we'll extract their audio later)
        return audio_files + video_files
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _ensure_audio(path: str, extracted_dir: Path) -> str:
    """
    If `path` is a video, extract audio to WAV in `extracted_dir` and return the WAV path.
    If already audio, return `path` unchanged.
    """
    p = Path(path)
    if p.suffix.lower() in VIDEO_EXTENSIONS:
        extracted_dir.mkdir(parents=True, exist_ok=True)
        wav_path = extracted_dir / f"{p.stem}.wav"
        if wav_path.exists():
            logger.info(f"Audio already extracted: {wav_path.name}")
            return str(wav_path)
        ok, out = extract_audio_from_video(str(p), str(wav_path))
        if not ok:
            raise RuntimeError(f"Failed to extract audio from video '{p.name}': {out}")
        return str(wav_path)
    return path


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase-1 ASR (Diarization + Multilingual Per-Segment Transcription)"
    )
    p.add_argument(
        "-i", "--input",
        type=str,
        default=str(AUDIO_DIR),
        help="Path to an audio/video file or a directory containing audio/video files (default: config.AUDIO_DIR)",
    )
    p.add_argument(
        "-o", "--output",
        type=str,
        default=str(OUTPUTS_DIR / "merged"),
        help="Output directory for merged transcripts (default: outputs/merged)",
    )
    p.add_argument(
        "--language",
        type=str,
        default=None,
        help="(Optional) Force a single language code for GLOBAL mode only (e.g., 'en'). "
             "Ignored if PER_SEGMENT_MODE=True unless LANGUAGE_MODE starts with 'forced:'."
    )
    p.add_argument(
        "--no-print-config",
        action="store_true",
        help="Do not print the configuration summary at startup."
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.no_print_config:
        print_config()
        logger.info(
            f"⚙️  Runtime: PER_SEGMENT_MODE={PER_SEGMENT_MODE} | "
            f"LANGUAGE_MODE={LANGUAGE_MODE} | MIN_SEG_DUR={MIN_SEG_DUR}s | PAD_SEC={PAD_SEC}s"
        )

    if not validate_config():
        logger.error("Configuration validation failed. Fix the issues above and re-run.")
        sys.exit(1)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        files = _gather_inputs(input_path)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)

    if not files:
        logger.warning("No input files found.")
        sys.exit(0)

    # Where to place extracted audio from videos
    extracted_dir = OUTPUTS_DIR / "extracted_audio"
    prepared_files: List[str] = []

    if len(files) == 1:
        # Single-file mode
        src = files[0]
        try:
            # If it's a video, extract audio first
            audio_path = _ensure_audio(src, extracted_dir)
            ok, err = validate_audio(audio_path)
            if not ok:
                logger.error(f"Invalid input: {err}")
                sys.exit(1)

            # Print info for the (possibly extracted) audio
            print_audio_info(audio_path)

            logger.info(f"Processing: {Path(audio_path).name}")
            result = process_audio_file(
                audio_path=audio_path,
                output_dir=output_dir,
                language=args.language,
                save_intermediate=True,
            )
            if not result.get("success"):
                sys.exit(1)
        except Exception as e:
            logger.error(f"Failed: {e}")
            sys.exit(1)
    else:
        # Batch mode — extract audio for all videos first
        for src in files:
            try:
                audio_path = _ensure_audio(src, extracted_dir)
                ok, err = validate_audio(audio_path)
                if not ok:
                    logger.warning(f"Skipping '{Path(src).name}': {err}")
                    continue
                prepared_files.append(audio_path)
            except Exception as e:
                logger.warning(f"Skipping '{Path(src).name}': {e}")

        if not prepared_files:
            logger.error("No valid audio files after preparing inputs.")
            sys.exit(1)

        # Show a quick summary
        logger.info("Discovered input files (after preparation):")
        print_file_summary(prepared_files, [], max_display=8)

        process_batch(
            audio_files=prepared_files,
            output_dir=output_dir,
            language=args.language,
        )

    logger.info("✅ Done.")


if __name__ == "__main__":
    main()
