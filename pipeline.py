"""
Complete ASR pipeline with transcription and diarization
Phase-1: supports two modes
  • PER_SEGMENT_MODE=True  → Diarize → slice per speaker turn → transcribe each slice
  • PER_SEGMENT_MODE=False → Transcribe full file once → map ASR segments to speakers

Outputs richer fields per row: start, end, duration, speaker, language, text, confidence

Notes on language control (important):
- If the CLI provides --language <code>, that language is ALWAYS used for ASR
  (both legacy and per-segment paths), overriding LANGUAGE_MODE.
- Otherwise:
    * If LANGUAGE_MODE == "forced:<code>", that code is used.
    * If LANGUAGE_MODE == "per_segment", language is auto-detected per slice.
    * If LANGUAGE_MODE == "global_auto", language hint is None (auto-detect on whole audio).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import librosa
import torch

from config import (
    MERGED_DIR,
    REPORTS_DIR,
    CLEAR_CACHE_AFTER_PROCESSING,
    SAMPLE_RATE,
    PER_SEGMENT_MODE,
    LANGUAGE_MODE,
    MIN_SEG_DUR,
    PAD_SEC,
)
from utils import (
    setup_logger, save_json, save_csv, format_duration,
)
from audio_processing import get_audio_info, validate_audio
from transcription import (
    load_whisper_model, extract_segments, unload_whisper_model
)
from diarization import (
    load_diarization_pipeline, run_diarization, diarization_to_segments,
    print_speaker_stats, save_diarization, unload_diarization_pipeline
)

logger = setup_logger("pipeline")

# =============================================================================
# Helpers
# =============================================================================

def _avg_conf_from_logs(segments: List[Dict]) -> Optional[float]:
    """
    Heuristic confidence from avg_logprob (if present). Returns None if unknown.
    Whisper/Faster-Whisper provide negative logprobs; map to ~[0..1] via sigmoid.
    """
    vals = [s.get("avg_logprob") for s in segments if isinstance(s.get("avg_logprob"), (int, float))]
    if not vals:
        return None
    x = float(np.mean(vals))
    # Squash: higher (less negative) → closer to 1
    # This is a heuristic just to give clients a relative feel.
    return float(1 / (1 + np.exp(-(x * 2.0))))

def _merge_short_turns(diar_segments: List[Dict], min_dur: float) -> List[Dict]:
    """
    Merge very short consecutive turns from the same speaker to stabilize slicing.
    Keeps chronological order.
    """
    if not diar_segments:
        return []
    merged: List[Dict] = []
    cur = dict(diar_segments[0])
    for nxt in diar_segments[1:]:
        if nxt["speaker"] == cur["speaker"] and (cur["end"] >= nxt["start"] or (nxt["start"] - cur["end"] <= 0.05)):
            # contiguous/overlapping same speaker → extend
            cur["end"] = max(cur["end"], nxt["end"])
        else:
            # finalize current, but if it's too short try to absorb tiny gaps with the next different speaker
            if (cur["end"] - cur["start"]) < min_dur and merged:
                # try to glue with previous if same spk
                prev = merged[-1]
                if prev["speaker"] == cur["speaker"] and (cur["start"] - prev["end"]) <= 0.05:
                    prev["end"] = cur["end"]
                else:
                    merged.append(cur)
            else:
                merged.append(cur)
            cur = dict(nxt)
    # tail
    if merged and (cur["end"] - cur["start"]) < min_dur:
        prev = merged[-1]
        if prev["speaker"] == cur["speaker"] and (cur["start"] - prev["end"]) <= 0.05:
            prev["end"] = cur["end"]
        else:
            merged.append(cur)
    else:
        merged.append(cur)
    # ensure strictly increasing and valid
    out = []
    for s in merged:
        if s["end"] > s["start"]:
            out.append(s)
    return out

def _slice_audio(y: np.ndarray, sr: int, start: float, end: float, pad: float) -> np.ndarray:
    s = max(0.0, start - pad)
    e = end + pad
    i0 = int(round(s * sr))
    i1 = int(round(e * sr))
    i0 = max(0, min(i0, y.shape[-1]))
    i1 = max(0, min(i1, y.shape[-1]))
    return y[i0:i1]

def _forced_language() -> Optional[str]:
    if LANGUAGE_MODE.startswith("forced:"):
        code = LANGUAGE_MODE.split(":", 1)[1].strip() or None
        return code
    return None

def _smart_assign_speaker(seg_start: float, seg_end: float, diar: List[Dict]) -> str:
    """
    Assign speaker by maximum overlap; if no overlap, pick the speaker whose
    segment is closest to the ASR segment center.
    """
    center = 0.5 * (seg_start + seg_end)
    max_overlap = 0.0
    assigned = None
    closest_dist = float("inf")
    closest_spk = None

    for d in diar:
        ds, de = d["start"], d["end"]
        # overlap
        o = max(0.0, min(seg_end, de) - max(seg_start, ds))
        if o > max_overlap:
            max_overlap = o
            assigned = d["speaker"]
        # distance to center (for backstop)
        if ds <= center <= de:
            closest_dist = 0.0
            closest_spk = d["speaker"]
        else:
            # distance to nearest edge
            dist = min(abs(center - ds), abs(center - de))
            if dist < closest_dist:
                closest_dist = dist
                closest_spk = d["speaker"]

    return assigned or closest_spk or "Unknown"

# =============================================================================
# Legacy path (global transcription, then overlap merge)
# =============================================================================
def _legacy_merge(audio_path: str,
                  output_dir: Path,
                  language: Optional[str],
                  save_intermediate: bool) -> Dict:
    """
    Transcribe whole file once; assign speakers by overlap; enrich rows.
    """
    base_name = Path(audio_path).stem

    # 1) Transcribe
    whisper_model = load_whisper_model()
    # If CLI passed language → honor it; else try forced from config; else None (auto)
    lang_arg = language or _forced_language() or None
    if lang_arg:
        logger.info(f"🔒 Forcing transcription language = {lang_arg} (legacy/global mode)")
    result = whisper_model.transcribe(audio_path, language=lang_arg)
    trans_segments = extract_segments(result)

    if save_intermediate:
        from config import TRANSCRIPTIONS_DIR
        trans_path = Path(TRANSCRIPTIONS_DIR) / f"{base_name}_transcription.json"
        save_json(result, trans_path)

    # 2) Diarize
    diar_pipeline = load_diarization_pipeline()
    diar_annotation = run_diarization(audio_path, pipeline=diar_pipeline)
    diar_segments = diarization_to_segments(diar_annotation)

    if save_intermediate:
        from config import DIARIZATION_DIR
        save_diarization(diar_annotation, str(Path(DIARIZATION_DIR) / base_name))

    # 3) Merge
    merged: List[Dict] = []
    for s in trans_segments:
        speaker = _smart_assign_speaker(s["start"], s["end"], diar_segments)
        row = {
            "start": s["start"],
            "end": s["end"],
            "duration": round(s["end"] - s["start"], 3),
            "speaker": speaker,
            "language": result.get("language", "unknown"),
            "text": s.get("text", "").strip(),
        }
        conf = _avg_conf_from_logs([s])
        if conf is not None:
            row["confidence"] = round(conf, 4)
        merged.append(row)

    return {
        "merged": merged,
        "diar_segments": diar_segments,
        "asr_language": result.get("language", "unknown"),
        "transcription_raw": result,
    }

# =============================================================================
# Per-segment path (diarize → slice → transcribe each turn)
# =============================================================================
def _per_segment_merge(audio_path: str,
                       output_dir: Path,
                       language: Optional[str],
                       save_intermediate: bool) -> Dict:
    """
    Diarize first, then for each speaker turn slice audio and transcribe
    that slice with language detection/forcing per config.

    LANGUAGE OVERRIDE ORDER (highest → lowest):
        1) CLI --language <code>
        2) LANGUAGE_MODE == "forced:<code>"
        3) LANGUAGE_MODE == "per_segment" → auto-detect per slice
        4) LANGUAGE_MODE == "global_auto" → auto-detect
    """
    base_name = Path(audio_path).stem

    # 1) Diarize
    diar_pipeline = load_diarization_pipeline()
    diar_annotation = run_diarization(audio_path, pipeline=diar_pipeline)
    diar_segments = diarization_to_segments(diar_annotation)
    diar_segments = _merge_short_turns(diar_segments, MIN_SEG_DUR)

    if save_intermediate:
        from config import DIARIZATION_DIR
        save_diarization(diar_annotation, str(Path(DIARIZATION_DIR) / base_name))

    # 2) Load full audio once
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # 3) Transcribe each slice
    whisper_model = load_whisper_model()
    forced_cfg = _forced_language()

    # If user supplied --language, lock to it for all slices
    locked_language = language or forced_cfg or None
    if locked_language:
        logger.info(f"🔒 Forcing transcription language = {locked_language} (per-segment mode)")

    merged: List[Dict] = []
    all_slice_meta: List[Dict] = []

    for turn in diar_segments:
        s, e, spk = float(turn["start"]), float(turn["end"]), turn["speaker"]
        if e <= s:
            continue

        slice_wav = _slice_audio(y, sr, s, e, PAD_SEC)
        if slice_wav.size == 0:
            continue

        # Choose language for this slice (CLI overrides everything)
        if locked_language is not None:
            lang_arg: Optional[str] = locked_language
        else:
            if LANGUAGE_MODE == "per_segment":
                lang_arg = None  # auto-detect per slice
            elif LANGUAGE_MODE == "global_auto":
                lang_arg = None  # auto-detect globally (no hint available here)
            else:
                # Default: auto
                lang_arg = None

        res = whisper_model.transcribe(slice_wav, language=lang_arg)
        segs = extract_segments(res)

        # Combine slice segments into a single row covering [s,e]
        text = " ".join([t.get("text", "").strip() for t in segs if t.get("text")]).strip()
        conf = _avg_conf_from_logs(segs)
        row = {
            "start": s,
            "end": e,
            "duration": round(e - s, 3),
            "speaker": spk,
            "language": res.get("language", locked_language or "unknown"),
            "text": text,
        }
        if conf is not None:
            row["confidence"] = round(conf, 4)
        merged.append(row)

        # keep slice meta if needed later
        all_slice_meta.append({
            "turn": dict(turn),
            "slice_language": res.get("language"),
            "num_inner_segments": len(segs),
        })

    return {
        "merged": merged,
        "diar_segments": diar_segments,
        "slices_meta": all_slice_meta,
    }

# =============================================================================
# PUBLIC API
# =============================================================================

def process_audio_file(audio_path: str,
                      output_dir: Optional[Path] = None,
                      language: Optional[str] = None,
                      save_intermediate: bool = True) -> Dict:
    """
    Process audio file end-to-end and write merged CSV/JSON.
    """
    start_time = datetime.now()

    logger.info("=" * 70)
    logger.info(f"PROCESSING: {os.path.basename(audio_path)}")
    logger.info("=" * 70)

    # Validate
    is_valid, error_msg = validate_audio(audio_path)
    if not is_valid:
        logger.error(f"Invalid input: {error_msg}")
        return {'success': False, 'audio_path': audio_path, 'error': error_msg}

    # Out dir
    output_dir = Path(output_dir or MERGED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(audio_path).stem

    try:
        # Branch by mode
        if PER_SEGMENT_MODE:
            logger.info("\n[1/2] Diarization → per-segment transcription…")
            pack = _per_segment_merge(audio_path, output_dir, language, save_intermediate)
        else:
            logger.info("\n[1/2] Global transcription → speaker assignment by overlap…")
            pack = _legacy_merge(audio_path, output_dir, language, save_intermediate)

        merged_segments = pack["merged"]
        diarization_segments = pack.get("diar_segments", [])

        # Save merged artifacts
        out_json = output_dir / f"{base_name}_transcript.json"
        out_csv = output_dir / f"{base_name}_transcript.csv"

        save_json(merged_segments, out_json)
        save_csv(merged_segments, out_csv)

        # Stats & logs
        processing_time = (datetime.now() - start_time).total_seconds()
        audio_info = get_audio_info(audio_path)
        audio_duration = audio_info.get('duration', 0.0)

        logger.info("\n" + "=" * 70)
        logger.info("✅ PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\n📊 Results:")
        logger.info(f"   Audio duration: {format_duration(audio_duration)}")
        logger.info(f"   Processing time: {format_duration(processing_time)}")
        if audio_duration > 0:
            logger.info(f"   Real-time factor: {processing_time / audio_duration:.2f}x")
        if not PER_SEGMENT_MODE:
            logger.info(f"   Global language: {pack.get('asr_language', 'unknown')}")
        logger.info(f"   Total segments: {len(merged_segments)}")

        print_speaker_stats(diarization_segments)

        # Sample
        print(f"\n📄 Sample Output (first 5 rows):")
        for seg in merged_segments[:5]:
            print(f"   [{seg['start']:.1f}s - {seg['end']:.1f}s] "
                  f"{seg['speaker']} ({seg.get('language','?')}): {seg.get('text','')[:120]}")

        if len(merged_segments) > 5:
            print(f"   ... and {len(merged_segments) - 5} more segments")

        logger.info(f"\n💾 Output saved:")
        logger.info(f"   JSON: {out_json}")
        logger.info(f"   CSV:  {out_csv}")

        # Cleanup caches if requested
        if CLEAR_CACHE_AFTER_PROCESSING and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'success': True,
            'audio_path': audio_path,
            'audio_duration': audio_duration,
            'processing_time': processing_time,
            'num_segments': len(merged_segments),
            'num_speakers': len({s['speaker'] for s in diarization_segments}),
            'output_json': str(out_json),
            'output_csv': str(out_csv),
        }

    except Exception as e:
        logger.error(f"\n❌ Processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'success': False, 'audio_path': audio_path, 'error': str(e)}


def process_batch(audio_files: List[str],
                 output_dir: Optional[Path] = None,
                 language: Optional[str] = None) -> List[Dict]:
    """
    Batch process multiple files (keeps models loaded).
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"BATCH PROCESSING: {len(audio_files)} files")
    logger.info(f"{'='*70}\n")

    # Load once
    logger.info("Loading models…")
    whisper_model = load_whisper_model()
    diar_pipeline = load_diarization_pipeline()
    logger.info("✅ Models loaded\n")

    results = []
    for i, audio_path in enumerate(audio_files, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"FILE {i}/{len(audio_files)} — {Path(audio_path).name}")
        logger.info(f"{'='*70}\n")

        result = process_audio_file(
            audio_path=audio_path,
            output_dir=output_dir,
            language=language,
            save_intermediate=True
        )
        results.append(result)

    # Report
    logger.info(f"\n{'='*70}")
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*70}\n")

    successful = sum(1 for r in results if r.get('success'))
    failed = len(results) - successful
    logger.info(f"📊 Batch Summary:")
    logger.info(f"   Total files: {len(results)}")
    logger.info(f"   Successful:  {successful}")
    logger.info(f"   Failed:      {failed}")

    if failed:
        logger.info("\n❌ Failed files:")
        for r in results:
            if not r.get('success'):
                logger.info(f"   - {Path(r['audio_path']).name}: {r.get('error', 'Unknown error')}")

    # Save batch report
    report_path = REPORTS_DIR / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_json(results, report_path)
    logger.info(f"\n📋 Batch report saved: {report_path}")

    # Cleanup
    if CLEAR_CACHE_AFTER_PROCESSING:
        unload_whisper_model()
        unload_diarization_pipeline()

    return results
