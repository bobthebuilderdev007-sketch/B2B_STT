"""
Transcription utilities
- Loads Whisper (OpenAI-Whisper or Faster-Whisper if available)
- Exposes a unified .transcribe(...) that accepts file paths or numpy waveforms
- Normalizes outputs and provides extract_segments(...)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from config import (
    DEVICE,
    WHISPER_MODEL_SIZE,
    WHISPER_TASK,
    WHISPER_VERBOSE,
    WHISPER_FP16,
)

# Globals for lifecycle control
_WHISPER_BACKEND: Optional[str] = None  # "openai" | "faster"
_WHISPER_MODEL = None


# -----------------------------------------------------------------------------
# Backend loaders
# -----------------------------------------------------------------------------
def _try_load_openai_whisper() -> Optional[Any]:
    try:
        import whisper  # OpenAI-Whisper
        return whisper
    except Exception:
        return None


def _try_load_faster_whisper() -> Optional[Any]:
    try:
        from faster_whisper import WhisperModel  # type: ignore
        return WhisperModel
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Wrapper
# -----------------------------------------------------------------------------
class _WhisperWrapper:
    """
    Simple unifying wrapper so pipeline code can call:
        model.transcribe(audio, language=None)
    and get a dict with keys: text, language, segments (list of {start,end,text,...})
    """

    def __init__(self, backend: str, model_impl: Any):
        self.backend = backend
        self._impl = model_impl

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        task: str = WHISPER_TASK,
        verbose: bool = WHISPER_VERBOSE,
    ) -> Dict[str, Any]:
        if self.backend == "openai":
            return self._transcribe_openai(audio, language, task, verbose)
        else:
            return self._transcribe_faster(audio, language, task)

    # --- OpenAI-Whisper path ---
    def _transcribe_openai(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str],
        task: str,
        verbose: bool,
    ) -> Dict[str, Any]:
        # OpenAI whisper expects audio as path or numpy array (float32, -1..1)
        result = self._impl.transcribe(
            audio=audio,
            language=language,
            task=task,
            verbose=verbose,
            fp16=WHISPER_FP16 if DEVICE == "cuda" else False,
        )
        # 'result' already has 'text', 'language', 'segments'
        # Ensure expected fields exist
        result.setdefault("language", result.get("language", "unknown"))
        result.setdefault("segments", result.get("segments", []))
        return result

    # --- Faster-Whisper path ---
    def _transcribe_faster(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str],
        task: str,
    ) -> Dict[str, Any]:
        # faster-whisper returns (segments_iterator, info)
        # It can take a path or numpy array directly.
        segments, info = self._impl.transcribe(
            audio=audio,
            language=language,
            task=task,
        )
        out_segments: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []
        for seg in segments:
            # seg has .start, .end, .text, .avg_logprob (optional), .no_speech_prob, etc.
            out = {
                "id": getattr(seg, "id", None),
                "start": float(getattr(seg, "start", 0.0)),
                "end": float(getattr(seg, "end", 0.0)),
                "text": getattr(seg, "text", "") or "",
            }
            # Attach optional confidence-like fields if present
            avg_lp = getattr(seg, "avg_logprob", None)
            if avg_lp is not None:
                out["avg_logprob"] = float(avg_lp)
            nsp = getattr(seg, "no_speech_prob", None)
            if nsp is not None:
                out["no_speech_prob"] = float(nsp)

            out_segments.append(out)
            if out["text"]:
                full_text_parts.append(out["text"].strip())

        language_code = getattr(info, "language", None) or "unknown"
        full_text = " ".join(full_text_parts).strip()

        return {
            "text": full_text,
            "language": language_code,
            "segments": out_segments,
        }


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def load_whisper_model() -> _WhisperWrapper:
    """
    Load Whisper model (tries Faster-Whisper first for speed, then OpenAI-Whisper).
    Returns a wrapper with .transcribe(...)
    """
    global _WHISPER_BACKEND, _WHISPER_MODEL

    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    # Prefer Faster-Whisper for performance if present
    Faster = _try_load_faster_whisper()
    if Faster is not None:
        compute_type = "float16" if (DEVICE == "cuda" and WHISPER_FP16) else "float32"
        device = "cuda" if DEVICE == "cuda" else "cpu"
        model = Faster(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
        _WHISPER_BACKEND = "faster"
        _WHISPER_MODEL = _WhisperWrapper("faster", model)
        return _WHISPER_MODEL

    # Fallback to OpenAI-Whisper
    whisper = _try_load_openai_whisper()
    if whisper is None:
        raise RuntimeError(
            "No Whisper backend found. Please install either 'faster-whisper' or 'whisper'."
        )

    model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
    _WHISPER_BACKEND = "openai"
    _WHISPER_MODEL = _WhisperWrapper("openai", model)
    return _WHISPER_MODEL


def unload_whisper_model() -> None:
    """Release model references (helps when running many files)."""
    global _WHISPER_MODEL, _WHISPER_BACKEND
    _WHISPER_MODEL = None
    _WHISPER_BACKEND = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_segments(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize and return segment list with at least: start, end, text.
    The input 'result' is whatever the wrapper returns.
    """
    segs = result.get("segments", [])
    norm: List[Dict[str, Any]] = []
    for s in segs:
        start = float(s.get("start", 0.0))
        end = float(s.get("end", start))
        text = (s.get("text") or "").strip()
        row = {
            "start": start,
            "end": end,
            "text": text,
        }
        # propagate optional fields if present
        if "avg_logprob" in s:
            row["avg_logprob"] = s["avg_logprob"]
        if "no_speech_prob" in s:
            row["no_speech_prob"] = s["no_speech_prob"]
        norm.append(row)
    return norm
