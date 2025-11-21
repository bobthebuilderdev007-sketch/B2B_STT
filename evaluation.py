# evaluation.py
"""
Evaluation utilities (auto-friendly)
- Auto mode (--guid) finds files and evaluates the **entire transcript** by default.
- Robustly loads merged transcripts (JSON/CSV) with schema:
  [{start,end,duration,speaker,language,text,confidence?}, ...] (case-insensitive)
- Works with your original JSON schema (Text/StartTime/EndTime).
- Computes WER using jiwer (if installed; version-safe)
- Computes DER using pyannote.metrics (if RTTMs exist & libs installed)

CLI examples:
  # Full auto, whole file (no slicing)
  python evaluation.py --guid 94c70cb1-7ddb-45b9-a50f-33c835a22d84

  # Manual paths, whole file
  python evaluation.py --hyp outputs/merged/94c70..._transcript.json --ref data/json/GT_94c70....json

  # Optional: slice 0–120s if you *want* a window
  python evaluation.py --guid 94c70... --first-seconds 120
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import pandas as pd


# ------------------------------ helpers -------------------------------------
def _best_text_field(d: Dict, field_priority: Optional[List[str]] = None) -> Optional[str]:
    defaults = ["text", "ref", "transcript", "sentence", "content", "value", "utterance"]
    fields = (field_priority or []) + defaults
    lower = {k.lower(): k for k in d.keys()}
    for want in fields:
        if want is None:
            continue
        k = lower.get(want.lower())
        if k and isinstance(d.get(k), str):
            val = d[k].strip()
            if val:
                return val
    if "Text" in d and isinstance(d["Text"], str) and d["Text"].strip():
        return d["Text"].strip()
    return None


def _time_from(d: Dict) -> Tuple[Optional[float], Optional[float]]:
    lower = {k.lower(): k for k in d.keys()}
    def get_num(key: str) -> Optional[float]:
        k = lower.get(key.lower())
        if k is None:
            return None
        try:
            return float(d[k])
        except Exception:
            return None

    start = (
        get_num("start") or get_num("starttime") or get_num("start_time")
        or get_num("begin") or get_num("from")
    )
    end = (
        get_num("end") or get_num("endtime") or get_num("end_time")
        or get_num("finish") or get_num("to")
    )
    dur = get_num("duration") or get_num("Duration")
    if end is None and start is not None and dur is not None:
        end = start + dur
    return start, end


def _overlaps_window(start: Optional[float], end: Optional[float], tmin: float, tmax: Optional[float]) -> bool:
    if start is None and end is None:
        return False
    s = start if start is not None else (end if end is not None else 0.0)
    e = end if end is not None else s
    if tmax is None:
        return e > tmin
    return max(s, tmin) < min(e, tmax)


def _slice_segments_by_time(segments: List[Dict], tmin: float = 0.0, tmax: Optional[float] = None) -> List[Dict]:
    if tmax is None and tmin <= 0:
        return segments
    out = []
    for s in segments:
        st, en = _time_from(s)
        if _overlaps_window(st, en, tmin, tmax):
            out.append(s)
    return out


# ------------------------------ I/O helpers ---------------------------------
def _load_segments(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suf = path.suffix.lower()
    if suf == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "segments" in data:
            data = data["segments"]
        if isinstance(data, list):
            if not data or isinstance(data[0], (dict, str)):
                if data and isinstance(data[0], str):
                    return [{"text": x} for x in data if isinstance(x, str)]
                return data
        raise ValueError(f"Unexpected JSON format in {path}")

    if suf == ".csv":
        df = pd.read_csv(path, encoding="utf-8")
        return df.to_dict("records")

    raise ValueError(f"Unsupported file type (expect .json or .csv): {path}")


def _segments_to_text(
    segments: List[Dict],
    join_with: str = " ",
    field_priority: Optional[List[str]] = None,
) -> str:
    parts: List[str] = []
    for s in segments:
        if isinstance(s, dict):
            t = _best_text_field(s, field_priority)
            if t:
                parts.append(t)
        elif isinstance(s, str):
            parts.append(s.strip())
    return join_with.join(parts).strip()


def _load_reference_text(
    ref_path: Path,
    ref_key: Optional[str] = None,
    ref_segments_key: Optional[str] = None,
    ref_text_field: Optional[str] = None,
) -> str:
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference not found: {ref_path}")

    suf = ref_path.suffix.lower()
    if suf == ".txt":
        return ref_path.read_text(encoding="utf-8").strip()

    if suf == ".json":
        data = json.loads(ref_path.read_text(encoding="utf-8"))

        if ref_key and isinstance(data, dict):
            val = data.get(ref_key)
            if isinstance(val, str):
                return val.strip()

        if ref_segments_key and isinstance(data, dict):
            segs = data.get(ref_segments_key)
            if isinstance(segs, list) and segs and isinstance(segs[0], (dict, str)):
                fields = [ref_text_field] if ref_text_field else None
                return _segments_to_text(segs, field_priority=fields)

        if isinstance(data, dict):
            for k in ("text", "Text", "transcript", "reference", "gt", "ground_truth", "content", "full_text"):
                if isinstance(data.get(k), str) and data[k].strip():
                    return data[k].strip()
            if "segments" in data and isinstance(data["segments"], list):
                return _segments_to_text(data["segments"], field_priority=[ref_text_field] if ref_text_field else None)
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], (dict, str)):
                    s = _segments_to_text(v, field_priority=[ref_text_field] if ref_text_field else None)
                    if s:
                        return s
            raise ValueError("Unsupported JSON reference format.")

        if isinstance(data, list):
            fields = [ref_text_field] if ref_text_field else None
            return _segments_to_text(data, field_priority=fields)

        raise ValueError("Unsupported JSON reference format.")

    if suf == ".csv":
        df = pd.read_csv(ref_path, encoding="utf-8")
        rows = df.to_dict("records")
        fields = [ref_text_field] if ref_text_field else None
        return _segments_to_text(rows, field_priority=fields)

    raise ValueError(f"Unsupported reference format: {ref_path}")


# ------------------------------ WER (jiwer) ---------------------------------
def compute_wer(hyp_text: str, ref_text: str) -> Optional[Dict]:
    """
    Compute WER using jiwer with robust normalization and version-safe length fallbacks.
    """

    def _normalize(t: str) -> str:
        # unify curly apostrophes to straight
        t = t.replace("\u2019", "'").replace("\u2018", "'")
        t = t.lower()
        # keep word chars + German/Latin accents + apostrophes; nuke the rest to spaces
        t = re.sub(r"[^\w\säöüßàáâãçèéêëìíîïñòóôõùúûü'’]", " ", t, flags=re.IGNORECASE)
        # collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    try:
        import jiwer  # type: ignore
    except Exception:
        return None

    ref_n = _normalize(ref_text or "")
    hyp_n = _normalize(hyp_text or "")

    if not ref_n:
        raise ValueError(
            "Reference text is empty after normalization. "
            "Check your --ref file shape/keys (expect 'text' or 'segments[*].Text')."
        )
    if not hyp_n:
        raise ValueError(
            "Hypothesis text is empty after normalization. "
            "Check your --hyp file contents."
        )

    measures = jiwer.compute_measures(ref_n, hyp_n)

    # Version-safe length extraction
    ref_len = (
        measures.get("reference_length")
        or measures.get("truth_length")
        or ((measures.get("hits") or 0) + (measures.get("substitutions") or 0) + (measures.get("deletions") or 0))
    )
    hyp_len = (
        measures.get("hypothesis_length")
        or ((measures.get("hits") or 0) + (measures.get("substitutions") or 0) + (measures.get("insertions") or 0))
    )

    return {
        "wer": measures.get("wer"),
        "mer": measures.get("mer"),
        "wil": measures.get("wil"),
        "wip": measures.get("wip"),
        "hits": measures.get("hits"),
        "substitutions": measures.get("substitutions"),
        "deletions": measures.get("deletions"),
        "insertions": measures.get("insertions"),
        "ref_words": ref_len,
        "hyp_words": hyp_len,
    }


# ------------------------------ DER (pyannote) -------------------------------
def _load_rttm_to_annotation(rttm_path: Path):
    """Load RTTM into a pyannote.core.Annotation.
    Tries pyannote.core.io.load_rttm, falls back to a simple parser.
    """
    try:
        from pyannote.core import Annotation, Segment
    except Exception as e:
        raise RuntimeError(f"pyannote.core not available: {e}")

    # First try the modern loader
    try:
        from pyannote.core.io import load_rttm  # type: ignore
        obj = load_rttm(str(rttm_path))
        if not obj:
            raise ValueError(f"Empty RTTM: {rttm_path}")
        _, ann = next(iter(obj.items()))
        if not isinstance(ann, Annotation):
            raise ValueError(f"RTTM did not yield an Annotation for {rttm_path}")
        return ann
    except Exception:
        pass  # fall back to manual parse

    # Minimal manual RTTM parser (supports SPEAKER lines)
    ann = Annotation()
    uri_found = None
    with open(rttm_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9 or parts[0].upper() != "SPEAKER":
                continue
            uri = parts[1]
            try:
                start = float(parts[3])
                dur = float(parts[4])
            except Exception:
                continue
            spk = parts[7]
            if dur <= 0:
                continue
            if uri_found is None:
                uri_found = uri
            # pyannote.core.Annotation doesn’t need URI attached here, we just add segments.
            ann[Segment(start, start + dur)] = spk
    if len(ann) == 0:
        raise ValueError(f"No valid SPEAKER lines parsed from RTTM: {rttm_path}")
    return ann



def compute_der(hyp_rttm: Path, ref_rttm: Path) -> Optional[Dict]:
    """Compute DER using pyannote.metrics, robust to v3+ API changes."""
    try:
        from pyannote.metrics.diarization import DiarizationErrorRate  # type: ignore
        from pyannote.core import Timeline  # type: ignore
    except Exception:
        return None

    hyp = _load_rttm_to_annotation(hyp_rttm)
    ref = _load_rttm_to_annotation(ref_rttm)

    metric = DiarizationErrorRate()

    # Build UEM = union of both annotations' support (prevents the warning and standardizes scoring region)
    try:
        uem: Timeline = ref.get_timeline().union(hyp.get_timeline()).support()  # type: ignore
    except Exception:
        uem = None  # fall back if timeline ops unavailable

    # 1) Get the aggregate DER value directly from the call (works on v3+)
    try:
        der_value = float(metric(ref, hyp, uem=uem) if uem is not None else metric(ref, hyp))
    except Exception:
        der_value = None

    # 2) Try to enrich with a report dataframe when available
    details = None
    try:
        rep = metric.report(display=False).df  # pandas DataFrame
        # row label is often "TOTAL" (upper) in v3; handle case-insensitively
        row = rep.loc[rep.index.astype(str).str.lower() == "total"]
        if not row.empty:
            # If present, prefer the dataframe’s DER (for consistency with its internal rounding)
            if "diarization error rate" in row.columns:
                der_value = float(row.iloc[0]["diarization error rate"])
            # Pull common components when present
            comp_cols = [c for c in ["confusion", "false alarm", "missed detection", "missed speech"]
                         if c in row.columns]
            if comp_cols:
                details = {c: float(row.iloc[0][c]) for c in comp_cols}
    except Exception:
        pass

    if der_value is None:
        return None

    return {"der": der_value, "details": details}



# ------------------------------ Auto-picking --------------------------------
def _auto_paths_by_guid(
    guid: str,
    hyp_dir: Path,
    gt_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    hyp = hyp_dir / f"{guid}_transcript.json"
    ref = None
    if gt_dir.exists():
        candidates = sorted(gt_dir.glob(f"*{guid}*.json"))
        if candidates:
            ref = candidates[0]
    return (hyp if hyp.exists() else None, ref)


def _maybe_autofill_rttm(guid: str) -> Tuple[Optional[Path], Optional[Path]]:
    hyp_rttm = Path("outputs/diarization") / f"{guid}.rttm"
    ref_rttm = Path("data/rttm") / f"{guid}.rttm"
    return (hyp_rttm if hyp_rttm.exists() else None, ref_rttm if ref_rttm.exists() else None)


# ---------------------------------- CLI --------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate outputs (WER/DER) with optional auto mode.")
    # Auto mode
    p.add_argument("--guid", help="GUID to auto-pick files (like _wer60.py), full-length by default.")
    p.add_argument("--gt-dir", default="data/json", help="Where to search reference JSON for --guid (default: data/json)")
    p.add_argument("--hyp-dir", default="outputs/merged", help="Where to read hypothesis for --guid (default: outputs/merged)")
    # Optional slicing (OFF by default)
    p.add_argument("--first-seconds", type=float, default=None, help="If set, slice both hyp & ref to [tmin, first-seconds). If omitted, evaluate the whole transcript.")
    p.add_argument("--tmin", type=float, default=0.0, help="Lower bound for slicing (seconds). Only used if --first-seconds is provided.")
    # Manual paths
    p.add_argument("--hyp", required=False, help="Hypothesis merged transcript (.json or .csv)")
    p.add_argument("--ref", required=False, help="Reference transcript (.txt/.json/.csv)")
    p.add_argument("--ref-key", required=False, help="Top-level key in JSON holding the full reference text")
    p.add_argument("--ref-segments-key", required=False, help="Key in JSON that holds a list of segment dicts")
    p.add_argument("--ref-text-field", required=False, help="Field name inside each segment for text (default autodetect incl. 'Text')")
    # DER
    p.add_argument("--hyp-rttm", required=False, help="Hypothesis diarization RTTM (from outputs/diarization)")
    p.add_argument("--ref-rttm", required=False, help="Reference diarization RTTM")
    return p.parse_args()


def main():
    args = parse_args()

    # Auto: infer files from GUID
    hyp_path: Optional[Path] = Path(args.hyp) if args.hyp else None
    ref_path: Optional[Path] = Path(args.ref) if args.ref else None

    if args.guid:
        gt_dir = Path(args.gt_dir)
        hyp_dir = Path(args.hyp_dir)
        auto_hyp, auto_ref = _auto_paths_by_guid(args.guid, hyp_dir=hyp_dir, gt_dir=gt_dir)
        if hyp_path is None and auto_hyp is not None:
            hyp_path = auto_hyp
        if ref_path is None and auto_ref is not None:
            ref_path = auto_ref
        # NOTE: No default slicing here—full transcript unless user passes --first-seconds.
        if not args.hyp_rttm or not args.ref_rttm:
            auto_hyp_rttm, auto_ref_rttm = _maybe_autofill_rttm(args.guid)
            if not args.hyp_rttm and auto_hyp_rttm:
                args.hyp_rttm = str(auto_hyp_rttm)
            if not args.ref_rttm and auto_ref_rttm:
                args.ref_rttm = str(auto_ref_rttm)

    if hyp_path is None:
        raise SystemExit("No hypothesis provided. Supply --hyp or use --guid.")

    # ---- load hypothesis segments ----
    segments = _load_segments(hyp_path)

    # Optional time slice (only if --first-seconds is given or tmin>0 with it)
    if args.first_seconds is not None or (args.tmin > 0.0 and args.first_seconds is not None):
        segments = _slice_segments_by_time(
            segments,
            tmin=float(args.tmin),
            tmax=float(args.first_seconds) if args.first_seconds is not None else None
        )
    hyp_text = _segments_to_text(segments)

    def _preview(label: str, text: str):
        snippet = text[:200].replace("\n", " ")
        print(f"{label}: {len(text)} chars")
        if snippet:
            print(f"   {snippet}{'…' if len(text) > 200 else ''}")

    print("\n📄 Loaded hypothesis text")
    print(f"   Hyp path: {hyp_path}")
    _preview("   Hyp", hyp_text)

    # --- WER ---
    if ref_path:
        ref_suf = Path(ref_path).suffix.lower()
        if ref_suf in (".json", ".csv"):
            ref_segments = _load_segments(Path(ref_path))
            if args.first_seconds is not None or (args.tmin > 0.0 and args.first_seconds is not None):
                ref_segments = _slice_segments_by_time(
                    ref_segments,
                    tmin=float(args.tmin),
                    tmax=float(args.first_seconds) if args.first_seconds is not None else None
                )
            ref_text = _segments_to_text(ref_segments, field_priority=[args.ref_text_field] if args.ref_text_field else None)
        else:
            ref_text = _load_reference_text(
                Path(ref_path),
                ref_key=args.ref_key,
                ref_segments_key=args.ref_segments_key,
                ref_text_field=args.ref_text_field,
            )

        print("\n📄 Loaded reference text")
        print(f"   Ref path: {ref_path}")
        _preview("   Ref", ref_text)

        try:
            wer_pack = compute_wer(hyp_text, ref_text)
        except Exception as e:
            print(f"\n⚠️  WER computation aborted: {e}")
            print("   Tips:")
            print("   • If your JSON uses 'Text' (capitalized), autodetect should handle it; otherwise pass --ref-text-field Text.")
            print("   • Ensure the reference actually contains content (not just metadata).")
            return

        if wer_pack is None:
            print("⚠️  jiwer not installed or incompatible. Skipping WER. (pip install jiwer)")
        else:
            print("\n🧪 WER Results")
            print(f"   Hyp: {hyp_path}")
            print(f"   Ref: {ref_path}")
            if args.first_seconds is not None:
                print(f"   Window: [{args.tmin:.1f}s, {args.first_seconds:.1f}s)")
            print(f"   WER: {wer_pack['wer']:.3f} | MER: {wer_pack['mer']:.3f} | WIL: {wer_pack['wil']:.3f} | WIP: {wer_pack['wip']:.3f}")
            print(f"   Ref words: {wer_pack['ref_words']} | Hyp words: {wer_pack['hyp_words']}")
            print(f"   S:{wer_pack['substitutions']} D:{wer_pack['deletions']} I:{wer_pack['insertions']} H:{wer_pack['hits']}")
            # Percent breakdown (relative to reference length), when available
            N = wer_pack.get("ref_words")
            if isinstance(N, (int, float)) and N:
                s = wer_pack.get("substitutions", 0) or 0
                d = wer_pack.get("deletions", 0) or 0
                i = wer_pack.get("insertions", 0) or 0
                try:
                    print(f"   Rates: S={s/N:.3%} D={d/N:.3%} I={i/N:.3%}")
                except Exception:
                    pass
    else:
        if args.guid:
            print("\nℹ️  Auto mode: no reference located in data/json for this GUID; skipping WER.")
        else:
            print("\nℹ️  No --ref provided; skipping WER.")

    # --- DER ---
    if args.hyp_rttm and args.ref_rttm:
        try:
            der_pack = compute_der(Path(args.hyp_rttm), Path(args.ref_rttm))
        except Exception as e:
            der_pack = None
            print(f"⚠️  DER computation failed: {e}")

        if der_pack is None:
            print("⚠️  pyannote.metrics/core not installed or RTTM load failed. Skipping DER. (pip install pyannote.metrics pyannote.core)")
        else:
            print("\n🧪 DER Results")
            print(f"   Hyp RTTM: {args.hyp_rttm}")
            print(f"   Ref RTTM: {args.ref_rttm}")
            print(f"   DER: {der_pack['der']:.3f}")
            if isinstance(der_pack.get("details"), dict):
                d = der_pack["details"]
                parts = []
                for key in ["confusion", "false alarm", "missed speech", "missed detection"]:
                    if key in d:
                        parts.append(f"{key}={d[key]:.3f}")
                if parts:
                    print("   " + " | ".join(parts))
    else:
        print("\nℹ️  No --hyp-rttm and/or --ref-rttm provided; skipping DER.")

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
