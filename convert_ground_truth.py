"""
convert_ground_truth.py
Utilities to convert human/ground-truth transcripts into clean references.

Features
- Accepts: .txt, .pdf, .csv, .json
- Produces: a cleaned reference .txt (for WER)
- Optional: writes a reference .rttm if you have start/end/speaker columns (for DER)

Examples
# 1) Make a clean text reference from a PDF:
python convert_ground_truth.py --in data/json/actual_transcription.pdf --out ref/actual.txt

# 2) From a CSV with 'text' column:
python convert_ground_truth.py --in outputs/merged/file_transcript.csv --text-col text --out ref/actual.txt

# 3) From a JSON list of segments:
python convert_ground_truth.py --in outputs/merged/file_transcript.json --json-segments --out ref/actual.txt

# 4) Also emit RTTM from CSV/JSON that include start/end/speaker:
python convert_ground_truth.py --in outputs/merged/file_transcript.csv --text-col text \
  --start-col start --end-col end --spk-col speaker --rttm-out ref/actual.rttm --uri file1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd


# ------------------------------ Loaders --------------------------------------
def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    """
    Try PyPDF2 first, fallback to pdfminer.six if available.
    """
    try:
        import PyPDF2  # type: ignore
        text_parts: List[str] = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)
    except Exception:
        # fallback
        try:
            from pdfminer.high_level import extract_text  # type: ignore
            return extract_text(str(path)) or ""
        except Exception as e:
            raise RuntimeError(f"Cannot read PDF (need PyPDF2 or pdfminer.six): {e}")


def _read_csv_to_rows(path: Path) -> List[Dict]:
    df = pd.read_csv(path, encoding="utf-8")
    return df.to_dict("records")


def _read_json_to_rows_or_text(path: Path, json_segments: bool) -> List[Dict] | str:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if json_segments and "segments" in obj and isinstance(obj["segments"], list):
            return obj["segments"]
        # If it has a 'text' field
        if "text" in obj and isinstance(obj["text"], str):
            return obj["text"]
    # Best effort: if dict but no recognized keys, stringify
    return json.dumps(obj, ensure_ascii=False)


# ------------------------------ Cleaning -------------------------------------
_TS_PATTERN = re.compile(r"\[\s*\d{1,2}:\d{2}(?::\d{2})?\s*-\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]\s*")
_SPK_PATTERN = re.compile(r"^(?:SPEAKER_\d{2,}|Speaker\s*\d+|Spk\d+|SPK\d+)\s*[:\-]\s*", flags=re.IGNORECASE)

def _strip_timestamps_and_speakers(line: str) -> str:
    # remove [00:00 - 00:05] like blocks
    line = _TS_PATTERN.sub("", line)
    # remove leading "SPEAKER_00:" style prefixes
    line = _SPK_PATTERN.sub("", line)
    return line.strip()


def clean_reference_text(raw: str, aggressive: bool = False) -> str:
    """
    Remove timestamps/speaker prefixes and normalize whitespace.
    If aggressive=True, also strip stray brackets and collapse punctuation spacing.
    """
    # Split and clean line-by-line
    lines = [l.strip() for l in raw.splitlines()]
    lines = [_strip_timestamps_and_speakers(l) for l in lines]
    text = " ".join([l for l in lines if l])

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if aggressive:
        # Remove stray [] and () wrappers commonly left by exports
        text = re.sub(r"[\[\]\(\)]", "", text)
        # Collapse punctuation spaces
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text


# ------------------------------ Building refs --------------------------------
def rows_to_text(
    rows: List[Dict],
    text_col: str = "text",
    join_with: str = " "
) -> str:
    parts: List[str] = []
    for r in rows:
        t = (r.get(text_col) or "").strip()
        if t:
            # Remove inline timestamps/speaker tags if present
            t = _strip_timestamps_and_speakers(t)
            parts.append(t)
    return join_with.join(parts).strip()


def rows_to_rttm(
    rows: List[Dict],
    start_col: str = "start",
    end_col: str = "end",
    spk_col: str = "speaker",
    uri: str = "file"
) -> str:
    """
    Convert rows with start/end/speaker into an RTTM string.
    """
    out_lines: List[str] = []
    for r in rows:
        try:
            s = float(r[start_col])
            e = float(r[end_col])
            if e <= s:
                continue
            dur = e - s
            spk = str(r.get(spk_col, "SPEAKER_00"))
            # Standard RTTM SPEAKER line:
            # SPEAKER <uri> 1 <start> <duration> <ortho> <stype> <name> <conf>
            out_lines.append(
                f"SPEAKER {uri} 1 {s:.3f} {dur:.3f} <NA> <NA> {spk} <NA>"
            )
        except Exception:
            # Skip bad rows silently
            continue
    return "\n".join(out_lines) + ("\n" if out_lines else "")


# ----------------------------------- CLI -------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Convert human/ground-truth transcript to clean reference.")
    p.add_argument("--in", dest="inp", required=True, help="Input file (.txt/.pdf/.csv/.json)")
    p.add_argument("--out", dest="out_txt", required=True, help="Output cleaned reference .txt")
    p.add_argument("--aggressive", action="store_true", help="Aggressive cleaning (remove brackets, tighten punctuation)")

    # CSV/JSON options
    p.add_argument("--text-col", default="text", help="Column/key name containing text (CSV/JSON)")
    p.add_argument("--json-segments", action="store_true", help="When JSON has {'segments': [...]}, read that list")

    # Optional RTTM generation
    p.add_argument("--rttm-out", default=None, help="Optional RTTM output path")
    p.add_argument("--start-col", default="start", help="Column/key for start time")
    p.add_argument("--end-col", default="end", help="Column/key for end time")
    p.add_argument("--spk-col", default="speaker", help="Column/key for speaker label")
    p.add_argument("--uri", default="file", help="URI field for RTTM lines")

    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.inp)
    out_txt = Path(args.out_txt)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    # --- Load ---
    suf = in_path.suffix.lower()
    rows: Optional[List[Dict]] = None

    if suf == ".txt":
        raw = _read_txt(in_path)
        ref_text = clean_reference_text(raw, aggressive=args.aggressive)

    elif suf == ".pdf":
        raw = _read_pdf(in_path)
        ref_text = clean_reference_text(raw, aggressive=args.aggressive)

    elif suf == ".csv":
        rows = _read_csv_to_rows(in_path)
        ref_text = clean_reference_text(rows_to_text(rows, text_col=args.text_col), aggressive=args.aggressive)

    elif suf == ".json":
        obj = _read_json_to_rows_or_text(in_path, json_segments=args.json_segments)
        if isinstance(obj, str):
            ref_text = clean_reference_text(obj, aggressive=args.aggressive)
        else:
            rows = obj  # list[dict]
            ref_text = clean_reference_text(rows_to_text(rows, text_col=args.text_col), aggressive=args.aggressive)

    else:
        raise ValueError(f"Unsupported input type: {suf}")

    # --- Save cleaned text ---
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(ref_text, encoding="utf-8")
    print(f"✅ Wrote reference text: {out_txt}")

    # --- Optional RTTM ---
    if args.rttm_out:
        if rows is None:
            print("ℹ️  RTTM requested, but no tabular rows available (need CSV/JSON with start/end/speaker). Skipping RTTM.")
        else:
            rttm_str = rows_to_rttm(
                rows,
                start_col=args.start_col,
                end_col=args.end_col,
                spk_col=args.spk_col,
                uri=args.uri,
            )
            rttm_path = Path(args.rttm_out)
            rttm_path.parent.mkdir(parents=True, exist_ok=True)
            rttm_path.write_text(rttm_str, encoding="utf-8")
            print(f"✅ Wrote reference RTTM: {rttm_path}")


if __name__ == "__main__":
    main()
