# Phase‑1 — Multilingual ASR with Advanced Speaker Diarization

This repository builds an **end‑to‑end pipeline** that:
- extracts or loads audio (from audio/video),
- performs **speaker diarization**,
- transcribes each speaker turn with **per‑segment multilingual detection** (or forced language),
- merges everything into clean, time‑aligned JSON/CSV,
- and optionally evaluates **WER**/**DER**.

**Status:** Phase‑1 complete with off‑the‑shelf models — no custom training required.

---

## 0) Prereqs & Python Version (very important)

> We run on **Python 3.11.9** (Windows/macOS/Linux). Please use exactly this minor version to avoid version pinning issues.

### Create and activate a virtual environment (Windows PowerShell)
```powershell
py -3.11 -m venv venv
.venv\Scripts\activate
python --version  # should print 3.11.9
python -m pip install --upgrade pip
```

> If you're on macOS/Linux, use `python3.11 -m venv venv && source venv/bin/activate` accordingly.

---

## 1) Install Dependencies

> ⚠️ **GPU users (recommended):** Install a matching Torch build first. Example for CUDA 12.1:
```powershell
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```
> (For CPU‑only: `pip install torch==2.1.0 torchaudio==2.1.0` without the index URL.)

Then install the rest:
```powershell
pip install -r requirements.txt
```

### Hugging Face token (for pyannote)
Create a `.env` file in the project root:
```
HF_TOKEN=hf_********************************
```

---

## 2) Key Configuration

Edit `config.py` for the behavior you want:

- `PER_SEGMENT_MODE = True`  
  - `True`: **Diarize → slice per speaker → transcribe each slice** (recommended)
  - `False`: Transcribe once globally → assign segments by overlap

- `LANGUAGE_MODE`  
  - `"per_segment"` → auto‑detect per slice (multilingual ready)  
  - `"global_auto"` → one language for whole file (auto or `--language` hint)  
  - `"forced:de"` (example) → force German for all slices

- `MIN_SEG_DUR` (e.g., `0.8`) — merges very short turns before slicing  
- `PAD_SEC` (e.g., `0.15`) — padding added around each slice to avoid truncation

**Paths**
- Inputs: `data/audio/`, `data/video/`  
- Outputs:
  - `outputs/transcriptions/`
  - `outputs/diarization/`
  - `outputs/merged/`
  - `outputs/reports/`

---

## 3) Run the Pipeline

### Single file (audio or video)
```powershell
python main.py -i data\audio\meeting.wav

# or pass a video (audio auto‑extracted to WAV first)
python main.py -i data\audio\call.mp4
```

### Directory (batch)
```powershell
python main.py -i data\audio```

**Useful flags**
```powershell
# Hint a global language when not using per‑segment mode
python main.py -i data\audio\file.wav --language de

# Hide the config printout
python main.py -i data\audio\file.wav --no-print-config
```

> Example you used successfully:
> ```powershell
> python main.py -i data\audio\ --language de --no-print-config
> ```

---

## 4) Outputs

- **Merged transcript** (one row per speaker turn):  
  `outputs/merged/<GUID>_transcript.json|csv`  
  Fields: `start, end, duration, speaker, language, text, (confidence?)`

- **Intermediates** (if enabled):  
  - ASR raw JSON → `outputs/transcriptions/<GUID>_transcription.json`  
  - Diarization JSON/RTTM → `outputs/diarization/<GUID>.json|.rttm`

- **Batch run report** (one JSON for the entire directory run):  
  `outputs/reports/batch_report_YYYYMMDD_HHMMSS.json`  
  Each item includes `success`, `audio_path`, `audio_duration`, `processing_time`, `num_segments`, `num_speakers`, and output paths.  
  *(Matches the report you shared.)*

---

## 5) Evaluation (WER / DER)

We provide `evaluation.py` with an **auto‑mode** that evaluates the **entire file** by default.

### Quick auto‑mode (recommended)
```
python evaluation.py --guid <GUID>
```
- Hypothesis is read from `outputs/merged/<GUID>_transcript.json`.
- Reference JSON is auto‑discovered under `data/json/*<GUID>*.json`.
- JSON with your original shape (`Text, StartTime, EndTime`) is supported.
- If diarization RTTMs exist (`outputs/diarization/<GUID>.rttm` and `data/rttm/<GUID>.rttm`), DER is computed automatically.

### Manual paths
```powershell
python evaluation.py --hyp outputs\merged\<GUID>_transcript.json --ref data\json\<GUID>.json
# with diarization
python evaluation.py --hyp outputs\merged\<GUID>_transcript.json --ref data\json\<GUID>.json ^
  --hyp-rttm outputs\diarization\<GUID>.rttm --ref-rttm data
ttm\<GUID>.rttm
```

### Optional window (if you *want* to slice, e.g., first 120s)
```powershell
python evaluation.py --guid <GUID> --first-seconds 120
```

> If `jiwer` or `pyannote.metrics` aren’t installed, that metric is skipped with a helpful message.

---

## 6) What “multilingual per‑segment” means

1) **Diarize** the full audio → create speaker turns.  
2) **Merge tiny turns** (`MIN_SEG_DUR`) to stabilize boundaries.  
3) For **each speaker turn**:  
   - Slice audio with **padding** (`PAD_SEC`),  
   - **Auto‑detect language** per slice (or force one),  
   - Transcribe slice → emit **one row**: `start, end, speaker, language, text`.

This is robust for mixed‑language meetings.

---

## 7) Troubleshooting

- **pyannote login** → ensure `HF_TOKEN` present in `.env`.  
- **Videos** → `moviepy` needs `ffmpeg` in your PATH.  
- **Low‑VRAM GPUs** → set `WHISPER_MODEL_SIZE="small"` or `"base"` in `config.py`.  
- **Choppy outputs** → increase `MIN_SEG_DUR` (e.g., `1.2`) or `PAD_SEC` (`0.20`).  
- **Windows paths** → use `\` in PowerShell or escape appropriately.

---

## 8) Phase‑1 Deliverables Checklist

- ✅ Time‑aligned transcript with speaker labels  
- ✅ Multilingual per‑segment handling  
- ✅ Diarization artifacts (`.json` + optional `.rttm`)  
- ✅ Evaluation scripts (WER/DER) + ground‑truth converter  
- ✅ Clear CLI + configs + batch report  
- ✅ **No custom model training required** for Phase‑1 (baseline established)
