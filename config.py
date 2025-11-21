"""
Configuration file for Phase 1 ASR Project
Contains all paths, constants, and configuration settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# BASE DIRECTORIES
# ============================================================================
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
VIDEO_DIR = DATA_DIR / "video"
JSON_DIR = DATA_DIR / "json"

# Output directories
OUTPUTS_DIR = BASE_DIR / "outputs"
TRANSCRIPTIONS_DIR = OUTPUTS_DIR / "transcriptions"
DIARIZATION_DIR = OUTPUTS_DIR / "diarization"
MERGED_DIR = OUTPUTS_DIR / "merged"
REPORTS_DIR = OUTPUTS_DIR / "reports"
CHUNKS_DIR = OUTPUTS_DIR / "chunks"

# Model and log directories
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
TEMP_DIR = BASE_DIR / "temp"

# Create all directories
DIRS_TO_CREATE = [
    DATA_DIR, AUDIO_DIR, VIDEO_DIR, JSON_DIR,
    OUTPUTS_DIR, TRANSCRIPTIONS_DIR, DIARIZATION_DIR, 
    MERGED_DIR, REPORTS_DIR, CHUNKS_DIR,
    MODELS_DIR, LOGS_DIR, TEMP_DIR
]
for directory in DIRS_TO_CREATE:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# API KEYS & AUTHENTICATION
# ============================================================================
HF_TOKEN = os.getenv("HF_TOKEN", None)

if not HF_TOKEN:
    print("⚠️  Warning: HF_TOKEN not found in .env file")
    print("   You'll need to provide it when running diarization")
    print("   Get your token from: https://huggingface.co/settings/tokens")

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================
# Whisper model options: tiny, base, small, medium, large, large-v2, large-v3
WHISPER_MODEL_SIZE = "small"

# Pyannote diarization model
PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
PYANNOTE_SEGMENTATION_MODEL = "pyannote/segmentation-3.0"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4 if torch.cuda.is_available() else 2

# ============================================================================
# AUDIO PROCESSING SETTINGS
# ============================================================================
SAMPLE_RATE = 16000  # 16kHz is optimal for Whisper and Pyannote
CHANNELS = 1         # Mono audio

# Chunking settings for long audio files
MAX_CHUNK_DURATION = 600  # seconds (10 minutes)
CHUNK_OVERLAP = 30        # seconds overlap between chunks

# ============================================================================
# FILE FORMATS
# ============================================================================
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus']
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS + VIDEO_EXTENSIONS

# ============================================================================
# PROCESSING OPTIONS
# ============================================================================
# Whisper options
WHISPER_LANGUAGE = None       # Auto-detect by default, can set to 'de', 'en', etc.
WHISPER_TASK = "transcribe"   # or "translate" for translation to English
WHISPER_VERBOSE = False

# Diarization options
MIN_SPEAKERS = None           # None for auto-detection
MAX_SPEAKERS = None           # None for auto-detection
MIN_SEGMENT_DURATION = 0.5    # Minimum segment duration in seconds

# === Multilingual + per-segment controls (Phase-1) ===
# If True → diarize → slice each diarized segment → transcribe with language auto-detect per slice.
# If False → transcribe whole file once and map ASR segments to speakers by overlap (previous behavior).
PER_SEGMENT_MODE: bool = False

# Merge very short consecutive turns from the same speaker before slicing/transcribing (seconds)
MIN_SEG_DUR: float = 1.0

# Padding (seconds) added on both sides when slicing audio for a diarization segment
PAD_SEC: float = 0.2

# Language strategy:
#   "per_segment"  → auto-detect per segment (default)
#   "global_auto"  → auto-detect in each call but treat as a single-language workflow
#   "forced:<iso>" → force a specific ISO code per segment (e.g., "forced:en", "forced:ur")
LANGUAGE_MODE: str = "forced:de"

# ============================================================================
# OUTPUT FORMATS
# ============================================================================
OUTPUT_FORMATS = ["json", "csv"]  # Supported output formats
SAVE_RTTM = True                  # Save diarization in RTTM format
SAVE_INTERMEDIATE = True          # Save intermediate results

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "asr_pipeline.log"

# ============================================================================
# BATCH PROCESSING
# ============================================================================
BATCH_SIZE = 1             # Number of files to process in parallel
MAX_FILES_PER_BATCH = 10   # Maximum files to process in one batch run

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================
# Memory management
CLEAR_CACHE_AFTER_PROCESSING = True  # Clear GPU cache after each file
LOW_MEMORY_MODE = False              # Use smaller chunks if True

# Speed settings
WHISPER_FP16 = True if DEVICE == "cuda" else False  # Use half precision on GPU
BATCH_INFERENCE = False                             # Experimental: batch inference

# ============================================================================
# EVALUATION METRICS
# ============================================================================
COMPUTE_WER = True  # Compute Word Error Rate (requires reference)
COMPUTE_DER = True  # Compute Diarization Error Rate (requires reference)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_device_info():
    """Get device information"""
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "name": torch.cuda.get_device_name(0),
            "memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        }
    return {"device": "cpu", "name": "CPU", "memory": "N/A"}

def print_config():
    """Print current configuration"""
    print("\n" + "="*70)
    print(" "*20 + "CONFIGURATION")
    print("="*70)
    
    print(f"\n📁 Directories:")
    print(f"   Base: {BASE_DIR}")
    print(f"   Audio Input: {AUDIO_DIR}")
    print(f"   Video Input: {VIDEO_DIR}")
    print(f"   Outputs: {OUTPUTS_DIR}")
    print(f"   Models: {MODELS_DIR}")
    print(f"   Logs: {LOGS_DIR}")
    
    print(f"\n🤖 Models:")
    print(f"   Whisper: {WHISPER_MODEL_SIZE}")
    print(f"   Diarization: {PYANNOTE_DIARIZATION_MODEL}")
    
    device_info = get_device_info()
    print(f"\n💻 Device:")
    print(f"   Type: {device_info['device'].upper()}")
    print(f"   Name: {device_info['name']}")
    print(f"   Memory: {device_info['memory']}")
    
    print(f"\n🎵 Audio Settings:")
    print(f"   Sample Rate: {SAMPLE_RATE} Hz")
    print(f"   Channels: {CHANNELS} (Mono)")
    print(f"   Max Chunk: {MAX_CHUNK_DURATION}s")
    print(f"   Chunk Overlap: {CHUNK_OVERLAP}s")
    
    print(f"\n⚙️  Processing:")
    print(f"   Language (legacy global): {WHISPER_LANGUAGE or 'Auto-detect'}")
    print(f"   Per-Segment Mode: {PER_SEGMENT_MODE}")
    print(f"   Min Seg Dur (merge): {MIN_SEG_DUR}s")
    print(f"   Slice Padding: {PAD_SEC}s")
    print(f"   Language Mode: {LANGUAGE_MODE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   FP16: {WHISPER_FP16}")
    print(f"   Low Memory Mode: {LOW_MEMORY_MODE}")
    
    print(f"\n📊 Output:")
    print(f"   Formats: {', '.join(OUTPUT_FORMATS)}")
    print(f"   Save RTTM: {SAVE_RTTM}")
    print(f"   Save Intermediate: {SAVE_INTERMEDIATE}")
    
    print("\n" + "="*70 + "\n")

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    """Validate configuration settings"""
    issues = []
    
    # Check if audio/video directories exist
    if not AUDIO_DIR.exists():
        issues.append(f"Audio directory not found: {AUDIO_DIR}")
    if not VIDEO_DIR.exists():
        issues.append(f"Video directory not found: {VIDEO_DIR}")
    
    # Check HuggingFace token for diarization
    if not HF_TOKEN:
        issues.append("HF_TOKEN not set - diarization will require manual authentication")
    
    # Check model size
    valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    if WHISPER_MODEL_SIZE not in valid_models:
        issues.append(f"Invalid Whisper model: {WHISPER_MODEL_SIZE}")
    
    # Check chunk settings
    if MAX_CHUNK_DURATION < 60:
        issues.append(f"MAX_CHUNK_DURATION too small: {MAX_CHUNK_DURATION}s (min 60s)")
    if CHUNK_OVERLAP >= MAX_CHUNK_DURATION:
        issues.append("CHUNK_OVERLAP must be less than MAX_CHUNK_DURATION")

    # LANGUAGE_MODE sanity
    if not (LANGUAGE_MODE == "per_segment" or LANGUAGE_MODE == "global_auto" or LANGUAGE_MODE.startswith("forced:")):
        issues.append(f"Invalid LANGUAGE_MODE: {LANGUAGE_MODE}")

    if issues:
        print("\n⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    
    return True

# Run validation on import
if __name__ != "__main__":
    validate_config()

