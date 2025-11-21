"""
Utility functions for the ASR pipeline
Contains helpers for file management, logging, and data processing
"""

import os
import glob
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import csv
import pandas as pd

from config import (
    AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, LOGS_DIR,
    LOG_FORMAT, LOG_LEVEL
)

# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logger(name: str = "asr_pipeline", log_file: str = None) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = LOGS_DIR / f"asr_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, LOG_LEVEL))
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

# ============================================================================
# FILE OPERATIONS
# ============================================================================
def scan_media_files(directory: Path) -> Tuple[List[str], List[str]]:
    """
    Scan directory for audio and video files
    
    Args:
        directory: Directory to scan
    
    Returns:
        Tuple of (audio_files, video_files) lists
    """
    audio_files = []
    video_files = []
    
    # Scan for audio files
    for ext in AUDIO_EXTENSIONS:
        pattern = os.path.join(directory, f"*{ext}")
        audio_files.extend(glob.glob(pattern))
    
    # Scan for video files
    for ext in VIDEO_EXTENSIONS:
        pattern = os.path.join(directory, f"*{ext}")
        video_files.extend(glob.glob(pattern))
    
    return sorted(audio_files), sorted(video_files)

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get file information
    
    Args:
        file_path: Path to file
    
    Returns:
        Dictionary with file information
    """
    stat = os.stat(file_path)
    return {
        "name": os.path.basename(file_path),
        "path": file_path,
        "size_mb": stat.st_size / (1024 * 1024),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "extension": Path(file_path).suffix
    }

def print_file_summary(audio_files: List[str], video_files: List[str], 
                       max_display: int = 5) -> None:
    """
    Print summary of found files
    
    Args:
        audio_files: List of audio file paths
        video_files: List of video file paths
        max_display: Maximum number of files to display per category
    """
    print(f"\n📊 Files Found:")
    print(f"   Audio files: {len(audio_files)}")
    print(f"   Video files: {len(video_files)}")
    print(f"   Total: {len(audio_files) + len(video_files)}")
    
    if audio_files:
        print(f"\n🎵 Sample Audio Files:")
        for i, f in enumerate(audio_files[:max_display], 1):
            info = get_file_info(f)
            print(f"   {i}. {info['name']} ({info['size_mb']:.1f} MB)")
        if len(audio_files) > max_display:
            print(f"   ... and {len(audio_files) - max_display} more")
    
    if video_files:
        print(f"\n🎬 Sample Video Files:")
        for i, f in enumerate(video_files[:max_display], 1):
            info = get_file_info(f)
            print(f"   {i}. {info['name']} ({info['size_mb']:.1f} MB)")
        if len(video_files) > max_display:
            print(f"   ... and {len(video_files) - max_display} more")
    
    if not audio_files and not video_files:
        print("\n⚠️  No audio or video files found")

# ============================================================================
# JSON & CSV OPERATIONS
# ============================================================================
def save_json(data: Any, output_path: Path, indent: int = 2) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        output_path: Output file path
        indent: JSON indentation
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def load_json(file_path: Path) -> Any:
    """
    Load data from JSON file
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Loaded data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_csv(
    rows: List[Dict],
    path: Path,
    fieldnames: Optional[List[str]] = None,
    encoding: str = "utf-8"
) -> None:
    """
    Save a list of dicts to CSV with a stable, client-friendly column order.
    If 'fieldnames' is not provided, we prefer a default transcript schema
    and then append any extra keys found in rows.

    Default order aims to match Phase-1 outputs:
      start, end, duration, speaker, language, text, confidence
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Preferred default order for Phase-1
    default_order = ["start", "end", "duration", "speaker", "language", "text", "confidence"]

    # If no fieldnames provided, build them:
    if fieldnames is None:
        # Union of keys across rows (preserves appearance order somewhat)
        seen = []
        for r in rows or []:
            for k in r.keys():
                if k not in seen:
                    seen.append(k)

        # Start with default order if keys exist in the data
        fieldnames = [k for k in default_order if k in seen]
        # Then append any remaining keys
        fieldnames += [k for k in seen if k not in fieldnames]

        # Fallback when there are no rows
        if not fieldnames:
            fieldnames = default_order

    with path.open("w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            # Ensure all fields exist (use empty string for missing)
            clean = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(clean)

def load_csv(file_path: Path) -> List[Dict]:
    """
    Load data from CSV file
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        List of dictionaries
    """
    df = pd.read_csv(file_path, encoding='utf-8')
    return df.to_dict('records')

# ============================================================================
# TIME & DURATION UTILITIES
# ============================================================================
def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)

def format_timestamp(seconds: float) -> str:
    """
    Format timestamp for display
    
    Args:
        seconds: Timestamp in seconds
    
    Returns:
        Formatted string (e.g., "01:23:45")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# ============================================================================
# PROGRESS TRACKING
# ============================================================================
class ProgressTracker:
    """Simple progress tracker for batch processing"""
    
    def __init__(self, total: int, task_name: str = "Processing"):
        self.total = total
        self.current = 0
        self.task_name = task_name
        self.start_time = datetime.now()
        self.logger = setup_logger("progress")
    
    def update(self, increment: int = 1):
        """Update progress"""
        self.current += increment
        percentage = (self.current / self.total) * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.current > 0:
            eta_seconds = (elapsed / self.current) * (self.total - self.current)
            eta = format_duration(eta_seconds)
        else:
            eta = "N/A"
        
        self.logger.info(
            f"{self.task_name}: {self.current}/{self.total} "
            f"({percentage:.1f}%) - ETA: {eta}"
        )
    
    def complete(self):
        """Mark as complete"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            f"{self.task_name} complete! "
            f"Total time: {format_duration(elapsed)}"
        )

# ============================================================================
# DATA VALIDATION
# ============================================================================
def validate_audio_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate audio file
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    if os.path.getsize(file_path) == 0:
        return False, "File is empty"
    
    ext = Path(file_path).suffix.lower()
    if ext not in AUDIO_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
        return False, f"Unsupported file format: {ext}"
    
    return True, ""

# ============================================================================
# TRANSCRIPT FORMATTING
# ============================================================================
def format_transcript_text(segments: List[Dict], 
                          include_timestamps: bool = True,
                          include_speakers: bool = True) -> str:
    """
    Format transcript segments into readable text
    
    Args:
        segments: List of transcript segments
        include_timestamps: Include timestamps in output
        include_speakers: Include speaker labels in output
    
    Returns:
        Formatted transcript text
    """
    lines = []
    
    for seg in segments:
        parts = []
        
        if include_timestamps:
            start = format_timestamp(seg['start'])
            end = format_timestamp(seg['end'])
            parts.append(f"[{start} - {end}]")
        
        if include_speakers and 'speaker' in seg:
            parts.append(f"{seg['speaker']}:")
        
        parts.append(seg['text'].strip())
        
        lines.append(" ".join(parts))
    
    return "\n".join(lines)

def print_sample_transcript(segments: List[Dict], num_samples: int = 5) -> None:
    """
    Print sample transcript segments
    
    Args:
        segments: List of transcript segments
        num_samples: Number of segments to display
    """
    print(f"\n📄 Sample Transcript (first {num_samples} segments):")
    
    for seg in segments[:num_samples]:
        start = format_timestamp(seg['start'])
        end = format_timestamp(seg['end'])
        speaker = seg.get('speaker', 'Unknown')
        text = seg['text'].strip()
        
        print(f"   [{start} - {end}] {speaker}: {text}")
    
    if len(segments) > num_samples:
        print(f"   ... and {len(segments) - num_samples} more segments")

# ============================================================================
# SPEAKER STATISTICS
# ============================================================================
def compute_speaker_statistics(segments: List[Dict]) -> Dict[str, Any]:
    """
    Compute speaker statistics from segments
    
    Args:
        segments: List of segments with speaker labels
    
    Returns:
        Dictionary with speaker statistics
    """
    df = pd.DataFrame(segments)
    
    if 'speaker' not in df.columns:
        return {}
    
    # Count segments per speaker
    speaker_counts = df['speaker'].value_counts().to_dict()
    
    # Calculate speaking time per speaker
    df['duration'] = df['end'] - df['start']
    speaker_time = df.groupby('speaker')['duration'].sum().to_dict()
    
    # Total speaking time
    total_time = df['duration'].sum()
    
    # Speaking percentage
    speaker_percentage = {
        speaker: (time / total_time * 100) if total_time > 0 else 0
        for speaker, time in speaker_time.items()
    }
    
    return {
        "num_speakers": len(speaker_counts),
        "segment_counts": speaker_counts,
        "speaking_time": speaker_time,
        "speaking_percentage": speaker_percentage,
        "total_time": total_time
    }

def print_speaker_statistics(stats: Dict[str, Any]) -> None:
    """
    Print speaker statistics
    
    Args:
        stats: Dictionary with speaker statistics
    """
    if not stats:
        return
    
    print(f"\n👥 Speaker Statistics:")
    print(f"   Unique speakers: {stats['num_speakers']}")
    print(f"   Total duration: {format_duration(stats['total_time'])}")
    
    print(f"\n   Speaker breakdown:")
    for speaker in sorted(stats['segment_counts'].keys()):
        count = stats['segment_counts'][speaker]
        time = stats['speaking_time'][speaker]
        pct = stats['speaking_percentage'][speaker]
        print(f"      {speaker}: {count} segments, "
              f"{format_duration(time)} ({pct:.1f}%)")

# ============================================================================
# ERROR HANDLING
# ============================================================================
class ProcessingError(Exception):
    """Custom exception for processing errors"""
    pass

def safe_execute(func, *args, error_msg: str = "Operation failed", **kwargs):
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Positional arguments
        error_msg: Error message prefix
        **kwargs: Keyword arguments
    
    Returns:
        Tuple of (success, result_or_error)
    """
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        logger = setup_logger()
        logger.error(f"{error_msg}: {str(e)}")
        return False, str(e)

