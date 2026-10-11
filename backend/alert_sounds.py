"""Audio tone and alert sound synthesizer for DrishtiDrive AI.

Generates standard WAV tone files for each alert stage without external dependencies.
Applies smooth attack and decay envelopes to avoid harsh clicks or startling transients.

Sound specifications:
- Stage 1: Gentle attention beep (520 Hz soft chime, ~250ms).
- Stage 2: Firmer repeating beep (750 Hz dual-pulse, ~380ms total).
- Stage 3: Urgent but understandable alarm (880 Hz - 1100 Hz alternating cadence).
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import struct
from typing import Dict, List, Optional
import wave

from backend.temporal_state import AlertStage


def synthesize_wav_tone(
    filepath: str | Path,
    frequencies: List[float],
    durations: List[float],
    volume: float = 0.5,
    sample_rate: int = 44100,
) -> Path:
    """Generate and write a multi-tone WAV file with smooth envelope shaping.

    Args:
        filepath: Target destination for the WAV file.
        frequencies: List of frequencies in Hz for each segment (0.0 for silence).
        durations: Duration in seconds for each corresponding frequency.
        volume: Amplitude scaling factor between 0.0 and 1.0.
        sample_rate: Audio sampling frequency in Hz.

    Returns:
        Path to the saved WAV audio file.
    """
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames: List[bytes] = []
    clamped_vol = max(0.0, min(1.0, volume))

    for freq, duration in zip(frequencies, durations):
        n_samples = int(sample_rate * duration)
        fade_samples = int(sample_rate * min(0.02, duration / 4))

        for i in range(n_samples):
            # Smooth trapezoidal envelope to prevent acoustic pops
            env = 1.0
            if i < fade_samples and fade_samples > 0:
                env = i / fade_samples
            elif i > n_samples - fade_samples and fade_samples > 0:
                env = (n_samples - i) / fade_samples

            if freq > 0.0:
                sample_val = env * clamped_vol * math.sin(2.0 * math.pi * freq * (i / sample_rate))
            else:
                sample_val = 0.0

            int_val = int(max(-32768, min(32767, sample_val * 32767)))
            frames.append(struct.pack("<h", int_val))

    with wave.open(str(out_path), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(frames))

    return out_path


def ensure_alert_tones(assets_dir: Optional[str | Path] = None) -> Dict[AlertStage, Path]:
    """Ensure standard tone WAV files exist for all three alert stages.

    Creates pristine tone files if they are missing or empty.

    Returns:
        Mapping from AlertStage to the respective tone WAV Path.
    """
    base_dir = Path(assets_dir) if assets_dir else Path(__file__).resolve().parent / "assets"
    base_dir.mkdir(parents=True, exist_ok=True)

    stage1_path = base_dir / "stage1_beep.wav"
    stage2_path = base_dir / "stage2_beep.wav"
    stage3_path = base_dir / "stage3_alarm.wav"

    if not stage1_path.exists() or stage1_path.stat().st_size == 0:
        synthesize_wav_tone(
            stage1_path,
            frequencies=[520.0],
            durations=[0.25],
            volume=0.45,
        )

    if not stage2_path.exists() or stage2_path.stat().st_size == 0:
        synthesize_wav_tone(
            stage2_path,
            frequencies=[750.0, 0.0, 750.0],
            durations=[0.15, 0.08, 0.15],
            volume=0.60,
        )

    if not stage3_path.exists() or stage3_path.stat().st_size == 0:
        synthesize_wav_tone(
            stage3_path,
            frequencies=[880.0, 1100.0, 880.0, 1100.0, 880.0],
            durations=[0.12, 0.12, 0.12, 0.12, 0.16],
            volume=0.75,
        )

    return {
        AlertStage.STAGE_1_EARLY: stage1_path,
        AlertStage.STAGE_2_DROWSY: stage2_path,
        AlertStage.STAGE_3_CRITICAL: stage3_path,
    }
