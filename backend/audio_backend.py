"""Local audio playback and speech engine backends for DrishtiDrive AI.

Supports offline text-to-speech via pyttsx3, prerecorded audio fallbacks,
configurable volume, and graceful handling of missing or disconnected audio hardware.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from backend.alert_sounds import ensure_alert_tones
from backend.temporal_state import AlertStage

logger = logging.getLogger("drishtidrive.audio")


@dataclass
class AudioBackendStatus:
    """Telemetry describing audio hardware state and playback activity."""

    device_available: bool = True
    is_playing: bool = False
    volume: float = 0.8
    last_played_stage: Optional[AlertStage] = None
    last_played_time: float = 0.0
    error_message: str = ""
    engine_type: str = "local"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize status to dictionary."""
        return {
            "device_available": self.device_available,
            "is_playing": self.is_playing,
            "volume": round(self.volume, 2),
            "last_played_stage": self.last_played_stage.value if self.last_played_stage else None,
            "last_played_time": round(self.last_played_time, 3),
            "error_message": self.error_message,
            "engine_type": self.engine_type,
        }


class BaseAudioBackend(ABC):
    """Abstract interface for audio playback and speech output."""

    @abstractmethod
    def play_tone(self, stage: AlertStage, volume: float) -> bool:
        """Play attention tone or alarm for the given alert stage."""
        pass

    @abstractmethod
    def speak(self, text: str, stage: AlertStage, language: str = "en", volume: float = 0.8) -> bool:
        """Deliver voice announcement using prerecorded audio or offline TTS."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop any active audio playback immediately."""
        pass

    @abstractmethod
    def get_status(self) -> AudioBackendStatus:
        """Retrieve current status of audio hardware and playback."""
        pass


class LocalAudioBackend(BaseAudioBackend):
    """Production offline audio backend using local WAV synthesis, winsound, and pyttsx3."""

    def __init__(
        self,
        assets_dir: Optional[str | Path] = None,
        use_tts: bool = True,
        initial_volume: float = 0.8,
    ) -> None:
        """Initialize local audio backend with verified offline assets."""
        self.assets_dir = Path(assets_dir) if assets_dir else Path(__file__).resolve().parent / "assets"
        self.use_tts = use_tts
        self.status = AudioBackendStatus(
            volume=max(0.0, min(1.0, initial_volume)),
            engine_type="local_offline",
        )

        # Ensure WAV tones are synthesized and ready
        try:
            self.tone_paths = ensure_alert_tones(self.assets_dir)
        except Exception as exc:
            logger.warning("Failed to prepare tone files: %s", exc)
            self.tone_paths = {}
            self.status.device_available = False
            self.status.error_message = f"Tone preparation error: {exc}"

        self._tts_engine = None
        if self.use_tts:
            self._init_tts_engine()

        self._wmp = None

    def _init_tts_engine(self) -> None:
        """Safely initialize offline pyttsx3 speech synthesizer."""
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.setProperty("volume", self.status.volume)
            self._tts_engine = engine
            logger.info("pyttsx3 offline speech engine initialized.")
        except Exception as exc:
            logger.warning("pyttsx3 initialization failed (%s); using prerecorded audio.", exc)
            self._tts_engine = None
            self.status.error_message = f"TTS init warning: {exc}"

    def play_tone(self, stage: AlertStage, volume: float) -> bool:
        """Play attention tone for the given stage synchronously inside worker thread."""
        if not self.status.device_available:
            return False

        tone_path = self.tone_paths.get(stage)
        if not tone_path or not tone_path.exists():
            logger.warning("Tone file missing for stage %s: %s", stage, tone_path)
            return False

        self.status.is_playing = True
        self.status.last_played_stage = stage
        self.status.last_played_time = time.time()

        try:
            if sys.platform == "win32":
                import winsound

                winsound.PlaySound(str(tone_path), winsound.SND_FILENAME)
            else:
                # Platform fallback for non-Windows edge environments
                time.sleep(0.2)
            self.status.is_playing = False
            return True
        except Exception as exc:
            logger.warning("Audio device error during tone playback: %s", exc)
            self.status.device_available = False
            self.status.error_message = f"Tone playback error: {exc}"
            self.status.is_playing = False
            return False

    def speak(self, text: str, stage: AlertStage, language: str = "en", volume: float = 0.8) -> bool:
        """Deliver voice announcement using prerecorded WAV/MP3 fallback or offline TTS."""
        if not self.status.device_available:
            return False

        self.status.is_playing = True
        self.status.last_played_stage = stage
        self.status.last_played_time = time.time()

        success = False

        # Play strictly user-provided voice MP3s from backend/assets
        success = self._play_prerecorded_mp3(stage, volume)

        # Fall back to offline TTS only if explicitly enabled and MP3 was unavailable
        if not success and self.use_tts and self._tts_engine is not None:
            try:
                self._tts_engine.setProperty("volume", max(0.0, min(1.0, volume)))
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
                success = True
            except Exception as exc:
                logger.warning("TTS speech failed: %s", exc)
                self.status.error_message = f"TTS error: {exc}"

        self.status.is_playing = False
        return success

    def _play_prerecorded_mp3(self, stage: AlertStage, volume: float = 0.8) -> bool:
        """Play user-provided voice MP3 file using Windows Media subsystem."""
        mp3_names = {
            AlertStage.STAGE_1_EARLY: "Stage1_Voice.mp3",
            AlertStage.STAGE_2_DROWSY: "Stage2_voice.mp3",
            AlertStage.STAGE_3_CRITICAL: "Stage3_voice.mp3",
        }
        filename = mp3_names.get(stage)
        if not filename:
            return False

        mp3_path = self.assets_dir / filename
        if not mp3_path.exists():
            logger.warning("Target voice MP3 file not found: %s", mp3_path)
            return False

        if sys.platform != "win32":
            time.sleep(2.0)
            return True

        try:
            import win32com.client

            wmp = win32com.client.Dispatch("WMPlayer.OCX")
            wmp.settings.volume = int(max(0.0, min(1.0, volume)) * 100)
            media = wmp.newMedia(str(mp3_path.resolve()))
            wmp.currentPlaylist.clear()
            wmp.currentPlaylist.appendItem(media)
            wmp.controls.play()

            dur = getattr(media, "duration", 0.0) or 0.0
            if dur <= 0.0:
                dur_map = {
                    AlertStage.STAGE_1_EARLY: 4.1,
                    AlertStage.STAGE_2_DROWSY: 6.3,
                    AlertStage.STAGE_3_CRITICAL: 7.3,
                }
                dur = dur_map.get(stage, 4.0)

            t_start = time.time()
            while time.time() - t_start < dur:
                if not self.status.is_playing:
                    wmp.controls.stop()
                    break
                time.sleep(0.05)
            return True
        except Exception as exc:
            logger.warning("Voice MP3 playback failed: %s", exc)
            self.status.device_available = False
            self.status.error_message = f"Audio device/player error: {exc}"
            return False

    def stop(self) -> None:
        """Stop active audio output."""
        if sys.platform == "win32":
            try:
                import winsound

                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        if self._tts_engine:
            try:
                self._tts_engine.stop()
            except Exception:
                pass
        self.status.is_playing = False

    def get_status(self) -> AudioBackendStatus:
        """Return snapshot of current audio status."""
        return self.status


class MockAudioBackend(BaseAudioBackend):
    """Deterministic in-memory audio mock for unit and integration testing."""

    def __init__(self, simulate_device_failure: bool = False) -> None:
        """Initialize mock backend with telemetry tracking."""
        self.simulate_device_failure = simulate_device_failure
        self.status = AudioBackendStatus(
            device_available=not simulate_device_failure,
            engine_type="mock",
        )
        self.played_tones: List[Tuple[AlertStage, float]] = []
        self.spoken_messages: List[Tuple[str, AlertStage, str]] = []
        self.stop_calls: int = 0

    def play_tone(self, stage: AlertStage, volume: float) -> bool:
        """Record played tone in mock history."""
        if not self.status.device_available:
            return False
        self.played_tones.append((stage, volume))
        self.status.last_played_stage = stage
        self.status.last_played_time = time.time()
        return True

    def speak(
        self,
        text: str,
        stage: AlertStage,
        language: str = "en",
        volume: float = 0.8,
    ) -> bool:
        """Record spoken message in mock history."""
        if not self.status.device_available:
            return False
        self.spoken_messages.append((text, stage, language))
        self.status.last_played_stage = stage
        self.status.last_played_time = time.time()
        return True

    def stop(self) -> None:
        """Record stop call."""
        self.stop_calls += 1
        self.status.is_playing = False

    def get_status(self) -> AudioBackendStatus:
        """Return mock audio status."""
        return self.status
