"""
Universal Text-to-Speech module for AI Tour Guide Helmet.

Supports both macOS (development) and Raspberry Pi 5 (deployment).
Audio automatically routes to Bluetooth headphones on both platforms.
"""

import tempfile
import os
import subprocess
import platform
import wave
from typing import Optional, Dict, Any
from piper import PiperVoice


def init_tts() -> Dict[str, Any]:
    """Initialize TTS engine.

    Returns:
        TTS state dictionary containing voice and platform info
    """
    is_macos = platform.system() == "Darwin"
    is_pi = platform.system() == "Linux" and "arm" in platform.machine().lower()
    
    print(f"[TTS] Platform: {platform.system()} (macOS={is_macos}, Pi={is_pi})")
    
    try:
        # Get the directory where this script is located
        # Voice model downloaded from https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/ryan/medium
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "en_US-ryan-medium.onnx")
        
        voice = PiperVoice.load(model_path)
        
        print(f"[TTS] Initialized with Piper voice model")
        
        return {
            "voice": voice,
            "is_macos": is_macos,
            "is_pi": is_pi
        }
        
    except Exception as e:
        print(f"[TTS] Error initializing voice: {e}")
        return {"voice": None, "is_macos": is_macos, "is_pi": is_pi}


def speak(state: Dict[str, Any], text: str) -> bool:
    """Speak text via Bluetooth headphones.
    
    Args:
        state: TTS state from init_tts()
        text: Text to speak
    
    Returns:
        True if successful, False otherwise
    """
    if not text.strip():
        return False
    
    voice = state.get("voice")
    if not voice:
        print("[TTS] No voice available")
        return False
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        # Synthesize to WAV file
        with wave.open(tmp_path, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        
        # Play audio using platform-specific player
        if state.get("is_macos"):
            player = "afplay"
        elif state.get("is_pi"):
            player = "aplay"
        else:
            # WSL/Linux (logic is not ARM processor)
            player = "paplay"
        subprocess.run([player, tmp_path], check=True, capture_output=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[TTS] Playback error: {e}")
        return False
    except Exception as e:
        print(f"[TTS] Error: {e}")
        return False
    finally:
        # Cleanup temp file
        try:
            os.remove(tmp_path)
        except:
            pass


if __name__ == "__main__":
    # Quick test
    print("=== TTS Test ===")
    state = init_tts()
    speak(state, "AI Tour Guide system initialized. Ready for landmarks.")
    print("✓ Test complete")