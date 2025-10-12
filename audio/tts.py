"""
Universal Text-to-Speech module for AI Tour Guide Helmet.

Supports both macOS (development) and Raspberry Pi 5 (deployment).
Audio automatically routes to Bluetooth headphones on both platforms.
"""

import tempfile
import os
import subprocess
import platform
from typing import Optional, Dict, Any


def init_tts(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initialize TTS engine.
    
    Args:
        config: Optional configuration dictionary
            - rate: Speech rate (default: 200 for macOS, 150 for Pi)
            - volume: Volume level 0.0-1.0 (default: 0.9)
            - voice: Voice ID (optional)
    
    Returns:
        TTS state dictionary containing engine and platform info
    """
    cfg = config or {}
    is_macos = platform.system() == "Darwin"
    is_pi = platform.system() == "Linux" and "arm" in platform.machine().lower()
    
    print(f"[TTS] Platform: {platform.system()} (macOS={is_macos}, Pi={is_pi})")
    
    try:
        import pyttsx3
        engine = pyttsx3.init()
        
        # Platform-optimized defaults
        rate = cfg.get("rate", 200 if is_macos else 150)
        volume = cfg.get("volume", 0.9)
        
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)
        
        if voice := cfg.get("voice"):
            engine.setProperty("voice", voice)
        
        print(f"[TTS] Initialized (rate={rate}, volume={volume})")
        
        return {
            "engine": engine,
            "is_macos": is_macos,
            "is_pi": is_pi
        }
        
    except Exception as e:
        print(f"[TTS] Error initializing engine: {e}")
        return {"engine": None, "is_macos": is_macos, "is_pi": is_pi}


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
    
    engine = state.get("engine")
    if not engine:
        print("[TTS] No engine available")
        return False
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        # Synthesize to file
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        
        # Play audio using platform-specific player
        player = "afplay" if state.get("is_macos") else "aplay"
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


def close(state: Dict[str, Any]) -> None:
    """Cleanup TTS resources.
    
    Args:
        state: TTS state from init_tts()
    """
    if engine := state.get("engine"):
        try:
            engine.stop()
        except:
            pass


if __name__ == "__main__":
    # Quick test
    print("=== TTS Test ===")
    state = init_tts()
    speak(state, "AI Tour Guide system initialized. Ready for landmarks.")
    close(state)
    print("✓ Test complete")