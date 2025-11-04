import pyttsx3
import tempfile
import subprocess
import os

engine = pyttsx3.init()
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
    filename = tmpfile.name

engine.save_to_file("This is a test of the audio playback system.", filename)
print(f"audio saved to {filename}")
