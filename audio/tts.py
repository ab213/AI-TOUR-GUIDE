import pyttsx3
import os
import tempfile
import threading
import queue
import time
import platform
import subprocess
import logging
import shutil

# === Logging Configuration ===
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(threadName)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)


class HybridTTS:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True, name="TTS-Worker")
        self.running = True
        self.lock = threading.Lock()
        self.system = platform.system()

        # Configure speaking rate
        rate = self.engine.getProperty("rate")
        self.engine.setProperty("rate", 200)
        logging.info(f"[INIT] Engine ready (rate={rate}) on {self.system} ✅")

        # Optional: check if Bluetooth speaker is connected
        self._check_bluetooth_audio()

        self.thread.start()

    # === Bluetooth Detection ===
    def _check_bluetooth_audio(self):
        try:
            if self.system == "Darwin":
                result = subprocess.run(
                    ["system_profiler", "SPBluetoothDataType"],
                    capture_output=True,
                    text=True
                )
                if "Connected: Yes" in result.stdout:
                    logging.info("[BT] Bluetooth speaker detected 🎧")
                else:
                    logging.warning("[BT] No Bluetooth audio device found 🔊")

            elif self.system == "Linux":
                result = subprocess.run(["pactl", "list", "sinks"], capture_output=True, text=True)
                if "bluez" in result.stdout.lower():
                    logging.info("[BT] Bluetooth speaker connected 🎧")
                else:
                    logging.warning("[BT] No Bluetooth sink detected, using default ALSA 🔊")

        except Exception as e:
            logging.debug(f"[BT] Bluetooth check skipped or failed: {e}")

    # === Worker Thread ===
    def _worker(self):
        logging.debug("[THREAD] Started worker loop...")
        while self.running:
            try:
                text = self.queue.get(timeout=1)
                if text is None:
                    break
                self._speak_text(text)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.exception(f"[ERROR] Worker exception: {e}")
        logging.debug("[THREAD] Worker loop ended.")

    # === Speak Text ===
    def _speak_text(self, text: str):
        logging.info(f"[THREAD] Speaking text ({len(text)} chars)")
        try:
            if self.system == "Darwin":
                self._speak_system_macos(text)
            elif self.system == "Linux":
                self._speak_system_linux(text)
            else:
                self._speak_pyttsx3(text)
        except Exception as e:
            logging.exception(f"[ERROR] Speak failed: {e}")

    # === macOS: use built-in 'say' ===
    def _speak_system_macos(self, text: str):
        with self.lock:
            logging.debug(f"[MACOS] Using 'say' command for text: {text[:50]}...")
            subprocess.run(["say", text])
            logging.debug("[MACOS] Completed 'say' playback ✅")

    # === Linux (Raspberry Pi): use espeak/aplay fallback ===
    def _speak_system_linux(self, text: str):
        with self.lock:
            logging.debug(f"[LINUX] Preparing to speak: {text[:50]}...")
            if shutil.which("espeak"):
                logging.debug("[LINUX] Using 'espeak' for playback")
                subprocess.run(["espeak", text])
            elif shutil.which("aplay"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                    path = tmpfile.name
                self.engine.save_to_file(text, path)
                self.engine.runAndWait()
                subprocess.run(["aplay", path])
                os.remove(path)
            else:
                logging.warning("[LINUX] No TTS backend found, falling back to pyttsx3 engine")
                self.engine.say(text)
                self.engine.runAndWait()
            logging.debug("[LINUX] Completed playback ✅")

    # === Generic Fallback ===
    def _speak_pyttsx3(self, text: str):
        with self.lock:
            self.engine.say(text)
            self.engine.runAndWait()
        logging.debug("[GENERIC] Playback done ✅")

    # === Queue Handling ===
    def say(self, text: str):
        if not text.strip():
            logging.debug("[QUEUE] Ignored empty text")
            return
        self.queue.put(text)
        logging.info(f"[QUEUE] Added text → size={self.queue.qsize()}")

    # === Graceful Shutdown ===
    def stop(self):
        logging.info("[CLOSE] Stopping TTS...")
        self.running = False
        self.queue.put(None)
        self.thread.join(timeout=5)
        self.engine.stop()
        logging.info("[CLOSE] Engine stopped cleanly ✅")


# === TESTING ===
if __name__ == "__main__":
    tts = HybridTTS()
    try:
        logging.info("[TEST] Starting multi-queue test 🚀")
        tts.say("Hello Aditya, the hybrid text to speech system is now online.")
        tts.say("This is the second message. It should play after the first one.")
        tts.say("Finally, this confirms all queues, locks, and threading are working.")
        time.sleep(20)
    finally:
        tts.stop()
