import io
import hashlib
import threading
import queue
import tempfile
import os
from collections import OrderedDict
from typing import Generator, Optional, Dict, Any

# simple thread-safe LRU cache
class _LRUCache:
    def __init__(self, max_entries: int = 64):
        self.max_entries = max_entries
        self.lock = threading.Lock()
        self.data: "OrderedDict[str, bytes]" = OrderedDict()

    def get(self, key: str) -> Optional[bytes]:
        with self.lock:
            v = self.data.get(key)
            if v is not None:
                # move to end -> most recently used
                self.data.move_to_end(key)
            return v

    def set(self, key: str, value: bytes):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
            self.data[key] = value
            if len(self.data) > self.max_entries:
                # pop oldest
                self.data.popitem(last=False)

_cache = _LRUCache(max_entries=64)

def init_tts(config: Optional[Dict[str, Any]] = None):
    """Initialize TTS driver.

    This scaffold prefers an on-device engine (`pyttsx3`) if available.
    `config` can include engine options like rate/voice/cache_dir.
    """
    cfg = config or {}
    engine = None
    try:
        print("init_tts: attempting to initialize pyttsx3 engine")
        import pyttsx3

        engine = pyttsx3.init()
        print("init_tts: engine initialized")
        rate = cfg.get("rate")
        if rate:
            engine.setProperty("rate", rate)
            print(f"init_tts: set rate={rate}")
        voice = cfg.get("voice")
        if voice:
            engine.setProperty("voice", voice)
            print(f"init_tts: set voice={voice}")

    except Exception as e:
        print(f"init_tts: pyttsx3 not available or failed to init: {e}")
        engine = None

    return {"engine": engine, "config": cfg}

def _synthesize_blocking(engine, text: str) -> bytes:
    """Synchronous synthesis to bytes using pyttsx3 when available.

    pyttsx3 doesn't natively return bytes; we drive it to write to a temporary
    file-like buffer using its save_to_file API if supported, otherwise return
    empty bytes.
    """
    try:
        # write to a temporary file to avoid polluting project dir.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            filename = tmp.name
        print(f"_synthesize_blocking: temporary file -> {filename}")

        # request pyttsx3 to write to filename and wait
        print("_synthesize_blocking: calling engine.save_to_file")
        engine.save_to_file(text, filename)
        print("_synthesize_blocking: waiting for engine to finish")
        engine.runAndWait()

        # read result and remove the temp file
        try:
            with open(filename, "rb") as f:
                data = f.read()
            print(f"_synthesize_blocking: read {len(data)} bytes from {filename}")
        finally:
            try:
                os.remove(filename)
                print(f"_synthesize_blocking: removed temp file {filename}")
            except Exception as e:
                print(f"_synthesize_blocking: failed to remove temp file {filename}: {e}")
        return data
    except Exception as e:
        print(f"_synthesize_blocking: synthesis failed: {e}")
        return b""

def synthesize(state: Dict[str, Any], text: str, chunk_size: int = 4096) -> Generator[bytes, None, None]:
    """Return a generator that yields audio bytes for `text`.

    Behavior:
    - If text has been generated before, return cached bytes (fast).
    - If an engine is present, synthesize in a background thread and stream
      chunks via a queue.
    - Otherwise, yield a short silence placeholder (so callers get some
      bytes and can keep the pipeline moving).
    """
    key = hashlib.sha1(text.encode()).hexdigest()
    print(f"synthesize: request key={key} text_len={len(text)}")
    cached = _cache.get(key)
    if cached is not None:
        print("synthesize: cache hit")
        data = cached
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]
        return

    engine = state.get("engine")
    if engine:
        # bounded queue to apply backpressure and avoid unbounded memory growth
        q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=4)

        def worker():
            print("worker: starting synthesis")
            data = _synthesize_blocking(engine, text)
            # cache
            try:
                if data:
                    _cache.set(key, data)
                    print(f"worker: cached {len(data)} bytes for key={key}")
                else:
                    print(f"worker: no data returned for key={key}")
            except Exception as e:
                print(f"worker: cache set failed: {e}")
            # push chunks (block if queue is full)
            for i in range(0, len(data), chunk_size):
                q.put(data[i : i + chunk_size])
            q.put(None)
            print("worker: finished")

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        print("synthesize: waiting for chunks from worker")
        while True:
            chunk = q.get()
            if chunk is None:
                print("synthesize: worker finished, no more chunks")
                break
            yield chunk
        return

    # Fallback: yield a short silence WAV header or empty bytes.
    # Keep minimal usable WAV header for silence (44 bytes + empty data).
    # Fallback: produce a minimal valid WAV with a tiny silence frame using wave
    print("synthesize: no engine available, producing silence fallback")
    try:
        import wave
        import struct

        with io.BytesIO() as buf:
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)  # 16-bit
                w.setframerate(22050)
                # one frame of silence
                w.writeframes(struct.pack("h", 0))
            silence = buf.getvalue()
    except Exception as e:
        print(f"synthesize: failed to build wave silence: {e}")
        silence = b"\x52\x49\x46\x46" + (b"\x00" * 40)

    try:
        _cache.set(key, silence)
    except Exception as e:
        print(f"synthesize: cache set failed for silence: {e}")
    yield silence

def close(state: Dict[str, Any]):
    engine = state.get("engine")
    try:
        if engine:
            print("close: stopping engine")
            engine.stop()
    except Exception:
        print("close: engine.stop() failed")

if __name__ == "__main__":
    state = init_tts()
    text = "Hello World! This is MORPHEUS."
    gen = synthesize(state, text)
    out_path = "../audio/out.wav"
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in gen:
            f.write(chunk)
    close(state)
    print(f"Wrote {out_path}")
