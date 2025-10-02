# llm/llm_inference.py
from mlc_llm import MLCEngine
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model")
PROMPT_FILE = os.path.join(BASE_DIR, "prompt.txt")

_engine = None
_system_prompt = None

def initialize_model():
    global _engine, _system_prompt
    if _engine is None:
        print("Loading LLM model...")
        _engine = MLCEngine(MODEL_PATH)
        print("Model loaded successfully.")

    if _system_prompt is None:
        with open(PROMPT_FILE, "r") as f:
            _system_prompt = f.read()
    return _engine, _system_prompt

def generate_response(poi_info: dict) -> str:
    engine, system_prompt = initialize_model()
    user_input = f"Tell me about {poi_info.get('name', 'this place')} located in {poi_info.get('city', 'Atlanta')}."
    
    response = engine.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        model=MODEL_PATH,
        stream=False,
        max_tokens=800,
        temperature=0.7,
        top_p=0.9
    )

    if response.choices and response.choices[0].message.content:
        return response.choices[0].message.content.strip()
    return "No response generated."
