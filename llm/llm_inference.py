from mlc_llm import MLCEngine

import os



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#MODEL_PATH = os.path.join(BASE_DIR, "model")
MODEL_PATH = "HF://mlc-ai/SmolLM2-135M-Instruct-q0f32-MLC"
PROMPT_FILE = os.path.join(BASE_DIR, "test_prompt.txt")

_engine = None
_system_prompt = None

def initialize_model():
    global _engine, _system_prompt
    if _engine is None:
        print("Loading LLM model...")
        _engine = MLCEngine(MODEL_PATH, device='cpu')
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


# ---------------------------------------------------------------------
# 🧪 TEST BLOCK
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("--- LLM Inference Module Test ---")

    # Step 1: Initialize model
    try:
        import time
        start_t = time.time()
        engine, prompt = initialize_model()
        print(f"✅ Model initialized successfully in {time.time() - start_t}s.")
        print(f"Prompt preview: {prompt[:80]}...\n")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        exit(1)

    # Step 2: Test with example POI data
    test_pois = [
        {"name": "Georgia Aquarium", "city": "Atlanta"},
        {"name": "Eiffel Tower", "city": "Paris"},
        {"name": "Golden Gate Bridge", "city": "San Francisco"},
    ]

    for poi in test_pois:
        print(f"--- Generating response for {poi['name']} ---")
        try:
            start_t = time.time()
            response = generate_response(poi)
            print(f"Response (took {time.time() - start_t}s):\n", response)
        except Exception as e:
            print(f"❌ Error generating response for {poi['name']}: {e}")
        print("-" * 60)

    print("\n✅ Test completed.")
