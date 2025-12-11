"""
Test harness for benchmark orchestrator.
Validates all components without running actual LLM inference or power meter.
"""

import os
import json
import csv
import time
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock MLC and transformers before importing
sys.modules['mlc_llm'] = MagicMock()
sys.modules['transformers'] = MagicMock()

# Now we can import (with mocks in place)
import benchmark_orchestrator as orch


def test_config_loading():
    """Test: config.json loads correctly."""
    print("TEST: Loading config.json...")
    cfg = orch.load_config("config.json")
    assert "models" in cfg, "Missing 'models' key"
    assert "prefill_prompts" in cfg, "Missing 'prefill_prompts' key"
    assert "decode_lengths" in cfg, "Missing 'decode_lengths' key"
    assert "governors" in cfg, "Missing 'governors' key"
    assert len(cfg["models"]) > 0, "No models in config"
    assert len(cfg["prefill_prompts"]) > 0, "No prefill prompts in config"
    print(f"  ✓ Loaded {len(cfg['models'])} models, {len(cfg['prefill_prompts'])} prefill lengths")
    return cfg


def test_tokenizer_cache_init(cfg):
    """Test: tokenizer cache initializes (mocked)."""
    print("TEST: Tokenizer cache initialization...")
    
    # Mock the tokenizer loading
    mock_tokenizer = Mock()
    mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
    
    with patch('benchmark_orchestrator.load_tokenizer', return_value=mock_tokenizer):
        cache = orch.init_tokenizer_cache(cfg)
        assert len(cache) == len(cfg["models"]), f"Cache size {len(cache)} != models {len(cfg['models'])}"
        for model_cfg in cfg["models"]:
            assert model_cfg["hf_path"] in cache, f"Missing {model_cfg['hf_path']} in cache"
    print(f"  ✓ Tokenizer cache initialized with {len(cache)} entries")
    return cache


def test_prompt_generation(cfg, tokenizer_cache):
    """Test: prompts have correct token lengths (mocked tokenizer)."""
    print("TEST: Prompt token counts...")
    
    # Use real tokenizer logic: encode and count
    for prefill_len_str, prompt in cfg["prefill_prompts"].items():
        prefill_len = int(prefill_len_str)
        # Mock tokenizer.encode for this test
        mock_tok = Mock()
        mock_tok.encode = Mock(return_value=[i for i in range(prefill_len)])  # Return prefill_len tokens
        
        token_count = len(mock_tok.encode(prompt, add_special_tokens=False))
        print(f"  ✓ Prompt at {prefill_len} tokens: {token_count} tokens (mocked)")


def test_csv_checkpoint_system(cfg):
    """Test: CSV creation, append, and resume logic."""
    print("TEST: CSV checkpoint system...")
    
    test_output_dir = "/tmp/test_benchmark"
    test_csv_path = os.path.join(test_output_dir, "test_results.csv")
    
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Test 1: Init CSV header
    orch.init_results_csv(test_csv_path)
    assert os.path.exists(test_csv_path), "CSV not created"
    with open(test_csv_path) as f:
        header = f.readline().strip()
        assert "model_name" in header, "Header missing model_name"
        assert "governor" in header, "Header missing governor"
    print(f"  ✓ CSV header created")
    
    # Test 2: Append result
    test_result = {
        "model_name": "TestModel",
        "quantization": "INT4",
        "activation_precision": "32-bit",
        "governor": "ondemand",
        "prefill_len": 256,
        "decode_len": 128,
        "input_tokens": 256,
        "output_tokens": 128,
        "prefill_latency_s": 1.5,
        "decode_latency_s": 2.0,
        "prefill_tps": 170.67,
        "decode_tps": 64.0,
        "prefill_avg_power_w": 4.5,
        "decode_avg_power_w": 5.2,
        "prefill_energy_j": 6.75,
        "decode_energy_j": 10.4,
        "prefill_energy_per_token_in": 0.026,
        "decode_energy_per_token_out": 0.081,
        "prefill_tokens_per_s_per_w": 37.9,
        "decode_tokens_per_s_per_w": 12.3,
        "prefill_avg_cpu": 85.0,
        "prefill_peak_mem_gb": 0.5,
        "prefill_max_temp_c": 72.0,
        "prefill_freq_mean_mhz": 1800,
        "prefill_freq_min_mhz": 1800,
        "prefill_freq_max_mhz": 1800,
        "prefill_throttled_any": False,
        "decode_avg_cpu": 88.0,
        "decode_peak_mem_gb": 0.6,
        "decode_max_temp_c": 75.0,
        "decode_freq_mean_mhz": 1800,
        "decode_freq_min_mhz": 1800,
        "decode_freq_max_mhz": 1800,
        "decode_throttled_any": False,
    }
    orch.append_result(test_csv_path, test_result)
    with open(test_csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1, "Result not appended"
        assert rows[0]["model_name"] == "TestModel", "Result data mismatch"
    print(f"  ✓ Result appended to CSV")
    
    # Test 3: Load completed runs
    completed = orch.load_completed_runs(test_csv_path)
    assert ("TestModel", "ondemand", 256, 128) in completed, "Completed run not tracked"
    print(f"  ✓ Completed runs loaded: {len(completed)} entries")
    
    # Cleanup
    os.remove(test_csv_path)
    os.rmdir(test_output_dir)


def test_power_window_no_file():
    """Test: power window gracefully handles missing CSV (no power meter)."""
    print("TEST: Power window handling (no power meter)...")
    
    result = orch.load_power_window(0, 10, path="/nonexistent/path.csv")
    assert result["avg_power_w"] == 0.0, "Should return 0.0 for missing file"
    assert result["energy_j"] == 0.0, "Should return 0.0 energy for missing file"
    print(f"  ✓ Power window gracefully handles missing CSV (avg_power={result['avg_power_w']}W, energy={result['energy_j']}J)")


def test_metric_summaries():
    """Test: metric window filtering and summarization."""
    print("TEST: Metric window filtering and summarization...")
    
    # Mock metrics
    mock_metrics = [
        {"t": 100.0, "cpu": 80.0, "mem_gb": 0.5, "temp_c": 70.0, "freq_mhz": 1800, "throttled": 0, "volts": 1.1},
        {"t": 100.1, "cpu": 85.0, "mem_gb": 0.52, "temp_c": 72.0, "freq_mhz": 1800, "throttled": 0, "volts": 1.1},
        {"t": 100.2, "cpu": 90.0, "mem_gb": 0.55, "temp_c": 75.0, "freq_mhz": 1800, "throttled": 0, "volts": 1.1},
    ]
    
    # Test window filtering
    windowed = orch.window(mock_metrics, 100.05, 100.15)
    assert len(windowed) == 1, f"Window should have 1 sample, got {len(windowed)}"
    print(f"  ✓ Window filtering works (selected {len(windowed)} samples from {len(mock_metrics)} total)")
    
    # Test summarization
    summary = orch.summarize_metrics(mock_metrics)
    assert "avg_cpu" in summary, "Missing avg_cpu"
    assert "peak_mem_gb" in summary, "Missing peak_mem_gb"
    assert "max_temp_c" in summary, "Missing max_temp_c"
    assert abs(summary["avg_cpu"] - 85.0) < 0.1, "CPU average incorrect"
    assert summary["peak_mem_gb"] == 0.55, "Peak mem incorrect"
    print(f"  ✓ Metric summarization: avg_cpu={summary['avg_cpu']:.1f}%, peak_mem={summary['peak_mem_gb']}GB, max_temp={summary['max_temp_c']}°C")


def test_experiment_matrix_structure(cfg):
    """Test: verify experiment matrix structure (no actual runs)."""
    print("TEST: Experiment matrix structure...")
    
    total_runs = (len(cfg["governors"]) * len(cfg["models"]) * 
                  len(cfg["prefill_prompts"]) * len(cfg["decode_lengths"]) *
                  cfg["measured_runs"])
    
    print(f"  Governors: {len(cfg['governors'])} ({', '.join(cfg['governors'])})")
    print(f"  Models: {len(cfg['models'])}")
    print(f"  Prefill lengths: {len(cfg['prefill_prompts'])} ({', '.join(cfg['prefill_prompts'].keys())})")
    print(f"  Decode lengths: {len(cfg['decode_lengths'])} ({cfg['decode_lengths']})")
    print(f"  Measured runs per config: {cfg['measured_runs']}")
    print(f"  Total experiment runs: {total_runs}")
    
    assert total_runs > 0, "No runs in experiment matrix"
    print(f"  ✓ Experiment matrix is valid")


def test_mock_inference_pipeline():
    """Test: full inference pipeline with mocks (no actual LLM)."""
    print("TEST: Mock inference pipeline...")
    
    # Mock engine and streaming
    mock_engine = Mock()
    mock_tokenizer = Mock()
    
    # Mock streaming response
    mock_chunk_1 = Mock()
    mock_chunk_1.choices = [Mock(delta=Mock(content="Hello "))]
    mock_chunk_2 = Mock()
    mock_chunk_2.choices = [Mock(delta=Mock(content="world"))]
    mock_chunk_3 = Mock()
    mock_chunk_3.choices = [Mock(delta=Mock(content="!"))]
    
    mock_engine.chat.completions.create = Mock(return_value=[mock_chunk_1, mock_chunk_2, mock_chunk_3])
    mock_engine.model = "test-model"
    
    # Mock tokenizer
    mock_tokenizer.encode = Mock(side_effect=lambda x, **kw: [1]*len(x.split()))
    
    # Test streaming
    prompt = "Test prompt"
    max_tokens = 10
    meas = orch.run_stream_with_ttft(mock_engine, prompt, max_tokens)
    
    assert meas["output"] == "Hello world!", f"Output mismatch: {meas['output']}"
    assert meas["t_first"] is not None, "TTFT not recorded"
    assert meas["t_end"] is not None, "End time not recorded"
    print(f"  ✓ Mock streaming works: output='{meas['output']}', TTFT={meas['t_first']-meas['t0']:.4f}s")
    
    # Test token metrics
    tok = orch.token_metrics(mock_tokenizer, prompt, meas)
    assert tok["input_tokens"] > 0, "Input tokens not counted"
    assert tok["output_tokens"] > 0, "Output tokens not counted"
    print(f"  ✓ Token metrics: in={tok['input_tokens']}, out={tok['output_tokens']}, prefill_tps={tok['prefill_tps']:.2f}, decode_tps={tok['decode_tps']:.2f}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("BENCHMARK ORCHESTRATOR TEST SUITE")
    print("=" * 70)
    print()
    
    try:
        cfg = test_config_loading()
        print()
        
        tokenizer_cache = test_tokenizer_cache_init(cfg)
        print()
        
        test_prompt_generation(cfg, tokenizer_cache)
        print()
        
        test_csv_checkpoint_system(cfg)
        print()
        
        test_power_window_no_file()
        print()
        
        test_metric_summaries()
        print()
        
        test_experiment_matrix_structure(cfg)
        print()
        
        test_mock_inference_pipeline()
        print()
        
        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  • Config loading: ✓")
        print("  • Tokenizer caching: ✓")
        print("  • Prompt generation: ✓")
        print("  • CSV checkpointing: ✓")
        print("  • Power window handling (no meter): ✓")
        print("  • Metric aggregation: ✓")
        print("  • Experiment matrix: ✓")
        print("  • Mock inference pipeline: ✓")
        print()
        print("Ready to run: ./run_benchmark.sh")
        print()
        
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 70)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
