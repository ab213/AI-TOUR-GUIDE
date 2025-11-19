#!/usr/bin/env python3
"""
LLM Profiling System for Edge Devices
Measures latency, throughput, memory, CPU, temperature, and power for LLM inference.
"""

import os
import sys
import json
import time
import csv
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, fields
from datetime import datetime

import psutil
import matplotlib.pyplot as plt
import pandas as pd
from mlc_llm import MLCEngine
from tokenizers import Tokenizer
try:
    from parse_profiling_log import parse_log as parse_log_entries
except ImportError:
    parse_log_entries = None


def log_print(*args, **kwargs):
    """Print with immediate flush for logging compatibility"""
    print(*args, **kwargs, flush=True)

@dataclass
class Metrics:
    """Container for profiling metrics"""
    model: str
    quantization: str
    activation_precision: str
    prefill_len: int
    decode_len: int
    prefill_latency: float
    decode_latency: float
    tokens_per_sec: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    memory_mb: float
    cpu_pct: float
    temp_c: float
    power_w: float
    timestamp: str
    source: str = "live"


class MetricCollector:
    """Collects system metrics during inference"""
    
    def __init__(self, power_coefficient: float = 2.5):
        self.power_coefficient = power_coefficient
        self.process = psutil.Process()
        self.memory_samples = []
        self.cpu_samples = []
        self.temp_samples = []
        
    def start_monitoring(self):
        """Start background monitoring"""
        self.memory_samples = []
        self.cpu_samples = []
        self.temp_samples = []
        
    def sample(self):
        """Take a sample of current metrics"""
        self.memory_samples.append(self.process.memory_info().rss / (1024 * 1024))  # MB
        self.cpu_samples.append(psutil.cpu_percent(interval=None))
        temp = self._get_temperature()
        if temp is not None:
            self.temp_samples.append(temp)
    
    def _get_temperature(self) -> Optional[float]:
        """Get CPU temperature using vcgencmd (Raspberry Pi)"""
        try:
            result = subprocess.run(
                ['vcgencmd', 'measure_temp'],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                temp_str = result.stdout.strip().split('=')[1].split("'")[0]
                return float(temp_str)
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
            # vcgencmd not available (e.g., not on Raspberry Pi)
            # Try alternative: /sys/class/thermal/thermal_zone0/temp
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp_millidegrees = int(f.read().strip())
                    return temp_millidegrees / 1000.0
            except (FileNotFoundError, ValueError, IOError):
                pass
        return None
    
    def get_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics"""
        memory_mb = max(self.memory_samples) if self.memory_samples else 0.0
        cpu_pct = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
        temp_c = sum(self.temp_samples) / len(self.temp_samples) if self.temp_samples else 0.0
        power_w = (cpu_pct / 100.0) * self.power_coefficient
        
        return {
            'memory_mb': memory_mb,
            'cpu_pct': cpu_pct,
            'temp_c': temp_c,
            'power_w': power_w
        }


class Profiler:
    """Handles MLC engine initialization and inference"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        self.engine: Optional[MLCEngine] = None
        self.tokenizer: Optional[Tokenizer] = None
        
    def _load_tokenizer(self) -> bool:
        """Load tokenizer from model path. Returns True if successful."""
        try:
            # Try to find tokenizer.json in the model path
            # For HF:// paths, MLC downloads models to cache
            # We need to find the actual model directory
            
            # If model_path is a local path
            if not self.model_path.startswith("HF://"):
                tokenizer_path = Path(self.model_path) / "tokenizer.json"
                if tokenizer_path.exists():
                    self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
                    return True
            
            # For HF:// paths, try to find in MLC cache
            # MLC typically caches models in ~/.cache/mlc_llm or similar
            import os
            home = os.path.expanduser("~")
            cache_dirs = [
                Path(home) / ".cache" / "mlc_llm",
                Path(home) / ".cache" / "mlc-llm",
            ]
            
            # Extract model name from HF path
            if self.model_path.startswith("HF://"):
                model_name = self.model_path.replace("HF://", "").split("/")[-1]
                for cache_dir in cache_dirs:
                    if cache_dir.exists():
                        # Search for tokenizer.json in subdirectories
                        for tokenizer_file in cache_dir.rglob("tokenizer.json"):
                            # Check if it's in a directory that matches our model
                            if model_name.lower() in str(tokenizer_file.parent).lower():
                                self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
                                return True
            
            return False
        except Exception as e:
            log_print(f"Warning: Failed to load tokenizer: {e}")
            return False
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tokenizer if available, else estimate."""
        if self.tokenizer is not None:
            try:
                encoded = self.tokenizer.encode(text)
                return len(encoded.ids)
            except Exception as e:
                log_print(f"Warning: Tokenizer encoding failed: {e}, using estimation")
        
        # Fallback to word-based estimation
        return int(len(text.split()) * 0.75)
    
    def initialize(self):
        """Initialize the MLC engine and tokenizer"""
        if self.engine is None:
            log_print(f"Loading model: {self.model_path}")
            self.engine = MLCEngine(self.model_path, device=self.device)
            log_print("Model loaded successfully")
            
            # Try to load tokenizer
            if self._load_tokenizer():
                log_print("Tokenizer loaded successfully")
            else:
                log_print("Warning: Using word-based token estimation")
    
    def generate_with_metrics(
        self,
        prompt: str,
        max_tokens: int,
        collector: MetricCollector,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, Dict[str, float], int]:
        """
        Generate response and collect metrics.
        Returns: (response_text, metrics_dict, num_tokens)
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized")
        
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        collector.start_monitoring()
        
        # Measure prefill (time to first token)
        prefill_start = time.perf_counter()
        first_token_time = None
        tokens = []
        
        # Use streaming to measure prefill vs decode
        response_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        
        # Count prompt tokens using tokenizer
        if self.tokenizer is not None:
            try:
                # Count tokens in the full message (including system prompt if present)
                full_message = "\n".join([msg.get("content", "") for msg in messages])
                prompt_tokens = self.count_tokens(full_message)
            except Exception:
                pass
        
        last_chunk = None
        for chunk in self.engine.chat.completions.create(
            messages=messages,
            model=self.model_path,
            stream=True,
            max_tokens=max_tokens,
            temperature=0.7
        ):
            last_chunk = chunk  # Keep last chunk to check for usage stats
            if first_token_time is None:
                first_token_time = time.perf_counter()
                prefill_latency = first_token_time - prefill_start
            
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                response_text += content
                tokens.append(content)
                collector.sample()
        
        decode_end = time.perf_counter()
        
        # Check for usage statistics in the last chunk (MLC may provide this)
        if last_chunk is not None:
            # Check if chunk has usage attribute (like OpenAI API)
            if hasattr(last_chunk, 'usage'):
                usage = last_chunk.usage
                if hasattr(usage, 'prompt_tokens'):
                    prompt_tokens = usage.prompt_tokens
                if hasattr(usage, 'completion_tokens'):
                    completion_tokens = usage.completion_tokens
                if hasattr(usage, 'total_tokens'):
                    total_tokens = usage.total_tokens
        
        # Calculate decode latency (time after first token)
        if first_token_time:
            decode_latency = decode_end - first_token_time
        else:
            decode_latency = decode_end - prefill_start
            prefill_latency = decode_latency
        
        # Count tokens using tokenizer if available (if not provided by API)
        if completion_tokens == 0:
            num_tokens = self.count_tokens(response_text)
            completion_tokens = num_tokens
        else:
            num_tokens = completion_tokens
        
        # Calculate total if not provided
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        
        # Get final metrics
        metrics = collector.get_metrics()
        metrics['prefill_latency'] = prefill_latency
        metrics['decode_latency'] = decode_latency
        metrics['tokens_per_sec'] = num_tokens / decode_latency if decode_latency > 0 else 0.0
        metrics['prompt_tokens'] = prompt_tokens
        metrics['completion_tokens'] = completion_tokens
        metrics['total_tokens'] = total_tokens
        
        return response_text, metrics, num_tokens
    
    def cleanup(self):
        """Clean up engine resources"""
        if self.engine is not None:
            try:
                self.engine.terminate()
            except:
                pass
            self.engine = None


class ExperimentRunner:
    """Orchestrates the factorial experiment sweep"""
    
    def __init__(self, config: Dict, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = Path(self.config.get("results_file", self.output_dir / "results.csv"))
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        self.csv_fieldnames = [f.name for f in fields(Metrics)]
        self.results: List[Metrics] = []
        self.completed_counts: Dict[Tuple[str, str, str, int, int], int] = {}
        self._load_existing_results()
        
    def _experiment_key(
        self,
        model: str,
        quantization: str,
        activation_precision: str,
        prefill_len: int,
        decode_len: int,
    ) -> Tuple[str, str, str, int, int]:
        return (model, quantization, activation_precision, prefill_len, decode_len)

    def _load_existing_results(self):
        if not self.results_file.exists():
            return
        try:
            df = pd.read_csv(self.results_file)
        except Exception as exc:
            log_print(f"Warning: Could not read existing results file {self.results_file}: {exc}")
            return
        for _, row in df.iterrows():
            data = {field: row.get(field, None) for field in self.csv_fieldnames}
            try:
                metrics = Metrics(
                    model=str(data["model"]),
                    quantization=str(data["quantization"]),
                    activation_precision=str(data["activation_precision"]),
                    prefill_len=int(data["prefill_len"]),
                    decode_len=int(data["decode_len"]),
                    prefill_latency=float(data["prefill_latency"]),
                    decode_latency=float(data["decode_latency"]),
                    tokens_per_sec=float(data["tokens_per_sec"]),
                    prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(data.get("completion_tokens", 0) or 0),
                    total_tokens=int(data.get("total_tokens", 0) or 0),
                    memory_mb=float(data["memory_mb"]),
                    cpu_pct=float(data["cpu_pct"]),
                    temp_c=float(data["temp_c"]),
                    power_w=float(data["power_w"]),
                    timestamp=str(data["timestamp"]),
                    source=str(data.get("source", "log")),
                )
            except Exception as exc:
                log_print(f"Warning: Skipping malformed row in results file: {exc}")
                continue
            self.results.append(metrics)
            key = self._experiment_key(
                metrics.model,
                metrics.quantization,
                metrics.activation_precision,
                metrics.prefill_len,
                metrics.decode_len,
            )
            self.completed_counts[key] = self.completed_counts.get(key, 0) + 1
        log_print(f"Loaded {len(self.results)} existing result entries from {self.results_file}")

    def _record_result(self, metric_obj: Metrics):
        self.results.append(metric_obj)
        key = self._experiment_key(
            metric_obj.model,
            metric_obj.quantization,
            metric_obj.activation_precision,
            metric_obj.prefill_len,
            metric_obj.decode_len,
        )
        self.completed_counts[key] = self.completed_counts.get(key, 0) + 1
        self._append_to_csv(metric_obj)

    def _append_to_csv(self, metric_obj: Metrics):
        file_exists = self.results_file.exists()
        with self.results_file.open("a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(asdict(metric_obj))

    def generate_prompt(self, prefill_len: int, tokenizer: Optional[Tokenizer] = None) -> str:
        """Generate a test prompt of approximately the target token length using tokenizer if available"""
        template = self.config.get('test_prompt_template', 'Tell me about {topic}.')
        topics = self.config.get('test_topics', ['artificial intelligence'])
        topic = topics[0]  # Use first topic
        
        # Model has max context of 8192 tokens, so cap prefill_len to leave room
        # Account for chat template overhead (~10-20 tokens) and decode tokens
        max_prefill_tokens = 7000  # Conservative limit
        target_tokens = min(prefill_len, max_prefill_tokens)

        # Base sentence to repeat
        base_sentence = f"Explain {topic} in detail. "
        
        # If tokenizer is available, use it to build prompt accurately
        if tokenizer is not None:
            try:
                prompt = ""
                # Start with base sentence
                prompt = base_sentence
                current_tokens = len(tokenizer.encode(prompt).ids)
                
                # Keep adding sentences until we reach target (or close to it)
                while current_tokens < target_tokens:
                    # Add another sentence
                    prompt += base_sentence
                    encoded = tokenizer.encode(prompt)
                    current_tokens = len(encoded.ids)
                    
                    # Safety check to avoid infinite loop
                    if current_tokens >= target_tokens * 0.95:  # Stop at 95% to be safe
                        break
                
                # Trim if we overshot (shouldn't happen often, but be safe)
                if current_tokens > target_tokens:
                    # Binary search to find the right length
                    low, high = 0, len(prompt)
                    best_prompt = prompt
                    while low < high:
                        mid = (low + high) // 2
                        test_prompt = prompt[:mid]
                        test_tokens = len(tokenizer.encode(test_prompt).ids)
                        if test_tokens <= target_tokens:
                            best_prompt = test_prompt
                            low = mid + 1
                        else:
                            high = mid
                    prompt = best_prompt
                
                return prompt
            except Exception as e:
                log_print(f"Warning: Tokenizer-based prompt generation failed: {e}, using estimation")
        
        # Fallback to word-based estimation
        # Estimate: ~0.75 tokens per word for English text (or ~1.33 words per token)
        # Use conservative estimate: 1.5 words per token to account for longer words
        words_needed = int(target_tokens * 1.5)

        # Create a repeating pattern to reach target length
        words_per_sentence = len(base_sentence.split())

        # Calculate how many repetitions needed
        num_repetitions = max(1, words_needed // words_per_sentence)

        # Generate prompt by repeating the sentence
        prompt = base_sentence * num_repetitions

        # Trim to approximate target (slightly under to be safe)
        # Estimate ~4 characters per word, so target chars = words * 4
        target_chars = words_needed * 4
        prompt = prompt[:target_chars]

        return prompt
    
    def run_experiment(
        self,
        model_name: str,
        model_path: str,
        quantization: str,
        activation_precision: str,
        prefill_len: int,
        decode_len: int,
        warmup_runs: int,
        measured_runs: int
    ) -> List[Metrics]:
        """Run a single experiment configuration"""
        log_print(f"\n{'='*60}")
        log_print(f"Experiment: {model_name} | {quantization} | {activation_precision} | Prefill:{prefill_len} | Decode:{decode_len}")
        log_print(f"{'='*60}")
        
        profiler = Profiler(model_path, device=self.config.get('device', 'cpu'))
        collector = MetricCollector(self.config.get('power_coefficient', 2.5))
        key = self._experiment_key(model_name, quantization, activation_precision, prefill_len, decode_len)
        existing_runs = self.completed_counts.get(key, 0)
        runs_needed = max(0, measured_runs - existing_runs)
        if runs_needed == 0:
            log_print("All measured runs already completed for this configuration. Skipping.")
            return []
        if existing_runs > 0:
            log_print(f"Resuming configuration with {existing_runs} completed runs. {runs_needed} runs remaining.")
        
        try:
            profiler.initialize()
            
            # Generate prompt using tokenizer if available
            prompt = self.generate_prompt(prefill_len, tokenizer=profiler.tokenizer)
            
            # Warm-up runs
            if runs_needed > 0:
                log_print(f"Warm-up runs: {warmup_runs}")
                for i in range(warmup_runs):
                    log_print(f"  Warm-up {i+1}/{warmup_runs}...", end='\r')
                    try:
                        _, _, _ = profiler.generate_with_metrics(
                            prompt, decode_len, collector
                        )
                    except Exception as e:
                        log_print(f"\n  Warning: Warm-up failed: {e}")
                    time.sleep(0.5)  # Brief pause between runs
            else:
                log_print("Warm-up skipped (no runs remaining).")
            
            # Measured runs
            log_print(f"\nMeasured runs: {measured_runs}")
            run_metrics = []
            for run_index in range(existing_runs, measured_runs):
                display_idx = run_index + 1
                log_print(f"  Run {display_idx}/{measured_runs}...", end='\r')
                try:
                    response, metrics, num_tokens = profiler.generate_with_metrics(
                        prompt, decode_len, collector
                    )
                    
                    metric_obj = Metrics(
                        model=model_name,
                        quantization=quantization,
                        activation_precision=activation_precision,
                        prefill_len=prefill_len,
                        decode_len=decode_len,
                        prefill_latency=metrics['prefill_latency'],
                        decode_latency=metrics['decode_latency'],
                        tokens_per_sec=metrics['tokens_per_sec'],
                        prompt_tokens=int(metrics.get('prompt_tokens', 0)),
                        completion_tokens=int(metrics.get('completion_tokens', 0)),
                        total_tokens=int(metrics.get('total_tokens', 0)),
                        memory_mb=metrics['memory_mb'],
                        cpu_pct=metrics['cpu_pct'],
                        temp_c=metrics['temp_c'],
                        power_w=metrics['power_w'],
                        timestamp=datetime.now().isoformat(),
                        source='live'
                    )
                    run_metrics.append(metric_obj)
                    self._record_result(metric_obj)
                    log_print(f"  Run {display_idx}/{measured_runs} - Latency: {metrics['prefill_latency']:.3f}s prefill, {metrics['decode_latency']:.3f}s decode")
                except Exception as e:
                    log_print(f"\n  Error in run {display_idx}: {e}")
                
                time.sleep(0.5)  # Brief pause between runs
            
            if existing_runs > 0:
                log_print("Completed remaining runs for resumed configuration.")
            
            return run_metrics
            
        finally:
            profiler.cleanup()
    
    def run_full_sweep(self):
        """Run the complete factorial experiment matrix"""
        models = self.config.get('models', [])
        prefill_lengths = self.config.get('prefill_lengths', [])
        decode_lengths = self.config.get('decode_lengths', [])
        warmup_runs = self.config.get('warmup_runs', 1)
        measured_runs = self.config.get('measured_runs', 3)
        
        total_experiments = len(models) * len(prefill_lengths) * len(decode_lengths)
        current = 0
        
        for model_cfg in models:
            model_name = model_cfg['name']
            model_path = model_cfg['hf_path']
            quantization = model_cfg.get('quantization', 'UNKNOWN')
            activation_precision = model_cfg.get('activation_precision', 'UNKNOWN')
            
            for prefill_len in prefill_lengths:
                for decode_len in decode_lengths:
                    current += 1
                    log_print(f"\n[{current}/{total_experiments}] Running experiment...")
                    
                    metrics = self.run_experiment(
                        model_name, model_path, quantization, activation_precision,
                        prefill_len, decode_len,
                        warmup_runs, measured_runs
                    )
                    self.results.extend(metrics)
        
        return self.results


class ResultsGenerator:
    """Generates CSV output and matplotlib visualizations"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def save_csv(self, results: List[Metrics], csv_path: Optional[Path] = None):
        """Save results to CSV file"""
        if csv_path is None:
            csv_path = self.output_dir / "results.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(Metrics)])
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))
        
        log_print(f"\nResults saved to: {csv_path}")
        return csv_path
    
    def generate_plots(self, results: List[Metrics]):
        """Generate matplotlib visualizations"""
        if not results:
            log_print("No results to plot")
            return
        
        df = pd.DataFrame([asdict(r) for r in results])
        
        # Group by model and quantization for plotting
        metrics_to_plot = [
            ('prefill_latency', 'Prefill Latency (s)'),
            ('decode_latency', 'Decode Latency (s)'),
            ('tokens_per_sec', 'Throughput (tokens/s)'),
            ('memory_mb', 'Memory Usage (MB)'),
            ('cpu_pct', 'CPU Utilization (%)'),
            ('temp_c', 'Temperature (°C)'),
            ('power_w', 'Power (W)')
        ]
        
        for metric, ylabel in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Group by model, quantization, and activation_precision
            for model in df['model'].unique():
                for quant in df['quantization'].unique():
                    for act_prec in df['activation_precision'].unique():
                        subset = df[(df['model'] == model) & 
                                   (df['quantization'] == quant) & 
                                   (df['activation_precision'] == act_prec)]
                        if len(subset) > 0:
                            # Average across runs for same config
                            grouped = subset.groupby(['prefill_len', 'decode_len'])[metric].mean().reset_index()
                            label = f"{model} | {quant} | {act_prec}"
                            ax.plot(
                                grouped['prefill_len'],
                                grouped[metric],
                                marker='o',
                                label=label
                            )
            
            ax.set_xlabel('Prefill Length')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel} vs Prefill Length')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_path = self.output_dir / f"{metric}_plot.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            log_print(f"Plot saved: {plot_path}")


def test_model(model_path: str, device: str = 'cpu'):
    """Quick test to verify a model works"""
    log_print(f"Testing model: {model_path}")
    profiler = Profiler(model_path, device=device)
    collector = MetricCollector()
    
    try:
        profiler.initialize()
        test_prompt = "Say hello in one sentence."
        log_print(f"Prompt: {test_prompt}")
        
        response, metrics, num_tokens = profiler.generate_with_metrics(
            test_prompt, 50, collector
        )
        
        log_print(f"\nResponse: {response[:200]}...")
        log_print(f"\nMetrics:")
        log_print(f"  Prompt tokens: {metrics.get('prompt_tokens', 0)}")
        log_print(f"  Completion tokens: {metrics.get('completion_tokens', num_tokens)}")
        log_print(f"  Total tokens: {metrics.get('total_tokens', 0)}")
        log_print(f"  Prefill latency: {metrics['prefill_latency']:.3f}s")
        log_print(f"  Decode latency: {metrics['decode_latency']:.3f}s")
        log_print(f"  Throughput: {metrics['tokens_per_sec']:.2f} tokens/s")
        log_print(f"  Memory: {metrics['memory_mb']:.1f} MB")
        log_print(f"  CPU: {metrics['cpu_pct']:.1f}%")
        log_print(f"  Temperature: {metrics['temp_c']:.1f}°C")
        log_print(f"  Power: {metrics['power_w']:.2f}W")
        log_print("\n✅ Model test successful!")
        
    except Exception as e:
        log_print(f"\n❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        profiler.cleanup()
    
    return True


def main():
    parser = argparse.ArgumentParser(description='LLM Profiling System')
    parser.add_argument(
        '--config',
        type=str,
        default='scripts/profiling_config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--test-model',
        type=str,
        help='Test a single model (HF path) before running full experiments'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use (cpu, cuda, etc.)'
    )
    parser.add_argument(
        '--import-log',
        type=str,
        help='Parse an existing profiling log file, merge into results, and exit'
    )
    parser.add_argument(
        '--results-file',
        type=str,
        help='Path to results CSV (overrides default output_dir/results.csv)'
    )
    
    args = parser.parse_args()

    # Force unbuffered output for logging (important for nohup/background processes)
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        # Python < 3.7 compatibility
        pass
    
    # Test model mode
    if args.test_model:
        success = test_model(args.test_model, device=args.device)
        sys.exit(0 if success else 1)
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        log_print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if args.results_file:
        config['results_file'] = args.results_file
    
    # Override device if specified
    if args.device != 'cpu':
        config['device'] = args.device
    
    output_dir = config.get('output_dir', 'profiling_results')
    
    log_print("="*60)
    log_print("LLM Profiling System")
    log_print("="*60)
    log_print(f"Config: {config_path}")
    log_print(f"Output: {output_dir}")
    log_print(f"Models: {len(config.get('models', []))}")
    log_print(f"Prefill lengths: {config.get('prefill_lengths', [])}")
    log_print(f"Decode lengths: {config.get('decode_lengths', [])}")
    log_print("="*60)
    
    # Import log mode
    if args.import_log:
        if parse_log_entries is None:
            log_print("Log import requested but parse_profiling_log module not available.")
            sys.exit(1)
        log_path = Path(args.import_log)
        if not log_path.exists():
            log_print(f"Log file not found: {log_path}")
            sys.exit(1)
        log_text = log_path.read_text()
        records = parse_log_entries(log_text)
        if not records:
            log_print("No completed experiments found in log.")
            sys.exit(0)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        results_file = Path(config.get('results_file', output_dir_path / 'results.csv'))
        results_file.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(records)
        if results_file.exists():
            existing_df = pd.read_csv(results_file)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.drop_duplicates(
                subset=['model','quantization','activation_precision','prefill_len','decode_len','timestamp'],
                keep='first',
                inplace=True
            )
        else:
            combined_df = new_df
        combined_df.to_csv(results_file, index=False)
        log_print(f"Imported {len(new_df)} entries from log into {results_file}")
        sys.exit(0)
    
    # Run experiments
    runner = ExperimentRunner(config, output_dir)
    results = runner.run_full_sweep()
    
    # Generate output
    generator = ResultsGenerator(Path(output_dir))
    generator.save_csv(results, csv_path=runner.results_file)
    generator.generate_plots(results)
    
    log_print(f"\n✅ Profiling complete! Results in: {output_dir}")


if __name__ == '__main__':
    main()

