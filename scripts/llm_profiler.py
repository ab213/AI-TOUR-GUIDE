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
from dataclasses import dataclass, asdict
from datetime import datetime

import psutil
import matplotlib.pyplot as plt
import pandas as pd
from mlc_llm import MLCEngine


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
    memory_mb: float
    cpu_pct: float
    temp_c: float
    power_w: float
    timestamp: str


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
        
    def initialize(self):
        """Initialize the MLC engine"""
        if self.engine is None:
            print(f"Loading model: {self.model_path}")
            self.engine = MLCEngine(self.model_path, device=self.device)
            print("Model loaded successfully")
    
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
        for chunk in self.engine.chat.completions.create(
            messages=messages,
            model=self.model_path,
            stream=True,
            max_tokens=max_tokens,
            temperature=0.7
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter()
                prefill_latency = first_token_time - prefill_start
            
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                response_text += content
                tokens.append(content)
                collector.sample()
        
        decode_end = time.perf_counter()
        
        # Calculate decode latency (time after first token)
        if first_token_time:
            decode_latency = decode_end - first_token_time
        else:
            decode_latency = decode_end - prefill_start
            prefill_latency = decode_latency
        
        # Estimate token count (rough approximation: ~0.75 tokens per word)
        # This is a reasonable approximation for English text
        num_tokens = int(len(response_text.split()) * 0.75)
        
        # Get final metrics
        metrics = collector.get_metrics()
        metrics['prefill_latency'] = prefill_latency
        metrics['decode_latency'] = decode_latency
        metrics['tokens_per_sec'] = num_tokens / decode_latency if decode_latency > 0 else 0.0
        
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
        self.results: List[Metrics] = []
        
    def generate_prompt(self, prefill_len: int) -> str:
        """Generate a test prompt of approximately the target length"""
        template = self.config.get('test_prompt_template', 'Tell me about {topic}.')
        topics = self.config.get('test_topics', ['artificial intelligence'])
        topic = topics[0]  # Use first topic
        
        # Create a prompt that will result in approximately prefill_len tokens
        # Simple approach: repeat topic with context
        base_prompt = f"Write a detailed explanation about {topic}. "
        # Rough estimate: ~10 words per token, so multiply by 10
        words_needed = max(50, prefill_len * 10)
        prompt = base_prompt * (words_needed // len(base_prompt.split()) + 1)
        return prompt[:words_needed * 6]  # Rough character limit
    
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
        print(f"\n{'='*60}")
        print(f"Experiment: {model_name} | {quantization} | {activation_precision} | Prefill:{prefill_len} | Decode:{decode_len}")
        print(f"{'='*60}")
        
        profiler = Profiler(model_path, device=self.config.get('device', 'cpu'))
        collector = MetricCollector(self.config.get('power_coefficient', 2.5))
        
        try:
            profiler.initialize()
            
            # Generate prompt
            prompt = self.generate_prompt(prefill_len)
            
            # Warm-up runs
            print(f"Warm-up runs: {warmup_runs}")
            for i in range(warmup_runs):
                print(f"  Warm-up {i+1}/{warmup_runs}...", end='\r')
                try:
                    _, _, _ = profiler.generate_with_metrics(
                        prompt, decode_len, collector
                    )
                except Exception as e:
                    print(f"\n  Warning: Warm-up failed: {e}")
                time.sleep(0.5)  # Brief pause between runs
            
            # Measured runs
            print(f"\nMeasured runs: {measured_runs}")
            run_metrics = []
            for i in range(measured_runs):
                print(f"  Run {i+1}/{measured_runs}...", end='\r')
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
                        memory_mb=metrics['memory_mb'],
                        cpu_pct=metrics['cpu_pct'],
                        temp_c=metrics['temp_c'],
                        power_w=metrics['power_w'],
                        timestamp=datetime.now().isoformat()
                    )
                    run_metrics.append(metric_obj)
                    print(f"  Run {i+1}/{measured_runs} - Latency: {metrics['prefill_latency']:.3f}s prefill, {metrics['decode_latency']:.3f}s decode")
                except Exception as e:
                    print(f"\n  Error in run {i+1}: {e}")
                
                time.sleep(0.5)  # Brief pause between runs
            
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
                    print(f"\n[{current}/{total_experiments}] Running experiment...")
                    
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
        
    def save_csv(self, results: List[Metrics], filename: str = "results.csv"):
        """Save results to CSV file"""
        csv_path = self.output_dir / filename
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[f.name for f in Metrics.__dataclass_fields__.values()])
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))
        
        print(f"\nResults saved to: {csv_path}")
        return csv_path
    
    def generate_plots(self, results: List[Metrics]):
        """Generate matplotlib visualizations"""
        if not results:
            print("No results to plot")
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
            
            print(f"Plot saved: {plot_path}")


def test_model(model_path: str, device: str = 'cpu'):
    """Quick test to verify a model works"""
    print(f"Testing model: {model_path}")
    profiler = Profiler(model_path, device=device)
    collector = MetricCollector()
    
    try:
        profiler.initialize()
        test_prompt = "Say hello in one sentence."
        print(f"Prompt: {test_prompt}")
        
        response, metrics, num_tokens = profiler.generate_with_metrics(
            test_prompt, 50, collector
        )
        
        print(f"\nResponse: {response[:200]}...")
        print(f"\nMetrics:")
        print(f"  Tokens: {num_tokens}")
        print(f"  Prefill latency: {metrics['prefill_latency']:.3f}s")
        print(f"  Decode latency: {metrics['decode_latency']:.3f}s")
        print(f"  Throughput: {metrics['tokens_per_sec']:.2f} tokens/s")
        print(f"  Memory: {metrics['memory_mb']:.1f} MB")
        print(f"  CPU: {metrics['cpu_pct']:.1f}%")
        print(f"  Temperature: {metrics['temp_c']:.1f}°C")
        print(f"  Power: {metrics['power_w']:.2f}W")
        print("\n✅ Model test successful!")
        
    except Exception as e:
        print(f"\n❌ Model test failed: {e}")
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
    
    args = parser.parse_args()
    
    # Test model mode
    if args.test_model:
        success = test_model(args.test_model, device=args.device)
        sys.exit(0 if success else 1)
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Override device if specified
    if args.device != 'cpu':
        config['device'] = args.device
    
    output_dir = config.get('output_dir', 'profiling_results')
    
    print("="*60)
    print("LLM Profiling System")
    print("="*60)
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Models: {len(config.get('models', []))}")
    print(f"Prefill lengths: {config.get('prefill_lengths', [])}")
    print(f"Decode lengths: {config.get('decode_lengths', [])}")
    print("="*60)
    
    # Run experiments
    runner = ExperimentRunner(config, output_dir)
    results = runner.run_full_sweep()
    
    # Generate output
    generator = ResultsGenerator(Path(output_dir))
    generator.save_csv(results)
    generator.generate_plots(results)
    
    print(f"\n✅ Profiling complete! Results in: {output_dir}")


if __name__ == '__main__':
    main()

