"""
Comprehensive benchmark analysis and visualization.
Generates graphs per analysis rubric: latency, throughput, energy, thermals, etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10


class BenchmarkAnalyzer:
    """Analyze and visualize benchmark results."""
    
    def __init__(self, csv_path, output_dir="analysis"):
        """Load results CSV and initialize."""
        self.df = pd.read_csv(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Derived metrics
        self._compute_derived_metrics()
        
        print(f"✓ Loaded {len(self.df)} runs")
        print(f"  Models: {self.df['model_name'].nunique()}")
        print(f"  Governors: {self.df['governor'].nunique()}")
        print(f"  Prefill lengths: {self.df['prefill_len'].nunique()}")
        print(f"  Decode lengths: {self.df['decode_len'].nunique()}")
        print()
    
    def _compute_derived_metrics(self):
        """Add normalized and derived columns."""
        # Normalize energy per token
        self.df['prefill_energy_per_token'] = (
            self.df['prefill_energy_j'] / self.df['input_tokens']
        )
        self.df['decode_energy_per_token'] = (
            self.df['decode_energy_j'] / self.df['output_tokens']
        )
        
        # Energy efficiency (tokens/s/W)
        self.df['prefill_efficiency'] = (
            self.df['prefill_tps'] / (self.df['prefill_avg_power_w'] + 1e-6)
        )
        self.df['decode_efficiency'] = (
            self.df['decode_tps'] / (self.df['decode_avg_power_w'] + 1e-6)
        )
        
        # Total metrics
        self.df['total_latency_s'] = self.df['prefill_latency_s'] + self.df['decode_latency_s']
        self.df['total_tokens'] = self.df['input_tokens'] + self.df['output_tokens']
        self.df['total_energy_j'] = self.df['prefill_energy_j'] + self.df['decode_energy_j']
        
        # Memory pressure
        self.df['memory_efficiency'] = self.df['total_tokens'] / (self.df['prefill_peak_mem_gb'] + 1e-6)
        
        # Thermal margin (assume 80°C is limit)
        self.df['thermal_margin_c'] = 80 - self.df['decode_max_temp_c']
        
        # Throttling indicator
        self.df['throttled'] = self.df['prefill_throttled_any'] | self.df['decode_throttled_any']
    
    # =====================================================================
    # CORE METRICS VISUALIZATIONS
    # =====================================================================
    
    def plot_latency_breakdown(self):
        """Prefill vs decode latency per model and quantization."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Group by model and quantization
        grouped = self.df.groupby(['model_name', 'quantization']).agg({
            'prefill_latency_s': 'mean',
            'decode_latency_s': 'mean',
        }).reset_index()
        
        x = np.arange(len(grouped))
        width = 0.35
        
        axes[0].bar(x - width/2, grouped['prefill_latency_s'], width, label='Prefill', alpha=0.8)
        axes[0].bar(x + width/2, grouped['decode_latency_s'], width, label='Decode', alpha=0.8)
        axes[0].set_xlabel('Model × Quantization')
        axes[0].set_ylabel('Latency (s)')
        axes[0].set_title('Latency Breakdown: Prefill vs Decode')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"{m[:12]}\n{q[:10]}" for m, q in zip(grouped['model_name'], grouped['quantization'])], fontsize=8)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Latency vs context length
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model].groupby('prefill_len').agg({
                'total_latency_s': 'mean'
            }).reset_index()
            axes[1].plot(model_data['prefill_len'], model_data['total_latency_s'], marker='o', label=model)
        
        axes[1].set_xlabel('Prefill Length (tokens)')
        axes[1].set_ylabel('Total Latency (s)')
        axes[1].set_title('Latency vs Context Length')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        self._save_fig('01_latency_breakdown')
    
    def plot_throughput_analysis(self):
        """Tokens/sec for prefill and decode."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Throughput per model
        grouped = self.df.groupby(['model_name', 'quantization']).agg({
            'prefill_tps': 'mean',
            'decode_tps': 'mean',
        }).reset_index()
        
        x = np.arange(len(grouped))
        width = 0.35
        
        axes[0].bar(x - width/2, grouped['prefill_tps'], width, label='Prefill', alpha=0.8)
        axes[0].bar(x + width/2, grouped['decode_tps'], width, label='Decode', alpha=0.8)
        axes[0].set_xlabel('Model × Quantization')
        axes[0].set_ylabel('Tokens/sec')
        axes[0].set_title('Throughput: Prefill vs Decode')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"{m[:12]}\n{q[:10]}" for m, q in zip(grouped['model_name'], grouped['quantization'])], fontsize=8)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Throughput vs governor
        for gov in self.df['governor'].unique():
            gov_data = self.df[self.df['governor'] == gov].groupby('model_name').agg({
                'decode_tps': 'mean'
            }).reset_index()
            axes[1].bar(gov_data['model_name'], gov_data['decode_tps'], label=gov, alpha=0.7)
        
        axes[1].set_ylabel('Decode Throughput (tokens/sec)')
        axes[1].set_title('Throughput by Governor')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        self._save_fig('02_throughput_analysis')
    
    def plot_energy_metrics(self):
        """Power consumption and energy per token."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Average power consumption
        grouped = self.df.groupby(['model_name', 'quantization']).agg({
            'prefill_avg_power_w': 'mean',
            'decode_avg_power_w': 'mean',
        }).reset_index()
        
        x = np.arange(len(grouped))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, grouped['prefill_avg_power_w'], width, label='Prefill', alpha=0.8)
        axes[0, 0].bar(x + width/2, grouped['decode_avg_power_w'], width, label='Decode', alpha=0.8)
        axes[0, 0].set_ylabel('Power (W)')
        axes[0, 0].set_title('Average Power Consumption')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f"{m[:12]}\n{q[:10]}" for m, q in zip(grouped['model_name'], grouped['quantization'])], fontsize=8)
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Energy per token (normalized)
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model].groupby('quantization').agg({
                'decode_energy_per_token': 'mean'
            }).reset_index()
            axes[0, 1].plot(range(len(model_data)), model_data['decode_energy_per_token'], marker='o', label=model)
        
        axes[0, 1].set_ylabel('Energy per Token (J)')
        axes[0, 1].set_title('Decode Energy Efficiency')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Total energy vs context length
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model].groupby('prefill_len').agg({
                'total_energy_j': 'mean'
            }).reset_index()
            axes[1, 0].plot(model_data['prefill_len'], model_data['total_energy_j'], marker='o', label=model)
        
        axes[1, 0].set_xlabel('Prefill Length (tokens)')
        axes[1, 0].set_ylabel('Total Energy (J)')
        axes[1, 0].set_title('Energy vs Context Length')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Energy efficiency: tokens/sec/W (prefill & decode, robust to outliers)
        grouped = (
            self.df
            .groupby(['model_name', 'quantization'])
            .agg({
                'prefill_efficiency': 'median',
                'decode_efficiency': 'median',
            })
            .reset_index()
        )

        x = np.arange(len(grouped))
        width = 0.35

        axes[1, 1].bar(
            x - width / 2,
            grouped['prefill_efficiency'],
            width,
            label='Prefill',
            alpha=0.8,
        )
        axes[1, 1].bar(
            x + width / 2,
            grouped['decode_efficiency'],
            width,
            label='Decode',
            alpha=0.8,
        )

        axes[1, 1].set_ylabel('Tokens/sec/W')
        axes[1, 1].set_title('Energy Efficiency (Higher is Better)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(
            [f"{m[:12]}\n{q[:10]}" for m, q in zip(grouped['model_name'], grouped['quantization'])],
            fontsize=8,
        )
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_fig('03_energy_metrics')
    
    def plot_memory_analysis(self):
        """Memory footprint and efficiency."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Peak memory by model
        grouped = self.df.groupby(['model_name', 'quantization']).agg({
            'prefill_peak_mem_gb': 'mean',
            'decode_peak_mem_gb': 'mean',
        }).reset_index()
        
        x = np.arange(len(grouped))
        width = 0.35
        
        axes[0].bar(x - width/2, grouped['prefill_peak_mem_gb'], width, label='Prefill', alpha=0.8)
        axes[0].bar(x + width/2, grouped['decode_peak_mem_gb'], width, label='Decode', alpha=0.8)
        axes[0].set_ylabel('Memory (GB)')
        axes[0].set_title('Peak Memory Usage')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"{m[:12]}\n{q[:10]}" for m, q in zip(grouped['model_name'], grouped['quantization'])], fontsize=8)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Memory efficiency: tokens per GB
        grouped = self.df.groupby(['model_name', 'quantization']).agg({
            'memory_efficiency': 'mean',
        }).reset_index()
        
        x = np.arange(len(grouped))
        axes[1].bar(x, grouped['memory_efficiency'], alpha=0.8, color='purple')
        axes[1].set_ylabel('Tokens/GB')
        axes[1].set_title('Memory Efficiency')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"{m[:12]}\n{q[:10]}" for m, q in zip(grouped['model_name'], grouped['quantization'])], fontsize=8)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_fig('04_memory_analysis')
    
    def plot_thermal_analysis(self):
        """Temperature and throttling behavior."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Peak temperatures
        grouped = self.df.groupby(['model_name', 'quantization']).agg({
            'prefill_max_temp_c': 'mean',
            'decode_max_temp_c': 'mean',
        }).reset_index()
        
        x = np.arange(len(grouped))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, grouped['prefill_max_temp_c'], width, label='Prefill', alpha=0.8)
        axes[0, 0].bar(x + width/2, grouped['decode_max_temp_c'], width, label='Decode', alpha=0.8)
        axes[0, 0].axhline(y=80, color='r', linestyle='--', label='Throttle Threshold', linewidth=2)
        axes[0, 0].set_ylabel('Temperature (°C)')
        axes[0, 0].set_title('Peak Temperature by Phase')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f"{m[:12]}\n{q[:10]}" for m, q in zip(grouped['model_name'], grouped['quantization'])], fontsize=8)
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Thermal margin
        grouped = self.df.groupby(['model_name', 'quantization']).agg({
            'thermal_margin_c': 'mean',
        }).reset_index()
        
        x = np.arange(len(grouped))
        colors = ['red' if m < 0 else 'orange' if m < 5 else 'green' for m in grouped['thermal_margin_c']]
        axes[0, 1].bar(x, grouped['thermal_margin_c'], alpha=0.8, color=colors)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_ylabel('Margin to Throttle (°C)')
        axes[0, 1].set_title('Thermal Headroom')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([f"{m[:12]}\n{q[:10]}" for m, q in zip(grouped['model_name'], grouped['quantization'])], fontsize=8)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Temperature vs context length
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model].groupby('prefill_len').agg({
                'decode_max_temp_c': 'mean'
            }).reset_index()
            axes[1, 0].plot(model_data['prefill_len'], model_data['decode_max_temp_c'], marker='o', label=model)
        
        axes[1, 0].axhline(y=80, color='r', linestyle='--', label='Throttle', linewidth=2)
        axes[1, 0].set_xlabel('Prefill Length (tokens)')
        axes[1, 0].set_ylabel('Temperature (°C)')
        axes[1, 0].set_title('Temperature vs Context Length')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Throttling summary
        throttled_pct = self.df.groupby(['model_name', 'quantization'])['throttled'].apply(lambda x: (x.sum() / len(x)) * 100).reset_index()
        throttled_pct.columns = ['model_name', 'quantization', 'throttle_pct']
        
        x = np.arange(len(throttled_pct))
        colors = ['red' if p > 0 else 'green' for p in throttled_pct['throttle_pct']]
        axes[1, 1].bar(x, throttled_pct['throttle_pct'], alpha=0.8, color=colors)
        axes[1, 1].set_ylabel('Throttled Runs (%)')
        axes[1, 1].set_title('Thermal Throttling Occurrence')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f"{m[:12]}\n{q[:10]}" for m, q in zip(throttled_pct['model_name'], throttled_pct['quantization'])], fontsize=8)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_fig('05_thermal_analysis')
    
    def plot_utilization_analysis(self):
        """CPU utilization and frequency scaling."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # CPU utilization
        grouped = self.df.groupby(['model_name', 'quantization']).agg({
            'prefill_avg_cpu': 'mean',
            'decode_avg_cpu': 'mean',
        }).reset_index()
        
        x = np.arange(len(grouped))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, grouped['prefill_avg_cpu'], width, label='Prefill', alpha=0.8)
        axes[0, 0].bar(x + width/2, grouped['decode_avg_cpu'], width, label='Decode', alpha=0.8)
        axes[0, 0].set_ylabel('CPU Usage (%)')
        axes[0, 0].set_title('CPU Utilization by Phase')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f"{m[:12]}\n{q[:10]}" for m, q in zip(grouped['model_name'], grouped['quantization'])], fontsize=8)
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Frequency scaling
        grouped = self.df.groupby(['model_name', 'quantization']).agg({
            'prefill_freq_mean_mhz': 'mean',
            'decode_freq_mean_mhz': 'mean',
        }).reset_index()
        
        x = np.arange(len(grouped))
        axes[0, 1].bar(x - width/2, grouped['prefill_freq_mean_mhz'], width, label='Prefill', alpha=0.8)
        axes[0, 1].bar(x + width/2, grouped['decode_freq_mean_mhz'], width, label='Decode', alpha=0.8)
        axes[0, 1].set_ylabel('Frequency (MHz)')
        axes[0, 1].set_title('CPU Frequency by Phase')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([f"{m[:12]}\n{q[:10]}" for m, q in zip(grouped['model_name'], grouped['quantization'])], fontsize=8)
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Governor impact on throughput
        for gov in self.df['governor'].unique():
            gov_data = self.df[self.df['governor'] == gov].groupby('model_name').agg({
                'decode_tps': 'mean'
            }).reset_index()
            axes[1, 0].plot(gov_data['model_name'], gov_data['decode_tps'], marker='o', label=gov)
        
        axes[1, 0].set_ylabel('Decode Throughput (tokens/sec)')
        axes[1, 0].set_title('Governor Impact on Throughput')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Governor impact on power
        for gov in self.df['governor'].unique():
            gov_data = self.df[self.df['governor'] == gov].groupby('model_name').agg({
                'decode_avg_power_w': 'mean'
            }).reset_index()
            axes[1, 1].plot(gov_data['model_name'], gov_data['decode_avg_power_w'], marker='s', label=gov)
        
        axes[1, 1].set_ylabel('Power (W)')
        axes[1, 1].set_title('Governor Impact on Power')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        self._save_fig('06_utilization_analysis')
    
    # =====================================================================
    # ADVANCED ANALYSIS
    # =====================================================================
    
    def plot_quantization_impact(self):
        """Quantization effect on accuracy proxy (latency increase)."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Latency difference (quantized vs full precision)
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model]
            full_prec = model_data[model_data['quantization'].str.contains('None')]
            quantized = model_data[~model_data['quantization'].str.contains('None')]
            
            if len(full_prec) > 0 and len(quantized) > 0:
                lat_improvement = ((full_prec['total_latency_s'].mean() - 
                                   quantized['total_latency_s'].mean()) / 
                                  full_prec['total_latency_s'].mean() * 100)
                axes[0, 0].bar(model, lat_improvement, alpha=0.7)
        
        axes[0, 0].set_ylabel('Latency Reduction (%)')
        axes[0, 0].set_title('Quantization Impact: Latency')
        axes[0, 0].grid(axis='y', alpha=0.3)
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Energy difference
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model]
            full_prec = model_data[model_data['quantization'].str.contains('None')]
            quantized = model_data[~model_data['quantization'].str.contains('None')]
            
            if len(full_prec) > 0 and len(quantized) > 0:
                energy_improvement = ((full_prec['total_energy_j'].mean() - 
                                      quantized['total_energy_j'].mean()) / 
                                     full_prec['total_energy_j'].mean() * 100)
                axes[0, 1].bar(model, energy_improvement, alpha=0.7, color='orange')
        
        axes[0, 1].set_ylabel('Energy Reduction (%)')
        axes[0, 1].set_title('Quantization Impact: Energy')
        axes[0, 1].grid(axis='y', alpha=0.3)
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Precision bit-width impact
        grouped = self.df.groupby('activation_precision').agg({
            'decode_tps': 'mean',
            'decode_avg_power_w': 'mean',
        }).reset_index()
        
        axes[1, 0].plot(grouped['activation_precision'], grouped['decode_tps'], marker='o', markersize=10)
        axes[1, 0].set_ylabel('Throughput (tokens/sec)')
        axes[1, 0].set_title('Activation Precision Impact')
        axes[1, 0].grid(alpha=0.3)
        
        axes[1, 1].plot(grouped['activation_precision'], grouped['decode_avg_power_w'], marker='s', markersize=10, color='red')
        axes[1, 1].set_ylabel('Power (W)')
        axes[1, 1].set_title('Precision Impact on Power')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        self._save_fig('07_quantization_impact')
    
    def plot_context_length_scaling(self):
        """How latency and energy scale with context length (knee analysis)."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prefill latency scaling
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model].groupby('prefill_len').agg({
                'prefill_latency_s': 'mean'
            }).reset_index()
            axes[0, 0].plot(model_data['prefill_len'], model_data['prefill_latency_s'], marker='o', label=model)
        
        axes[0, 0].set_xlabel('Prefill Length (tokens)')
        axes[0, 0].set_ylabel('Latency (s)')
        axes[0, 0].set_title('Prefill Latency Scaling')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Decode latency (should be constant)
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model].groupby('decode_len').agg({
                'decode_latency_s': 'mean'
            }).reset_index()
            axes[0, 1].plot(model_data['decode_len'], model_data['decode_latency_s'], marker='s', label=model)
        
        axes[0, 1].set_xlabel('Decode Length (tokens)')
        axes[0, 1].set_ylabel('Latency (s)')
        axes[0, 1].set_title('Decode Latency Scaling')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Prefill energy scaling
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model].groupby('prefill_len').agg({
                'prefill_energy_j': 'mean'
            }).reset_index()
            axes[1, 0].plot(model_data['prefill_len'], model_data['prefill_energy_j'], marker='o', label=model)
        
        axes[1, 0].set_xlabel('Prefill Length (tokens)')
        axes[1, 0].set_ylabel('Energy (J)')
        axes[1, 0].set_title('Prefill Energy Scaling')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Decode energy scaling
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model].groupby('decode_len').agg({
                'decode_energy_j': 'mean'
            }).reset_index()
            axes[1, 1].plot(model_data['decode_len'], model_data['decode_energy_j'], marker='s', label=model)
        
        axes[1, 1].set_xlabel('Decode Length (tokens)')
        axes[1, 1].set_ylabel('Energy (J)')
        axes[1, 1].set_title('Decode Energy Scaling')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        self._save_fig('08_context_length_scaling')
    
    def plot_pareto_frontiers(self):
        """Latency vs Energy and Throughput vs Power Pareto curves."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Latency vs Energy (lower-left is better)
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model]
            axes[0].scatter(model_data['total_latency_s'], model_data['total_energy_j'], 
                          label=model, alpha=0.6, s=100)
        
        axes[0].set_xlabel('Latency (s)')
        axes[0].set_ylabel('Energy (J)')
        axes[0].set_title('Latency-Energy Tradeoff (Lower-Left is Better)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Throughput vs Power (upper-left is better: high throughput, low power)
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model]
            axes[1].scatter(model_data['decode_avg_power_w'], model_data['decode_tps'], 
                          label=model, alpha=0.6, s=100)
        
        axes[1].set_xlabel('Power (W)')
        axes[1].set_ylabel('Throughput (tokens/sec)')
        axes[1].set_title('Throughput-Power Tradeoff (Upper-Left is Better)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        self._save_fig('09_pareto_frontiers')
    
    # =====================================================================
    # REPORTING & SUMMARY
    # =====================================================================
    
    def generate_summary_report(self):
        """Generate text summary with key findings and recommendations."""
        report = []
        report.append("=" * 80)
        report.append("BENCHMARK ANALYSIS SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Dataset overview
        report.append("1. DATASET OVERVIEW")
        report.append(f"  Total runs: {len(self.df)}")
        report.append(f"  Models tested: {self.df['model_name'].nunique()}")
        report.append(f"    - {', '.join(self.df['model_name'].unique())}")
        report.append(f"  Quantizations: {self.df['quantization'].nunique()}")
        report.append(f"    - {', '.join(self.df['quantization'].unique())}")
        report.append(f"  Governors: {', '.join(self.df['governor'].unique())}")
        report.append(f"  Context lengths: {sorted(self.df['prefill_len'].unique())} prefill, {sorted(self.df['decode_len'].unique())} decode")
        report.append("")
        
        # Performance summary
        report.append("2. PERFORMANCE SUMMARY (Averaged across all runs)")
        report.append(f"  Latency:")
        report.append(f"    - Prefill: {self.df['prefill_latency_s'].mean():.3f}s (σ={self.df['prefill_latency_s'].std():.3f})")
        report.append(f"    - Decode:  {self.df['decode_latency_s'].mean():.3f}s (σ={self.df['decode_latency_s'].std():.3f})")
        report.append(f"  Throughput:")
        report.append(f"    - Prefill: {self.df['prefill_tps'].mean():.1f} tokens/sec")
        report.append(f"    - Decode:  {self.df['decode_tps'].mean():.1f} tokens/sec")
        report.append(f"  Power:")
        report.append(f"    - Prefill: {self.df['prefill_avg_power_w'].mean():.2f}W")
        report.append(f"    - Decode:  {self.df['decode_avg_power_w'].mean():.2f}W")
        report.append(f"  Energy:")
        report.append(f"    - Per-token (decode): {self.df['decode_energy_per_token'].mean():.4f} J/token")
        report.append(f"  Efficiency:")
        report.append(f"    - Decode: {self.df['decode_efficiency'].mean():.2f} tokens/sec/W")
        report.append("")
        
        # Model comparison
        report.append("3. MODEL COMPARISON (Decoded phase)")
        for model in sorted(self.df['model_name'].unique()):
            model_data = self.df[self.df['model_name'] == model]
            report.append(f"  {model}:")
            report.append(f"    - Avg latency: {model_data['decode_latency_s'].mean():.3f}s")
            report.append(f"    - Avg throughput: {model_data['decode_tps'].mean():.1f} tokens/sec")
            report.append(f"    - Avg power: {model_data['decode_avg_power_w'].mean():.2f}W")
            report.append(f"    - Energy/token: {model_data['decode_energy_per_token'].mean():.4f} J/token")
            report.append(f"    - Efficiency: {model_data['decode_efficiency'].mean():.2f} tokens/sec/W")
        report.append("")
        
        # Quantization impact
        report.append("4. QUANTIZATION IMPACT")
        for model in sorted(self.df['model_name'].unique()):
            model_data = self.df[self.df['model_name'] == model]
            full_prec = model_data[model_data['quantization'].str.contains('None')]
            quantized = model_data[~model_data['quantization'].str.contains('None')]
            
            if len(full_prec) > 0 and len(quantized) > 0:
                lat_gain = ((full_prec['decode_latency_s'].mean() - quantized['decode_latency_s'].mean()) / 
                           full_prec['decode_latency_s'].mean() * 100)
                power_gain = ((full_prec['decode_avg_power_w'].mean() - quantized['decode_avg_power_w'].mean()) / 
                             full_prec['decode_avg_power_w'].mean() * 100)
                
                report.append(f"  {model}:")
                report.append(f"    - Quantization speedup: {lat_gain:.1f}%")
                report.append(f"    - Power reduction: {power_gain:.1f}%")
        report.append("")
        
        # Governor impact
        report.append("5. GOVERNOR IMPACT")
        for gov in sorted(self.df['governor'].unique()):
            gov_data = self.df[self.df['governor'] == gov]
            report.append(f"  {gov}:")
            report.append(f"    - Avg throughput: {gov_data['decode_tps'].mean():.1f} tokens/sec")
            report.append(f"    - Avg power: {gov_data['decode_avg_power_w'].mean():.2f}W")
            report.append(f"    - Avg frequency: {gov_data['decode_freq_mean_mhz'].mean():.0f} MHz")
        report.append("")
        
        # Thermal analysis
        report.append("6. THERMAL ANALYSIS")
        throttled_runs = self.df['throttled'].sum()
        throttled_pct = (throttled_runs / len(self.df)) * 100
        report.append(f"  Thermal throttling: {throttled_runs}/{len(self.df)} runs ({throttled_pct:.1f}%)")
        report.append(f"  Max temperature: {self.df['decode_max_temp_c'].max():.1f}°C")
        report.append(f"  Avg temperature: {self.df['decode_max_temp_c'].mean():.1f}°C")
        report.append(f"  Thermal margin: {(80 - self.df['decode_max_temp_c'].mean()):.1f}°C (headroom)")
        report.append("")
        
        # Memory analysis
        report.append("7. MEMORY ANALYSIS")
        report.append(f"  Peak memory (prefill): {self.df['prefill_peak_mem_gb'].max():.2f}GB")
        report.append(f"  Peak memory (decode): {self.df['decode_peak_mem_gb'].max():.2f}GB")
        report.append(f"  Memory efficiency: {self.df['memory_efficiency'].mean():.0f} tokens/GB")
        report.append("")
        
        # Recommendations
        report.append("8. DEPLOYMENT RECOMMENDATIONS")
        
        # Best latency
        best_lat = self.df.loc[self.df['decode_latency_s'].idxmin()]
        report.append(f"  Best latency: {best_lat['model_name']} @ {best_lat['quantization']}")
        report.append(f"    {best_lat['decode_latency_s']:.3f}s ({best_lat['decode_tps']:.1f} tokens/sec)")
        
        # Best efficiency
        best_eff = self.df.loc[self.df['decode_efficiency'].idxmax()]
        report.append(f"  Best efficiency: {best_eff['model_name']} @ {best_eff['quantization']}")
        report.append(f"    {best_eff['decode_efficiency']:.2f} tokens/sec/W")
        
        # Best overall (throughput per power)
        efficiency = self.df['decode_tps'] / (self.df['decode_avg_power_w'] + 1e-6)
        best_overall = self.df.loc[efficiency.idxmax()]
        report.append(f"  Best balanced: {best_overall['model_name']} @ {best_overall['quantization']}")
        
        # Scaling recommendations
        report.append("")
        report.append("  Context length scaling:")
        for prefill in sorted(self.df['prefill_len'].unique()):
            prefill_data = self.df[self.df['prefill_len'] == prefill]
            report.append(f"    {prefill} tokens: {prefill_data['prefill_latency_s'].mean():.2f}s prefill latency")
        
        report.append("")
        report.append("9. KEY FINDINGS & INSIGHTS")
        report.append("  - Quantization provides significant speedups with minimal thermal impact")
        report.append(f"  - Thermal margin: {'TIGHT (consider cooling)' if (80 - self.df['decode_max_temp_c'].mean()) < 5 else 'ADEQUATE'}")
        report.append(f"  - Governors: {'performance' if self.df[self.df['governor']=='performance']['decode_tps'].mean() > self.df[self.df['governor']=='ondemand']['decode_tps'].mean() else 'ondemand'} mode recommended")
        report.append(f"  - Scaling: {'Linear' if self.df['prefill_tps'].std() / self.df['prefill_tps'].mean() < 0.1 else 'Variable'} across context lengths")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save to file
        report_path = self.output_dir / "SUMMARY_REPORT.txt"
        with open(report_path, "w") as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✓ Report saved to {report_path}")
    
    def _save_fig(self, name):
        """Save figure with given name."""
        path = self.output_dir / f"{name}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {path}")


def main():
    """Run full analysis."""
    import sys
    
    csv_path = "results/results.csv" if len(sys.argv) < 2 else sys.argv[1]
    
    print(f"Loading results from {csv_path}...")
    analyzer = BenchmarkAnalyzer(csv_path)
    
    print("\nGenerating visualizations...")
    #analyzer.plot_latency_breakdown()
    #analyzer.plot_throughput_analysis()
    analyzer.plot_energy_metrics()
    #analyzer.plot_memory_analysis()
    #analyzer.plot_thermal_analysis()
    #analyzer.plot_utilization_analysis()
    #analyzer.plot_quantization_impact()
    #analyzer.plot_context_length_scaling()
    #analyzer.plot_pareto_frontiers()
    
    #print("\nGenerating summary report...")
    #analyzer.generate_summary_report()
    
    print(f"\n✓ All analyses complete!")
    print(f"  Output directory: {analyzer.output_dir}")


if __name__ == "__main__":
    main()

