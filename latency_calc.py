import numpy as np
from scipy import signal
import matplotlib.pyplot as plt # Optional: for visualization
import argparse
import os

# --- Parse Command Line Arguments ---
parser = argparse.ArgumentParser(description='Calculate audio latency between render and capture PCM files')
parser.add_argument('--render', '-r', type=str, default='render.pcm',
                    help='Path to the render PCM file (default: render.pcm)')
parser.add_argument('--capture', '-c', type=str, default='capture.pcm',
                    help='Path to the capture PCM file (default: capture.pcm)')
parser.add_argument('--max-latency', '-m', type=float, default=1.0,
                    help='Maximum latency to consider in seconds (default: 1.0)')
parser.add_argument('--normalization', choices=['none', 'peak', 'rms', 'zscore'], 
                    default='zscore', help='Signal normalization method')
parser.add_argument('--output', '-o', type=str, default='latency_analysis.png',
                    help='Output image path (default: latency_analysis.png)')
parser.add_argument('--dpi', type=int, default=300,
                    help='DPI for output image (default: 300)')
args = parser.parse_args()

# --- Parameters ---
sample_rate = 16000  # Hz
dtype = np.int16     # s16le format

render_file = args.render
capture_file = args.capture

def analyze_signal(data, name):
    """Analyze signal characteristics."""
    peak = np.max(np.abs(data))
    rms = np.sqrt(np.mean(data**2))
    crest_factor = peak / rms if rms > 0 else float('inf')
    
    print(f"\n{name} Signal Analysis:")
    print(f"Peak level: {peak}")
    print(f"RMS level: {rms:.2f}")
    print(f"Crest factor: {crest_factor:.2f}")
    print(f"Dynamic range: {20 * np.log10(peak / np.mean(np.abs(data))):.1f} dB")
    
    return peak, rms, crest_factor

def normalize_signal(data, method='zscore'):
    """Normalize signal using specified method."""
    if method == 'none':
        return data
    elif method == 'peak':
        return data / np.max(np.abs(data))
    elif method == 'rms':
        return data / np.sqrt(np.mean(data**2))
    elif method == 'zscore':
        return (data - np.mean(data)) / np.std(data)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def find_best_alignment(render_data, capture_data, max_latency_samples):
    """Find the best alignment between render and capture signals."""
    # Normalize both signals
    render_norm = normalize_signal(render_data, args.normalization)
    capture_norm = normalize_signal(capture_data, args.normalization)
    
    # Calculate correlation for the valid range
    correlation = signal.correlate(capture_norm, render_norm, mode='full')
    
    # Find the center point (zero lag)
    zero_lag = len(correlation) // 2
    
    # Define the search range based on max_latency
    start_idx = zero_lag
    end_idx = min(zero_lag + max_latency_samples, len(correlation))
    
    # Find the best alignment in the valid range
    search_range = correlation[start_idx:end_idx]
    best_idx = np.argmax(search_range)
    best_correlation = search_range[best_idx]
    
    # Calculate the latency
    latency_samples = best_idx
    latency_seconds = latency_samples / sample_rate
    
    # Calculate alignment quality metrics
    # Compare aligned signals
    aligned_length = min(len(render_norm), len(capture_norm) - latency_samples)
    aligned_render = render_norm[:aligned_length]
    aligned_capture = capture_norm[latency_samples:latency_samples + aligned_length]
    
    # Calculate various metrics
    mse = np.mean((aligned_render - aligned_capture) ** 2)
    correlation_coef = np.corrcoef(aligned_render, aligned_capture)[0, 1]
    
    return {
        'latency_samples': latency_samples,
        'latency_seconds': latency_seconds,
        'latency_ms': latency_seconds * 1000,
        'correlation': best_correlation,
        'mse': mse,
        'correlation_coef': correlation_coef,
        'aligned_length': aligned_length,
        'aligned_render': aligned_render,
        'aligned_capture': aligned_capture
    }

# --- 1. Load the PCM Data ---
try:
    render_data = np.fromfile(render_file, dtype=dtype)
    capture_data = np.fromfile(capture_file, dtype=dtype)
    print(f"Loaded {len(render_data)} samples from {render_file}")
    print(f"Loaded {len(capture_data)} samples from {capture_file}")

    # Signal analysis
    render_peak, render_rms, render_crest = analyze_signal(render_data, "Render")
    capture_peak, capture_rms, capture_crest = analyze_signal(capture_data, "Capture")
    
    # Report level differences
    level_diff_db = 20 * np.log10(capture_rms / render_rms) if render_rms > 0 else float('inf')
    print(f"\nLevel Difference Analysis:")
    print(f"RMS level difference: {level_diff_db:.1f} dB (capture vs render)")
    
    if abs(level_diff_db) > 20:  # More than 20dB difference
        print("Warning: Large level difference between signals might affect correlation accuracy")

except FileNotFoundError as e:
    print(f"Error loading files: {e}. Make sure '{render_file}' and '{capture_file}' are in the correct path.")
    exit()
except Exception as e:
    print(f"An error occurred during file loading: {e}")
    exit()

# --- 2. Find Best Global Alignment ---
try:
    print("\nFinding best global alignment...")
    max_latency_samples = int(args.max_latency * sample_rate)
    
    result = find_best_alignment(render_data, capture_data, max_latency_samples)
    
    print("\n--- Alignment Results ---")
    print(f"Latency: {result['latency_ms']:.2f} ms")
    print(f"Correlation coefficient: {result['correlation_coef']:.4f}")
    print(f"Mean squared error: {result['mse']:.4f}")
    
    # --- Visualization ---
    print("\nGenerating visualization...")
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Original signals
    plt.subplot(311)
    time_render = np.arange(len(render_data)) / sample_rate
    time_capture = np.arange(len(capture_data)) / sample_rate
    plt.plot(time_render, render_data, label=f'Render (RMS: {render_rms:.0f})', alpha=0.7)
    plt.plot(time_capture, capture_data, label=f'Capture (RMS: {capture_rms:.0f})', alpha=0.7)
    plt.title('Original Signals')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Plot 2: Aligned signals
    plt.subplot(312)
    aligned_time = np.arange(result['aligned_length']) / sample_rate
    plt.plot(aligned_time, result['aligned_render'], 
            label='Render', alpha=0.7)
    plt.plot(aligned_time, result['aligned_capture'], 
            label=f'Capture (shifted by {result["latency_ms"]:.1f} ms)', 
            alpha=0.7)
    plt.title('Aligned Signals')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.grid(True)

    # Plot 3: Difference between aligned signals
    plt.subplot(313)
    difference = result['aligned_render'] - result['aligned_capture']
    plt.plot(aligned_time, difference, label='Difference', alpha=0.7)
    plt.title(f'Alignment Error (MSE: {result["mse"]:.4f})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Difference')
    plt.legend()
    plt.grid(True)
    
    # Add overall title with results
    plt.suptitle(f'Latency Analysis Results\nLatency: {result["latency_ms"]:.2f} ms, Correlation: {result["correlation_coef"]:.4f}', 
                 fontsize=12, y=0.95)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the figure
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f"Visualization saved to: {args.output}")
    
    # Close the figure to free memory
    plt.close()

except Exception as e:
    print(f"An error occurred during alignment calculation: {e}")
    raise
