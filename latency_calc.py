#!/usr/bin/env python3
# File: latency_calc.py (Modified to use only corrected logic and 3-subplot plot)

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import argparse
import os
import warnings
import time

# --- Constants ---
DEFAULT_SAMPLE_RATE = 16000
PCM_DTYPE = np.int16

# --- Robust Normalization Function ---
def normalize_signal_robust(data_raw, method='zscore', context_label=""):
    """Normalizes raw data (e.g., int16) to float64."""
    if not isinstance(data_raw, np.ndarray):
        raise TypeError(f"Input data for {context_label} must be a NumPy array.")
    if data_raw.size == 0:
        print(f"[Normalize-{context_label}] Warning: Normalizing empty array ({method}).")
        return np.array([], dtype=np.float64)

    data_dbl = data_raw.astype(np.float64, copy=False)
    # print(f"[Normalize-{context_label}] Normalizing {data_raw.size} samples using '{method}' method...") # Verbose

    if method == 'zscore':
        mean = np.mean(data_dbl)
        std = np.std(data_dbl)
        if std < 1e-9:
            # print(f"[Normalize-{context_label}] Warning: Std dev is near zero ({std:.2e}). Normalizing to zeros.")
            return np.zeros_like(data_dbl, dtype=np.float64)
        else:
            return (data_dbl - mean) / std
    elif method == 'peak':
        peak = np.max(np.abs(data_dbl))
        if peak < 1e-9:
             # print(f"[Normalize-{context_label}] Warning: Peak is near zero. Normalizing to zeros.")
             return np.zeros_like(data_dbl, dtype=np.float64)
        else:
             return data_dbl / peak
    elif method == 'rms':
         rms_val = np.sqrt(np.mean(data_dbl**2))
         if rms_val < 1e-9:
              # print(f"[Normalize-{context_label}] Warning: RMS is near zero. Normalizing to zeros.")
              return np.zeros_like(data_dbl, dtype=np.float64)
         else:
              return data_dbl / rms_val
    elif method == 'none':
         return data_dbl
    else:
        raise ValueError(f"Unknown normalization method: {method}")

# --- Analyze Signal Function ---
def analyze_signal(data, name):
    """Analyze signal characteristics."""
    data_calc = data.astype(np.float64)
    peak = np.max(np.abs(data_calc))
    if data_calc.size == 0 or np.all(data_calc == 0):
        rms = 0.0
        mean_abs_val = 0.0
    else:
        rms = np.sqrt(np.mean(data_calc**2))
        mean_abs_val = np.mean(np.abs(data_calc))

    crest_factor = peak / rms if rms > 1e-9 else float('inf')
    dynamic_range = 20 * np.log10(peak / mean_abs_val) if mean_abs_val > 1e-9 and peak > 1e-9 else -float('inf')

    print(f"\n{name} Signal Analysis:")
    print(f"Peak level: {peak}")
    print(f"RMS level: {rms:.2f}")
    print(f"Crest factor: {crest_factor:.2f}")
    print(f"Dynamic range: {dynamic_range:.1f} dB")
    return peak, rms, crest_factor

# --- Main Alignment Function (Using Corrected SciPy Logic Interpretation) ---
def find_best_alignment(render_data_raw, capture_data_raw, max_latency_samples_param, current_sample_rate, current_normalization_method):
    print("\n--- Finding Best Alignment (SciPy Correlate - Corrected Logic) ---")
    render_norm = normalize_signal_robust(render_data_raw, current_normalization_method, "Render")
    capture_norm = normalize_signal_robust(capture_data_raw, current_normalization_method, "Capture")

    if render_norm.size == 0 or capture_norm.size == 0:
        return {'latency_samples': -1, 'error': 'Normalization failed'}
    
    # print(f"Length of render_norm: {len(render_norm)}") # Optional debug
    # print(f"Length of capture_norm: {len(capture_norm)}") # Optional debug
    
    correlation = signal.correlate(capture_norm, render_norm, mode='full', method='fft')
    # print(f"Length of full correlation array: {len(correlation)}") # Optional debug
    
    start_idx_true_zero_lag = len(render_norm) - 1
    # print(f"Search starts at TRUE zero-lag index: {start_idx_true_zero_lag}") # Optional debug

    end_idx_search = min(start_idx_true_zero_lag + max_latency_samples_param + 1, len(correlation))
    # print(f"Search window in correlation: indices [{start_idx_true_zero_lag} : {end_idx_search}]") # Optional debug
    
    search_range = correlation[start_idx_true_zero_lag : end_idx_search]
    latency_samples = -1; best_correlation_value = -np.inf; error_desc = None

    if search_range.size == 0:
        error_desc = "Search range empty"
    else:
        latency_samples = np.argmax(search_range) # This IS the latency
        best_correlation_value = search_range[latency_samples]
        # peak_idx_full_corr = start_idx_true_zero_lag + latency_samples # Optional debug
        # print(f"Peak index in search_range: {latency_samples}, Peak index in full_correlation: {peak_idx_full_corr}")

    results = {'latency_samples': latency_samples, 'correlation': best_correlation_value, 'error': error_desc}
    if latency_samples != -1:
        results['latency_seconds'] = latency_samples / current_sample_rate
        results['latency_ms'] = results['latency_seconds'] * 1000.0
        aligned_length = min(len(render_norm), len(capture_norm) - latency_samples)
        results['aligned_length'] = aligned_length
        if aligned_length > 0 and latency_samples >= 0 and latency_samples < len(capture_norm):
            aligned_render = render_norm[:aligned_length]; results['aligned_render'] = aligned_render
            aligned_capture = capture_norm[latency_samples : latency_samples + aligned_length]; results['aligned_capture'] = aligned_capture
            results['mse'] = np.mean((aligned_render - aligned_capture) ** 2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                corr_matrix = np.corrcoef(aligned_render, aligned_capture)
            if corr_matrix.shape == (2, 2) and not np.isnan(corr_matrix[0, 1]):
                results['correlation_coef'] = corr_matrix[0, 1]
            else:
                results['correlation_coef'] = 0.0 if np.std(aligned_render) < 1e-9 or np.std(aligned_capture) < 1e-9 else np.nan
        else:
            results.update({'mse': np.nan, 'correlation_coef': np.nan, 'aligned_render':np.array([]), 'aligned_capture':np.array([])})
    return results

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(
        description='Calculate audio latency using SciPy cross-correlation with corrected logic.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--render', '-r', type=str, required=True,
                        help='Path to the render PCM file (int16)')
    parser.add_argument('--capture', '-c', type=str, required=True,
                        help='Path to the capture PCM file (int16)')
    parser.add_argument('--sample-rate', '-s', type=int, default=DEFAULT_SAMPLE_RATE,
                        help='Sample rate in Hz')
    parser.add_argument('--max-latency', '-m', type=float, default=1.0,
                        help='Maximum latency to consider in seconds')
    parser.add_argument('--normalization', choices=['none', 'peak', 'rms', 'zscore'],
                        default='zscore', help='Signal normalization method')
    parser.add_argument('--output', '-o', type=str, default=None, # Default is None
                        required=False, help='Output image path (e.g., latency_analysis.png). If not specified, no plot is generated.')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for output image')
    parser.add_argument('--latency_file', type=str, default=None,
                        required=False, help='Path to output file for calculated latency in ms (plain text).')
    # --latency_old_file argument is removed
    args = parser.parse_args()

    # --- Set Variables ---
    sample_rate = args.sample_rate
    render_file = args.render
    capture_file = args.capture
    script_start_time = time.time()

    # --- 1. Load PCM Data ---
    print("--- Loading PCM files ---")
    try:
        render_data_raw = np.fromfile(render_file, dtype=PCM_DTYPE)
        capture_data_raw = np.fromfile(capture_file, dtype=PCM_DTYPE)
        print(f"Loaded {len(render_data_raw)} samples from {render_file}")
        print(f"Loaded {len(capture_data_raw)} samples from {capture_file}")
    except FileNotFoundError:
        print(f"Error: One or both files not found.\nRender: '{render_file}'\nCapture: '{capture_file}'")
        exit(1)
    except Exception as e:
        print(f"An error occurred during file loading: {e}")
        exit(1)

    # --- Optional: Initial Signal Analysis ---
    render_rms_val, capture_rms_val = 0.0, 0.0 # Initialize
    if len(render_data_raw) > 0 and len(capture_data_raw) > 0:
        _, render_rms_val, _ = analyze_signal(render_data_raw, "Render Raw")
        _, capture_rms_val, _ = analyze_signal(capture_data_raw, "Capture Raw")

    # --- 2. Calculate Max Latency Samples ---
    max_latency_samples = int(args.max_latency * sample_rate)
    print(f"Max latency to check: {args.max_latency} s ({max_latency_samples} samples)")

    # --- 3. Run The Main Alignment Method ---
    result = find_best_alignment(render_data_raw, capture_data_raw, max_latency_samples, args.sample_rate, args.normalization)

    # --- 4. Print Summary Results ---
    print("\n--- Alignment Results ---")
    if result and result.get('latency_samples', -1) != -1:
        print(f"Latency: {result['latency_ms']:.2f} ms ({result['latency_samples']} samples)")
        print(f"Correlation coefficient: {result.get('correlation_coef', float('nan')):.4f}")
        print(f"Mean squared error: {result.get('mse', float('nan')):.4f}")
    else:
        print(f"Alignment Failed: {result.get('error', 'Unknown error')}")

    # --- 5. Dump Corrected Latency to File if Argument Provided ---
    if args.latency_file and result and result.get('latency_samples', -1) != -1:
        try:
            # Ensure the directory for latency_file exists
            latency_file_dir = os.path.dirname(args.latency_file)
            if latency_file_dir and not os.path.exists(latency_file_dir):
                os.makedirs(latency_file_dir)
            with open(args.latency_file, 'w') as f:
                f.write(f"{result['latency_ms']:.2f}\n")
            print(f"\nCalculated latency ({result['latency_ms']:.2f} ms) written to {args.latency_file}")
        except Exception as e:
            print(f"Error writing latency to file '{args.latency_file}': {e}")

    # --- 6. Conditional Visualization (3 subplots) ---
    if args.output is not None:
        print("\nGenerating visualization...")
        if not (result and result.get('latency_samples', -1) != -1):
            print("Alignment method failed. Skipping visualization.")
        else:
            fig, axs = plt.subplots(3, 1, figsize=(15, 15), sharex=False) # 3 subplots now

            # Plot 1: Original raw signals
            axs[0].plot(np.arange(len(render_data_raw)) / args.sample_rate, render_data_raw, label=f'Render (Raw RMS: {render_rms_val:.0f})', alpha=0.7)
            axs[0].plot(np.arange(len(capture_data_raw)) / args.sample_rate, capture_data_raw, label=f'Capture (Raw RMS: {capture_rms_val:.0f})', alpha=0.7)
            axs[0].set_title('Original Raw Signals'); axs[0].set_xlabel('Time (seconds)'); axs[0].set_ylabel(f'Amplitude ({PCM_DTYPE})'); axs[0].legend(); axs[0].grid(True)

            # Plot 2: Full Cross-Correlation and Peak
            # Re-normalize and re-correlate once for this plot to get the full correlation array
            render_norm_plot = normalize_signal_robust(render_data_raw, args.normalization, "PlotRender")
            capture_norm_plot = normalize_signal_robust(capture_data_raw, args.normalization, "PlotCapture")

            if render_norm_plot.size > 0 and capture_norm_plot.size > 0:
                correlation_full = signal.correlate(capture_norm_plot, render_norm_plot, mode='full', method='fft')
                scipy_zero_lag_idx = len(render_norm_plot) - 1
                lags_plot_axis = np.arange(len(correlation_full)) - scipy_zero_lag_idx
                
                axs[1].plot(lags_plot_axis, correlation_full, label='Full Cross-Correlation (SciPy)', alpha=0.7, color='green', linewidth=0.8)
                axs[1].set_title('Full Cross-Correlation Function and Detected Peak'); axs[1].set_xlabel('Lag (samples)'); axs[1].set_ylabel('Correlation Value'); axs[1].grid(True)

                lag_corrected_method = result['latency_samples'] # This is already relative to true zero-lag
                axs[1].plot(lag_corrected_method, result['correlation'], 'rx', markersize=10, markeredgewidth=2, label=f"Detected Peak (Lag: {lag_corrected_method}, {result['latency_ms']:.2f}ms)")
                axs[1].axvline(lag_corrected_method, color='red', linestyle='--', alpha=0.7)
                
                plot_min_lag_display = max(lags_plot_axis[0], lag_corrected_method - int(max_latency_samples * 0.2))
                plot_max_lag_display = min(lags_plot_axis[-1], lag_corrected_method + int(max_latency_samples * 0.2))
                axs[1].set_xlim(plot_min_lag_display, plot_max_lag_display) # Focus around the peak
                axs[1].legend()
            else:
                axs[1].text(0.5, 0.5, "Correlation plot not available.", ha='center', va='center')

            # Plot 3: Aligned signals (Corrected Logic)
            if result.get('aligned_length', 0) > 0:
                time_aligned_corr = np.arange(result['aligned_length']) / args.sample_rate
                axs[2].plot(time_aligned_corr, result['aligned_render'], label='Render (Norm)', alpha=0.7)
                axs[2].plot(time_aligned_corr, result['aligned_capture'], label=f"Capture (Norm, shift: {result['latency_ms']:.1f}ms)", alpha=0.7)
                axs[2].set_title(f"Aligned Signals (Latency: {result['latency_ms']:.2f} ms)"); axs[2].set_xlabel('Time (s)'); axs[2].set_ylabel('Norm. Amplitude'); axs[2].legend(); axs[2].grid(True)
            else:
                axs[2].text(0.5, 0.5, "No valid overlap for plotting aligned signals.", ha='center', va='center'); axs[2].set_title("Aligned Signals")

            fig.suptitle(f'Latency Analysis (Corrected SciPy Logic, Norm: {args.normalization})\nLatency: {result["latency_ms"]:.2f} ms, Correlation Coef: {result.get("correlation_coef", float("nan")):.4f}',
                          fontsize=16, y=0.99)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir): # Ensure output directory exists
                os.makedirs(output_dir)
            plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
            print(f"\nVisualization saved to: {args.output}")
            plt.close(fig)
    else:
        print("\nSkipping visualization as --output file was not specified or alignment failed.")

    script_end_time = time.time()
    print("-" * 30)
    print(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds.")
    print("="*30)
