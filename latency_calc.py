#!/usr/bin/env python3
# File: compare_alignment_logics_v2.py

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
DEFAULT_SEGMENT_DURATION_S = 5.0 # Though not used if calling corrected_logic directly

# --- Robust Normalization Function (used by both alignment functions) ---
def normalize_signal_robust(data_raw, method='zscore'):
    """Normalizes raw data (e.g., int16) to float64."""
    if not isinstance(data_raw, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if data_raw.size == 0:
        # print(f"[Normalize] Warning: Normalizing empty array ({method}).") # Reduce verbosity
        return np.array([], dtype=np.float64)

    data_dbl = data_raw.astype(np.float64, copy=False)
    # print(f"[Normalize] Normalizing {data_raw.size} samples using '{method}' method...")

    if method == 'zscore':
        mean = np.mean(data_dbl)
        std = np.std(data_dbl)
        if std < 1e-9:
            # print(f"[Normalize] Warning: Standard deviation is near zero ({std:.2e}). Normalizing to array of zeros.")
            return np.zeros_like(data_dbl, dtype=np.float64)
        else:
            return (data_dbl - mean) / std
    elif method == 'peak':
        peak = np.max(np.abs(data_dbl))
        if peak < 1e-9:
             # print("[Normalize] Warning: Peak amplitude is near zero. Normalizing to array of zeros.")
             return np.zeros_like(data_dbl, dtype=np.float64)
        else:
             return data_dbl / peak
    elif method == 'rms':
         rms_val = np.sqrt(np.mean(data_dbl**2))
         if rms_val < 1e-9:
              # print("[Normalize] Warning: RMS level is near zero. Normalizing to array of zeros.")
              return np.zeros_like(data_dbl, dtype=np.float64)
         else:
              return data_dbl / rms_val
    elif method == 'none':
         # print("[Normalize] Method is 'none', returning data as float64.")
         return data_dbl
    else:
        raise ValueError(f"Unknown normalization method: {method}")

# --- Analyze Signal Function (for general stats) ---
def analyze_signal(data, name):
    """Analyze signal characteristics."""
    data_calc = data.astype(np.float64)
    peak = np.max(np.abs(data_calc))
    
    if np.all(data_calc == 0): rms = 0.0
    else: rms = np.sqrt(np.mean(data_calc**2))

    crest_factor = peak / rms if rms > 1e-9 else float('inf')
    mean_abs_val = np.mean(np.abs(data_calc))
    dynamic_range = 20 * np.log10(peak / mean_abs_val) if mean_abs_val > 1e-9 and peak > 1e-9 else -float('inf')

    print(f"\n{name} Signal Analysis:")
    print(f"Peak level: {peak}")
    print(f"RMS level: {rms:.2f}")
    print(f"Crest factor: {crest_factor:.2f}")
    print(f"Dynamic range: {dynamic_range:.1f} dB")
    return peak, rms, crest_factor

# --- Alignment Function (Corrected Logic for SciPy) ---
def find_best_alignment_corrected_logic(render_data_raw, capture_data_raw, max_latency_samples_param, current_sample_rate, current_normalization_method):
    print("\n--- Finding Alignment (Corrected SciPy Logic Interpretation) ---")
    render_norm = normalize_signal_robust(render_data_raw, current_normalization_method)
    capture_norm = normalize_signal_robust(capture_data_raw, current_normalization_method)

    if render_norm.size == 0 or capture_norm.size == 0: return {'latency_samples': -1, 'error': 'Normalization failed (Corrected Logic)'}
    
    correlation = signal.correlate(capture_norm, render_norm, mode='full', method='fft')
    
    start_idx_true_zero_lag = len(render_norm) - 1
    end_idx_search = min(start_idx_true_zero_lag + max_latency_samples_param + 1, len(correlation))
    
    search_range = correlation[start_idx_true_zero_lag : end_idx_search]
    latency_samples = -1; best_correlation_value = -np.inf; error_desc = None

    if search_range.size == 0: error_desc = "Search range empty (Corrected Logic)"
    else: latency_samples = np.argmax(search_range); best_correlation_value = search_range[latency_samples]

    results = {'latency_samples': latency_samples, 'correlation': best_correlation_value, 'error': error_desc}
    if latency_samples != -1:
        results['latency_seconds'] = latency_samples / current_sample_rate
        results['latency_ms'] = results['latency_seconds'] * 1000.0
        aligned_length = min(len(render_norm), len(capture_norm) - latency_samples)
        results['aligned_length'] = aligned_length
        if aligned_length > 0 and latency_samples < len(capture_norm):
            aligned_render = render_norm[:aligned_length]; results['aligned_render'] = aligned_render
            aligned_capture = capture_norm[latency_samples : latency_samples + aligned_length]; results['aligned_capture'] = aligned_capture
            results['mse'] = np.mean((aligned_render - aligned_capture) ** 2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                corr_matrix = np.corrcoef(aligned_render, aligned_capture)
            if corr_matrix.shape == (2, 2) and not np.isnan(corr_matrix[0, 1]): results['correlation_coef'] = corr_matrix[0, 1]
            else: results['correlation_coef'] = 0.0 if np.std(aligned_render) < 1e-9 or np.std(aligned_capture) < 1e-9 else np.nan
        else: results.update({'mse': np.nan, 'correlation_coef': np.nan, 'aligned_render':np.array([]), 'aligned_capture':np.array([]) })
    return results


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(
        description='Calculate audio latency comparing two scipy.signal.correlate interpretations.',
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
    # Modified --output argument
    parser.add_argument('--output', '-o', type=str, default=None, # Default is None
                        required=False, help='Output image path (e.g., latency_comparison.png). If not specified, no plot is generated.')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for output image')
    # New argument for latency file
    parser.add_argument('--latency_file', type=str, default=None,
                        required=False, help='Path to output file for corrected latency in ms (plain text).')

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
    if len(render_data_raw) > 0 and len(capture_data_raw) > 0:
        render_peak_orig, render_rms_orig, _ = analyze_signal(render_data_raw, "Render Raw")
        capture_peak_orig, capture_rms_orig, _ = analyze_signal(capture_data_raw, "Capture Raw")
    else:
        render_rms_orig, capture_rms_orig = 0,0 # Defaults if signals are empty

    # --- 2. Calculate Max Latency Samples ---
    max_latency_samples = int(args.max_latency * sample_rate)
    print(f"Max latency to check: {args.max_latency} s ({max_latency_samples} samples)")

    # --- 3. Run Both Alignment Methods ---
    result_corrected = find_best_alignment_corrected_logic(render_data_raw, capture_data_raw, max_latency_samples, args.sample_rate, args.normalization)

    # --- 4. Print Summary Results ---

    if result_corrected and result_corrected.get('latency_samples', -1) != -1:
        print(f"Corrected Logic Latency: {result_corrected['latency_ms']:.2f} ms ({result_corrected['latency_samples']} samples), CorrCoef: {result_corrected.get('correlation_coef', float('nan')):.4f}, MSE: {result_corrected.get('mse', float('nan')):.4f}")
    else:
        print(f"Corrected Logic Failed: {result_corrected.get('error', 'Unknown error')}")

    # --- 5. Dump Corrected Latency to File if Argument Provided ---
    if args.latency_file and result_corrected and result_corrected.get('latency_samples', -1) != -1:
        try:
            with open(args.latency_file, 'w') as f:
                f.write(f"{result_corrected['latency_ms']:.2f}\n") # Write latency in ms
            print(f"\nCorrected latency ({result_corrected['latency_ms']:.2f} ms) written to {args.latency_file}")
        except Exception as e:
            print(f"Error writing latency to file '{args.latency_file}': {e}")

    # --- 6. Conditional Visualization ---
    if args.output is not None: # Check if --output was specified
        print("\nGenerating combined visualization...")
        if not (result_corrected and result_corrected.get('latency_samples', -1) != -1):
            print("One or both alignment methods failed. Skipping full visualization.")
        else:
            fig, axs = plt.subplots(4, 1, figsize=(15, 20), sharex=False) # Increased height for 4 plots

            # Plot 1: Original raw signals
            axs[0].plot(np.arange(len(render_data_raw)) / args.sample_rate, render_data_raw, label=f'Render (Raw RMS: {render_rms_orig:.0f})', alpha=0.7)
            axs[0].plot(np.arange(len(capture_data_raw)) / args.sample_rate, capture_data_raw, label=f'Capture (Raw RMS: {capture_rms_orig:.0f})', alpha=0.7)
            axs[0].set_title('Original Raw Signals'); axs[0].set_xlabel('Time (seconds)'); axs[0].set_ylabel(f'Amplitude ({PCM_DTYPE})'); axs[0].legend(); axs[0].grid(True)

            # Plot 2: Full Cross-Correlation and Peaks
            render_norm_plot = normalize_signal_robust(render_data_raw, args.normalization)
            capture_norm_plot = normalize_signal_robust(capture_data_raw, args.normalization)
            if render_norm_plot.size > 0 and capture_norm_plot.size > 0:
                correlation_full = signal.correlate(capture_norm_plot, render_norm_plot, mode='full', method='fft')
                scipy_zero_lag_idx = len(render_norm_plot) - 1
                lags_plot_axis = np.arange(len(correlation_full)) - scipy_zero_lag_idx
                
                axs[1].plot(lags_plot_axis, correlation_full, label='Full Cross-Correlation (SciPy)', alpha=0.5, color='grey', linewidth=0.8)
                axs[1].set_title('Full Cross-Correlation Function and Detected Peaks'); axs[1].set_xlabel('Lag (samples)'); axs[1].set_ylabel('Correlation Value'); axs[1].grid(True)

                # Mark corrected logic peak
                lag_corrected_method = result_corrected['latency_samples']
                axs[1].plot(lag_corrected_method, result_corrected['correlation'], 'rx', markersize=10, markeredgewidth=2, label=f"Corrected Logic Peak (Lag: {lag_corrected_method}, {result_corrected['latency_ms']:.2f}ms)")
                axs[1].axvline(lag_corrected_method, color='red', linestyle=':', alpha=0.7)
                
                plot_min_lag_display = lag_corrected_method - int(max_latency_samples * 0.1)
                plot_max_lag_display = lag_corrected_method + int(max_latency_samples * 0.1)
                plot_min_lag_display = max(plot_min_lag_display, lags_plot_axis[0]) # Ensure within bounds
                plot_max_lag_display = min(plot_max_lag_display, lags_plot_axis[-1])# Ensure within bounds
                axs[1].set_xlim(plot_min_lag_display, plot_max_lag_display)
                axs[1].legend()
            else:
                axs[1].text(0.5, 0.5, "Correlation plot not available.", ha='center', va='center')

            # Plot 3: Aligned signals (Corrected Logic)
            if result_corrected.get('aligned_length', 0) > 0:
                time_aligned_corr = np.arange(result_corrected['aligned_length']) / args.sample_rate
                axs[3].plot(time_aligned_corr, result_corrected['aligned_render'], label='Render (Norm)', alpha=0.7)
                axs[3].plot(time_aligned_corr, result_corrected['aligned_capture'], label=f"Capture (Norm, shift: {result_corrected['latency_ms']:.1f}ms)", alpha=0.7)
                axs[3].set_title(f"Aligned by Corrected Logic (Latency: {result_corrected['latency_ms']:.2f} ms)"); axs[3].set_xlabel('Time (s)'); axs[3].set_ylabel('Norm. Amplitude'); axs[3].legend(); axs[3].grid(True)
            else:
                axs[3].text(0.5, 0.5, "Corrected logic: No valid overlap for plotting.", ha='center', va='center'); axs[3].set_title("Aligned by Corrected Logic")

            fig.suptitle(f'Comparison of Alignment Logics (Normalization: {args.normalization})', fontsize=16, y=0.99)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
            print(f"\nCombined visualization saved to: {args.output}")
            plt.close(fig)
    else:
        print("\nSkipping visualization as --output file was not specified or alignment failed.")


    script_end_time = time.time()
    print("-" * 30)
    print(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds.")
    print("="*30) 
