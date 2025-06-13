# SoundDelay - Audio Latency Measurement Tool

This tool calculates the audio latency between a rendered audio signal and its captured playback. It achieves this by finding the best global alignment between the two signals using cross-correlation.

## Overview

The tool is designed to measure system audio latency by:
1. Analyzing a reference audio signal (e.g., `render.pcm`).
2. Comparing it with a recorded version of the same signal (e.g., `capture.pcm`) which has passed through the system under test.
3. Computing the time delay (latency) between them using an FFT-based cross-correlation technique.
4. Optionally generating detailed visualizations of the analysis and outputting latency values to a file.

### Sample Analysis Output
![Sample Latency Analysis](samples/set1/latency_corelation.jpg)

The visualization typically shows:
- **Top**: Original render and capture signals (raw amplitude).
- **Middle**: The full cross-correlation function, with the detected peak lag highlighted.
- **Bottom**: Render and capture signals (normalized) aligned according to the calculated latency, along with their difference.

## Latency Measurement Principle: Cross-Correlation

This script determines the time delay (latency) between the `render` and `capture` signals using the principle of **cross-correlation**.

1.  **What is Cross-Correlation?**
    Cross-correlation is a measure of similarity between two series (signals) as a function of the displacement (lag or time shift) of one relative to the other. Essentially, one signal is "slid" past the other, and for each possible shift, a similarity score (the correlation value) is calculated by multiplying the overlapping sample values and summing them up. The lag that results in the maximum correlation value is considered the time offset at which the two signals are most similar.

2.  **Application to Latency Measurement:**
    * The `render.pcm` file contains the original audio signal that was played out or sent into the system.
    * The `capture.pcm` file contains the audio recorded after it has passed through the system under test (e.g., played out of speakers and recorded by a microphone, or passed through an audio processing chain). This captured signal will include any delays introduced by the system.
    * By cross-correlating the `capture` signal with the `render` signal, we can find the time shift (lag) needed to align the captured audio event with the original rendered audio event. This shift is the measured latency.

3.  **Steps in `latency_calc.py`:**
    * **Load Data:** Raw PCM audio data is loaded from the `render.pcm` and `capture.pcm` files into NumPy arrays.
    * **Normalization (Pre-processing):** Before correlation, both signals are typically normalized. By default, "z-score" normalization is used, which scales each signal to have a mean of 0 and a standard deviation of 1. This is important because:
        * It makes the correlation focus on the *shape* and *timing* similarity of the signals, rather than being skewed by differences in absolute amplitude or recording levels.
        * It provides a consistent basis for calculating quality metrics like Mean Squared Error (MSE).
    * **Cross-Correlation Calculation:**
        * The script uses the `scipy.signal.correlate()` function with `mode='full'` and `method='fft'`.
        * `method='fft'` indicates that the cross-correlation is computed efficiently in the frequency domain using the Fast Fourier Transform (FFT). The underlying principle is:
            1.  Both normalized signals (`capture_norm` and `render_norm`) are transformed into the frequency domain using FFT:
                `F_capture = FFT(capture_norm)`
                `F_render = FFT(render_norm)`
            2.  In the frequency domain, one transformed signal's spectrum is multiplied by the *complex conjugate* of the other's spectrum:
                `Correlation_In_FrequencyDomain = F_capture * conj(F_render)`
                (where `conj()` denotes the complex conjugate).
            3.  This product is then transformed back to the time domain using an Inverse FFT (IFFT):
                `Correlation_In_TimeDomain = IFFT(Correlation_In_FrequencyDomain)`
            * The resulting `Correlation_In_TimeDomain` vector contains the cross-correlation values for each possible lag.
    * **Peak Detection and Latency Calculation:**
        * The `signal.correlate(capture_norm, render_norm, mode='full')` output array has its "zero lag" point (representing no time shift between the conceptual start of `capture_norm` and `render_norm` for the correlation sum) at index `len(render_norm) - 1`.
        * The script searches for the maximum value in the `Correlation_In_TimeDomain` vector. This search is performed over a window that starts at this true zero-lag index and extends forward up to the `max_latency_samples` specified by the user.
        * The position (index) of this maximum value, *relative to the start of this search window (the true zero-lag point)*, directly gives the `latency_samples`.
        * This latency in samples is then converted to milliseconds using the provided sample rate.
    * **Alignment Quality Metrics:**
        * Once the latency is determined, the script calculates the Mean Squared Error (MSE) and the Pearson Correlation Coefficient between the `render_norm` signal and the `capture_norm` signal (shifted by the calculated latency). These metrics help assess the quality and reliability of the alignment.

## Features

- Accurate latency measurement using full signal cross-correlation (via FFT).
- Multiple normalization methods (z-score, peak, RMS, none) to adapt to different signal characteristics.
- Calculation of alignment quality metrics: Pearson Correlation Coefficient and Mean Squared Error.
- Optional detailed visualization of raw signals, the full cross-correlation function with detected peak, and aligned signals.
- Support for raw 16-bit signed integer PCM format.
- Configurable maximum latency search range.
- Configurable sample rate.
- Option to save calculated latency to a text file.

## Input File Format

### PCM File Requirements

- **Format**: Raw PCM (Pulse Code Modulation)
- **Encoding**: Signed 16-bit integer (`np.int16`)
- **Sample Rate**: Configurable via `--sample-rate` (default: 16000 Hz)
- **Channels**: Mono (current processing assumes and is optimized for mono signals)
- **Container**: Raw PCM (no header, i.e., `.pcm` or `.raw`)

### Input Files

1.  **render.pcm (or as specified by `-r`)**
    * The reference audio signal that was played out or sent into the system.
    * Should contain clear, distinct audio features or events for robust correlation.
    * Examples: Clicks, beeps, specific test tones, or a short, unique audio segment.
    * File size = (Duration in seconds) × (Sample Rate) × 2 bytes (for 16-bit mono).

2.  **capture.pcm (or as specified by `-c`)**
    * The audio signal recorded after passing through the system under test.
    * This signal contains the same audio events as the render signal but includes system-induced latency and potentially noise or distortion.
    * Should ideally be longer than the render signal to ensure the delayed render signal is fully captured.
    * Must use the same sample rate as the render signal.

### Creating Input Files

1.  **Using ALSA (Linux)**
    ```bash
    # Example: Generate a 2-second 1kHz sine wave test tone
    ffmpeg -f lavfi -i "sine=frequency=1000:duration=2" -f s16le -acodec pcm_s16le -ar 16000 -ac 1 render.pcm

    # Example: Record microphone input to capture.pcm (adjust device hw:X,Y as needed)
    # Ensure you play render.pcm during this recording, or have a loopback setup.
    arecord -D hw:1,0 -f S16_LE -r 16000 -c 1 -d 5 capture.pcm
    ```

2.  **Using FFmpeg**
    ```bash
    # Convert from other audio formats to the required PCM format
    ffmpeg -i input_audio.wav -f s16le -acodec pcm_s16le -ar 16000 -ac 1 output.pcm
    ```

## Output Files

### 1. Numerical Output (Console)
The script prints to the console:
- Analysis of the raw render and capture signals (Peak, RMS, Crest Factor, Dynamic Range).
- Key parameters used for alignment (max latency, normalization method).
- The calculated latency in milliseconds and samples.
- The Pearson Correlation Coefficient of the aligned signals.
- The Mean Squared Error (MSE) of the aligned signals.
- The peak value from the cross-correlation function.
- The length of the aligned segment.

Example:

```bash
--- Alignment Results ---
Latency:              428.25 ms (6852 samples)
Correlation coefficient: 0.1766
Mean squared error:   1.6675
Peak Correlation Val: 88927.7249 (Example value)
Aligned Length:       497148 samples
```


### 2. Latency Value File (Optional)
- If `--latency_file <path/to/latency.txt>` is provided, the script saves the calculated latency (in milliseconds, plain text, formatted to 2 decimal places) to the specified file.

### 3. Visualization (Optional PNG)
- If `--output <path/to/image.png>` is provided, a PNG image visualizing the analysis is saved.
- **Filename & DPI**: Configurable via `--output` and `--dpi` arguments.
- The visualization includes three subplots:
    1.  Original raw render and capture signals over time.
    2.  The full cross-correlation function, with the detected peak lag highlighted. The x-axis is focused around the detected peak.
    3.  The normalized render and capture signals, aligned according to the calculated latency, along with their difference.

## Use Cases

- Measure end-to-end audio system latency (acoustic or digital).
- Test and compare different audio configurations or devices.
- Benchmark audio drivers or processing pipelines.
- Verify audio system specifications in QA.
- Aid in research and development of audio systems.

## Example Workflows

### 1. Basic Latency Test
```bash
# 1. Generate a test signal (e.g., 2-second chirp)
# (Using FFmpeg for a chirp - more robust for correlation than a simple sine)
ffmpeg -f lavfi -i "sine=f=1000:d=2,volume=0.5[s1];sine=f=5000:d=2,volume=0.5[s2];amix=inputs=2:duration=shortest,asettb=expr='sin(2*PI*t*(100+t*1000))',volume=0.5" -f s16le -acodec pcm_s16le -ar 16000 -ac 1 render.pcm

# 2. Play render.pcm and simultaneously record it through the system to capture.pcm
# (This step is system-dependent and might involve loopback or physical speaker/mic setup)
# Ensure capture.pcm is long enough and starts recording before or around render playback.

# 3. Analyze latency, generate plot, and save latency value
python latency_calc.py -r render.pcm -c capture.pcm --output result_plot.png --latency_file result_latency.txt --sample-rate 16000
```

### 2. Batch Processing Multiple Folders

Create a bash script (e.g., `run_all_latency_tests.sh`):

```bash
#!/bin/bash
BASE_DIR="path/to_your_experiment_folders" # Modify this path
PYTHON_SCRIPT="latency_calc.py"         # Path to your latency_calc.py

find "$BASE_DIR" -mindepth 1 -type d -print0 | while IFS= read -r -d $'\0' folder_path; do
    render_file="$folder_path/render.pcm"
    capture_file="$folder_path/capture.pcm"
    # Use fixed output names inside each folder
    output_image="$folder_path/latency_correlation_plot.png"
    latency_txt="$folder_path/latency_ms.txt"

    if [ -f "$render_file" ] && [ -f "$capture_file" ]; then
        echo "Processing $folder_path..."
        python "$PYTHON_SCRIPT" \
            -r "$render_file" \
            -c "$capture_file" \
            --output "$output_image" \
            --latency_file "$latency_txt" \
            # Add other consistent arguments like --sample-rate, --max-latency, --normalization
            # e.g., --sample-rate 16000 --normalization zscore --max-latency 1.0
    else
        echo "Skipping $folder_path: PCM files not found."
    fi
done
```

Make it executable (`chmod +x run_all_latency_tests.sh`) and run it.

#### Requirements
- Python 3.x
- NumPy
- SciPy
- Matplotlib (only if generating plots via the --output argument)

Install dependencies:

```bash
pip install numpy scipy matplotlib
```

## Usage

### Basic Usage (calculates latency, prints to console)

```bash
python latency_calc.py -r path/to/render.pcm -c path/to/capture.pcm
```

### Generating Outputs

```bash
python latency_calc.py \
    -r render.pcm \
    -c capture.pcm \
    --sample-rate 16000 \
    --max-latency 0.5 \
    --normalization zscore \
    --output analysis_plot.png \
    --dpi 300 \
    --latency_file latency_ms.txt
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|--------|---------|-------------|
| `--render` | `-r` | `render.pcm` | Path to the render PCM file |
| `--capture` | `-c` | `capture.pcm` | Path to the capture PCM file |
| `--sample-rate` | `-s` | 16000 | Sample rate in Hz |
| `--max-latency` | `-m` | `1.0` | Maximum latency to consider (seconds) |
| `--normalization` | - | `zscore` | Signal normalization method |
| `--output` | `-o` | None | Output image path |
| `--dpi` | - | `300` | DPI for output image  (if generated) |
| `--latency_file` | - | None | Path to save the calculated latency (in ms) to a text file. If not provided, latency is not saved to a file. |

### Normalization Methods

- `none`: Uses raw signal values (internally cast to float64 for processing).
- `peak`: Normalizes by dividing each signal by its absolute maximum value. Resulting values range from -1 to 1.
- `rms`: Normalizes by dividing each signal by its Root Mean Square (RMS) value.
- `zscore`: Normalizes by subtracting the mean and dividing by the standard deviation. Results in a signal with a mean of approximately 0 and a standard deviation of approximately 1 (default method).

## Best Practices

1. **Signal Quality for `render.pcm`:**
   -  Use signals with sharp onsets or distinct features (e.g., clicks, short beeps, chirps) rather than continuous tones for more precise correlation peaks.
   -  Ensure the render.pcm signal is clean and has good amplitude without clipping.

2. **Recording `capture.pcm`:**
   - Minimize background noise during capture.
   - Ensure the recording level is adequate to capture the render signal clearly, but avoid clipping.
   -  The capture should start slightly before the render signal is expected and end after it, plus the maximum expected latency.

3. **Consistent Parameters:** Ensure the sample rate used in the script matches the actual sample rate of your PCM files.

4. **Verification:**
   -  Always check the Correlation Coefficient and MSE values. A high correlation coefficient (close to 1) and low MSE suggest a good quality alignment.
   -  Visually inspect the generated plot if available, especially the aligned signals and the peak in the correlation function.
   -  Perform multiple measurements if possible to check for consistency.


## Troubleshooting

### Common Issues
1. **Poor Correlation / Unreliable Latency:**

   **Signal Content:** The render.pcm might not have distinct enough features. Try a click, chirp, or short burst of white noise. Repetitive signals can lead to multiple ambiguous peaks.

   **Normalization:** Experiment with different --normalization methods if zscore isn't working well for your specific signals.

   **Noise:** High noise in capture.pcm can obscure the correlation peak.

   **Non-linearity:** If the system introduces significant non-linear distortion, the captured signal might not resemble the rendered one closely enough.

   **Clipping:** If either signal is clipped, its waveform is distorted, which will affect correlation.

   **Latency Value Unexpectedly High or Low:**

   -  `--max-latency`: Ensure this value is large enough to cover the true latency. If the true latency is outside this search window, the reported peak will be the best match within the window, not necessarily the global best.
   -  **File Mismatch:** Double-check that render.pcm and capture.pcm are indeed the correct corresponding files.
   -  **Sample Rate:** Verify the --sample-rate argument matches the actual sample rate of your files.

### Error Messages
   - **"File not found"**: Verify the paths provided for --render and --capture files.
   - **"ValueError: Unknown normalization method"**: Check the spelling of the method passed to --normalization.
   - **"Search range empty"**: This can happen if max_latency_samples is too small relative to the signal processing or if there's an issue with correlation output length. Ensure max_latency (in seconds) is appropriate.

## Contributing
Feel free to submit issues, bug reports, and enhancement requests via the project's issue tracker. Pull requests are also welcome!

## License
```
MIT License

Copyright (c) 2024 LatencyLens (Or your name/organization)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```