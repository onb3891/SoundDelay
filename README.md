# SoundDelay - Audio Latency Measurement Tool

This tool calculates the audio latency between a rendered audio signal and its captured playback by finding the best global alignment between the two signals.

## Overview

The tool is designed to measure system audio latency by:
1. Analyzing a reference audio signal (render.pcm)
2. Comparing it with a recorded version of the same signal (capture.pcm)
3. Computing the time delay between them using cross-correlation
4. Generating detailed visualizations of the analysis

### Sample Analysis Output
![Sample Latency Analysis](samples/set1/latency_corelation.jpg)

The visualization above shows:
- **Top**: Original render (blue) and capture (orange) signals
- **Middle**: Signals after alignment, showing how capture signal is shifted to match render
- **Bottom**: Difference plot showing alignment quality

## Features

- Accurate latency measurement using full signal correlation
- Multiple normalization methods for different signal types
- Quality metrics for alignment verification
- Detailed visualization of signals and alignment
- Support for various PCM formats
- Configurable maximum latency search range
- High-quality output visualization

## Input File Format

### PCM File Requirements

- **Format**: Raw PCM (Pulse Code Modulation)
- **Encoding**: Signed 16-bit integer (s16le)
- **Sample Rate**: 16000 Hz (default)
- **Channels**: Mono
- **Byte Order**: Little-endian
- **Container**: Raw PCM (no header)

### Input Files

1. **render.pcm**
   - Reference audio signal that was played
   - Should contain clear, distinct audio events
   - Typically periodic signals or beeps
   - Example: 1kHz test tone or pulse train
   - File size = (Duration in seconds) × 16000 × 2 bytes

2. **capture.pcm**
   - Recorded audio from system playback
   - Should be longer than render.pcm
   - Contains the same audio events with system latency
   - May include background noise
   - Should use same sample rate as render.pcm

### Creating Input Files

1. **Using ALSA**
```bash
# Record render.pcm (playback)
arecord -f S16_LE -r 16000 -c 1 render.pcm

# Record capture.pcm (microphone)
arecord -f S16_LE -r 16000 -c 1 capture.pcm
```

2. **Using FFmpeg**
```bash
# Convert from other formats to required PCM
ffmpeg -i input.wav -f s16le -acodec pcm_s16le -ar 16000 -ac 1 output.pcm
```

## Output Files

### 1. Numerical Output (Console)
```
Render Signal Analysis:
Peak level: 32767
RMS level: 12504.32
Crest factor: 2.62
Dynamic range: 42.3 dB

Capture Signal Analysis:
Peak level: 28123
RMS level: 10234.45
Crest factor: 2.75
Dynamic range: 40.1 dB

Alignment Results:
Latency: 342.50 ms
Correlation coefficient: 0.9823
Mean squared error: 0.0234
```

### 2. Visualization (PNG)
- **Filename**: Default 'latency_analysis.png' (configurable)
- **Resolution**: 300 DPI (configurable)
- **Format**: PNG (supports transparency)
- **Size**: Typically 15x12 inches

## Use Cases

### 1. Audio System Latency Testing
- Measure end-to-end audio system latency
- Test different audio configurations
- Compare different audio devices
- Benchmark audio drivers

### 2. Real-time System Analysis
- Measure audio processing delay
- Test audio buffer configurations
- Optimize system settings
- Monitor system performance

### 3. Quality Assurance
- Verify audio system specifications
- Test for regression in audio performance
- Validate audio hardware
- Compare different audio setups

### 4. Research and Development
- Audio system development
- Driver development
- Performance optimization
- Comparative analysis

## Example Workflows

### 1. Basic Latency Test
```bash
# 1. Generate test signal (1kHz tone)
ffmpeg -f lavfi -i "sine=frequency=1000:duration=5" -f s16le -acodec pcm_s16le -ar 16000 -ac 1 render.pcm

# 2. Play and record
# (Play render.pcm through speakers and record with microphone to capture.pcm)

# 3. Analyze latency
python latency_calc.py -r render.pcm -c capture.pcm -o results.png
```

### 2. Comparative Analysis
```bash
# Test different audio configurations
python latency_calc.py -r render.pcm -c capture_config1.pcm -o config1.png
python latency_calc.py -r render.pcm -c capture_config2.pcm -o config2.png
```

### 3. Continuous Monitoring
```bash
# Script for continuous monitoring
while true; do
    timestamp=$(date +%Y%m%d_%H%M%S)
    python latency_calc.py -r render.pcm -c capture.pcm -o "latency_${timestamp}.png"
    sleep 300  # Test every 5 minutes
done
```

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

Install dependencies:
```bash
pip install numpy scipy matplotlib
```

## Usage

### Basic Usage

```bash
python latency_calc.py -r render.pcm -c capture.pcm
```

### Advanced Options

```bash
python latency_calc.py \
    --render render.pcm \
    --capture capture.pcm \
    --max-latency 2.0 \
    --normalization zscore \
    --output analysis.png \
    --dpi 300
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|--------|---------|-------------|
| `--render` | `-r` | `render.pcm` | Path to the render PCM file |
| `--capture` | `-c` | `capture.pcm` | Path to the capture PCM file |
| `--max-latency` | `-m` | `1.0` | Maximum latency to consider (seconds) |
| `--normalization` | - | `zscore` | Signal normalization method |
| `--output` | `-o` | `latency_analysis.png` | Output image path |
| `--dpi` | - | `300` | DPI for output image |

### Normalization Methods

- `none`: No normalization
- `peak`: Normalize by peak amplitude
- `rms`: Normalize by RMS level
- `zscore`: Zero mean and unit variance (default)

## Output

The tool generates:

1. **Numerical Results**
   - Calculated latency in milliseconds
   - Correlation coefficient
   - Mean squared error
   - Signal quality metrics

2. **Visualization** (saved as image)
   - Original signals comparison
   - Aligned signals overlay
   - Alignment error plot

## Example Output

The generated visualization includes three plots:
1. **Top**: Original render and capture signals
2. **Middle**: Signals after alignment
3. **Bottom**: Difference between aligned signals

## Technical Details

### Signal Processing

- Uses cross-correlation to find optimal alignment
- Supports various normalization methods for different signal types
- Calculates alignment quality metrics:
  - Correlation coefficient
  - Mean squared error
  - Signal-to-noise ratio

### PCM Format

- Expects 16-bit signed integer PCM data (s16le)
- Default sample rate: 16000 Hz
- Supports both mono and stereo files

## Best Practices

1. **Signal Quality**
   - Ensure good signal-to-noise ratio in capture
   - Avoid clipping in both render and capture
   - Use consistent sample rates

2. **Measurement**
   - Use shorter files for faster processing
   - Verify results with multiple measurements
   - Check alignment quality metrics

## Troubleshooting

### Common Issues

1. **Poor Correlation**
   - Try different normalization methods
   - Check signal levels
   - Verify sample rates match

2. **Unexpected Latency**
   - Increase max_latency if needed
   - Check for signal clipping
   - Verify file formats

### Error Messages

- "File not found": Check file paths
- "Invalid PCM data": Verify file format
- "Poor correlation": Signal quality issue

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 LatencyLens

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
