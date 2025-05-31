# Motion Tracking

## Overview

This repository contains a motion tracking application that uses computer vision techniques to track and analyze movement in video feeds.

## Features

- Real-time motion detection and tracking
- Support for multiple video sources (webcam, video files, IP cameras)
- Motion path visualization with customizable display options
- Advanced motion data analysis and export (CSV, JSON)
- Configurable detection sensitivity and tracking parameters

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Motion_tracking.git

# Navigate to the project directory
cd Motion_tracking

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the main application
python motion_tracker.py --source video.mp4

# Run with optimized settings for performance
python motion_tracker.py --source video.mp4 --optimize

# Enable debug visualization
python motion_tracker.py --source webcam --debug
```

## Performance Optimizations

The tracking algorithm has been optimized with:

- Multi-threading for parallel processing
- GPU acceleration (CUDA support)
- Memory usage optimizations
- Frame skipping for high-frame-rate sources

## Requirements

- Python 3.7+
- OpenCV 4.5+
- NumPy 1.20+
- CUDA Toolkit 10.1+ (optional, for GPU acceleration)
- Other dependencies listed in requirements.txt

## License

This project is licensed under the MIT License - see the LICENSE file for details.
