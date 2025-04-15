# Gesture Swiper Engine 9000+

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.6%2B-brightgreen)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8.9%2B-orange)

A powerful hand gesture control system that lets you navigate between applications and browser tabs using intuitive hand movements. Built with MediaPipe and OpenCV for precise hand tracking and gesture recognition.

![Gesture Demo](https://example.com/demo.gif) <!-- Replace with actual demo when available -->

## ✨ Features

- 👐 Advanced hand tracking with MediaPipe
- 🖐️ Gesture-based Alt+Tab window switching (left hand)
- 👉 Swipe detection for browser tab navigation (right hand)
- 🔄 Forward and backward navigation support
- ⚙️ Customizable configuration settings
- 🐞 Debug mode with visual feedback
- 💡 Optimized for performance and reliability

## 📋 Requirements

- Python 3.6+
- Webcam or camera device
- The following Python packages:
  - OpenCV (`opencv-python`)
  - MediaPipe (`mediapipe`)
  - PyAutoGUI (`pyautogui`)
  - PyYAML (`pyyaml`) 
  - NumPy (`numpy`)

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/Abhinav-gh/Gesture_Based_Controller
cd Gesture_Based_Controller
```

2. Install the required dependencies:
```bash
pip install requirements.txt
```

3. Run the application:
```bash
python Gesture_Based_Controller.py
```

## 🎮 Gesture Controls

### Left Hand Controls (Window Switching)

| Gesture | Description | Action |
|---------|-------------|--------|
| 3 Fingers Up (index, middle, thumb) | Alt+Tab mode (forward) | Press and hold Alt, cycle with Tab |
| 4 Fingers Up (index, middle, ring, thumb) | Alt+Tab mode (backward) | Press and hold Alt, cycle with Shift+Tab |
| Change to any other gesture | Release Alt | Release the Alt key |

### Right Hand Controls (Browser Tab Navigation)

| Gesture | Description | Action |
|---------|-------------|--------|
| Swipe Right | Quick horizontal movement to the right | Next tab (Ctrl+Tab) |
| Swipe Left | Quick horizontal movement to the left | Previous tab (Ctrl+Shift+Tab) |

## 📝 How to Use Swipe Gestures Effectively

### Best Hand Positions for Swipes:

1. **Open Hand** - All fingers extended works best
2. **Pointing** - Index finger extended 
3. **Any pose** with at least one finger up

### Swipe Recognition Techniques:

#### 1. Quick Flick Method (Recommended)
- Start with your hand steady
- Make a quick, decisive horizontal motion
- Speed matters more than distance
- The system will detect this as a "Fast Flick"

#### 2. Deliberate Swipe Method
- Keep your hand at a steady height
- Move horizontally at a moderate pace
- Move at least 8% of the screen width
- Complete the motion within 1 second

### Troubleshooting Tips:
- Try faster, more decisive movements
- Keep your hand clearly visible to the camera
- Ensure good lighting conditions
- Use an open hand with extended fingers
- Make sure you're moving far enough horizontally

## ⚙️ Configuration

The application creates a YAML configuration file (`gesture_config.yaml`) on first run with default settings. You can modify these settings to adjust the sensitivity and behavior:

| Parameter | Default | Description |
|-----------|---------|-------------|
| gesture_delay | 1.5 | Time between consecutive gesture actions (seconds) |
| swipe_cooldown | 0.4 | Time between swipes (seconds) |
| min_swipe_velocity | 0.3 | Minimum velocity for swipe detection |
| min_swipe_distance | 0.08 | Minimum distance for swipe detection (normalized) |
| max_vertical_ratio | 0.7 | Max vertical/horizontal ratio for clean swipes |
| detection_confidence | 0.7 | Hand detection confidence threshold |
| tracking_confidence | 0.7 | Hand tracking confidence threshold |
| debug_mode | true | Show debug info on screen |
| flick_velocity_threshold | 1.0 | Velocity threshold for flick detection |
| swipe_intent_threshold | 0.2 | Threshold for detecting swipe intent |
| swipe_max_duration | 1.0 | Maximum time for a swipe to be considered valid (seconds) |

### Adjusting for Your Setup

For more sensitive swipe detection, try:
- Decrease `min_swipe_velocity` and `min_swipe_distance`
- Decrease `flick_velocity_threshold`
- Increase `max_vertical_ratio`

For fewer false positives:
- Increase `min_swipe_velocity` and `min_swipe_distance`
- Increase `flick_velocity_threshold`
- Decrease `max_vertical_ratio`

## 🖥️ Debug Mode

When `debug_mode` is set to `true`, the application will:
- Display a window with your camera feed
- Show hand landmarks and tracking information
- Display velocity vectors and swipe states
- Print detailed logs about detected gestures
- Show visual swipe effects
- Will NOT perform actual keyboard actions, only print what would happen

To exit debug mode, set `debug_mode: false` in the configuration file.

## 📊 How It Works

### Architecture Overview

1. **Hand Detection**: Uses MediaPipe's hand tracking model to detect and track hand landmarks in 3D space.

2. **Finger Detection**: Analyzes the positions of joints to determine which fingers are extended.

3. **Gesture Recognition**: Maps finger configurations to named gestures (e.g., "peace", "open", "three").

4. **Velocity Tracking**: Maintains a history of hand positions to calculate movement velocity and direction.

5. **Swipe Detection**: Uses a state machine approach:
   - **State 1 (none)**: Looking for potential swipe start
   - **State 2 (start)**: Tracking ongoing swipe motion
   - **State 3 (confirmed)**: Processing confirmed swipe action

6. **Action Execution**: Translates recognized gestures into keyboard actions via PyAutoGUI.

### Technical Implementation

The core algorithm uses a three-stage state machine for swipe detection:

1. **Intent Detection**: Identifies when a hand appears to be initiating a swipe motion.
2. **Motion Tracking**: Tracks the hand as it moves horizontally, measuring velocity and distance.
3. **Confirmation**: When certain thresholds are met, confirms the swipe and triggers the corresponding action.

For Alt+Tab functionality, the system monitors finger configurations of the left hand and controls modifier keys accordingly.

## 🔍 Troubleshooting

### Common Issues

- **No camera detected**: Ensure your webcam is connected and not in use by another application.
- **Poor hand detection**: Try improving lighting conditions and keeping your hand clearly visible.
- **Swipes not registering**: Try moving your hand faster or slower, and ensure it's clearly visible.
- **Erratic behavior**: Restart the application to reset hand tracking state.

### Debug Logs

If you encounter issues, enable `debug_mode` to see detailed information about the hand tracking and gesture recognition process.

## 📑 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [MediaPipe](https://mediapipe.dev/) for the excellent hand tracking solution
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [PyAutoGUI](https://pyautogui.readthedocs.io/) for keyboard control

---

Created by [Abhinav-gh](https://github.com/Abhinav-gh) | Last Updated: 2025-04-15