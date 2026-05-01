# 👁️ Driver Drowsiness Detection System

A real-time computer vision application that detects eye state (open/closed) and yawning using MediaPipe and MobileNetV2 neural networks.

## ✨ Features

- **Real-time Eye Detection**: Detects whether eyes are open or closed
- **Yawn Detection**: Identifies yawning behavior
- **Head Tilt Detection**: Tracks the pitch angle of the head to detect nodding off
- **Drowsiness Alert**: Tracks a dynamic fatigue score and visually/audibly alerts the user when drowsy
- **Fast Performance**: Heavy AI runs every 2nd frame, allowing 30-50 FPS on CPU
- **Lightweight Models**: MobileNetV2 models are extremely lightweight
- **Audio Alert**: Plays an alarm sound when critical drowsiness is detected

## 📋 Requirements

- Python 3.8 to 3.12
- Webcam (for real-time detection)
- 500 MB disk space (for virtual environment + models)

⚠️ **Important Note on Dependencies**: This project uses the highly stable legacy `mediapipe.solutions` API. It is strictly tied to the versions specified in `requirements.txt` (specifically `mediapipe==0.10.14` and `numpy==1.26.4`). Do not blindly upgrade these packages, or the Face Mesh tracking will break!

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/harshKhulbe07/Driver-Drowsiness-Detection-System.git
cd Driver-Drowsiness-Detection-System
```

### 2. Create Virtual Environment
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Model Weights
Because the AI models are large binary files, they are not stored directly in the source code repository. 

1. Go to the [Releases Page](https://github.com/harshKhulbe07/Driver-Drowsiness-Detection-System/releases) of this repository.
2. Download the `models.zip` file from the latest release.
3. Extract the zip file and place all three files (`eye_weights.weights.h5`, `yawn_weights.weights.h5`, and `alarm.wav`) directly into the main project folder.

## 🎮 Usage

### Run Real-time Detection
```bash
python main.py
```

### Control Keys
- `ESC` - Exit application

### Output Display
- **Green boxes**: Eyes open / Normal behavior
- **Red boxes**: Eyes closed / Yawning / Head nodding
- **Fatigue score**: Displayed in the top-left (out of 10.0)

## 🏗️ Project Structure

```text
Driver-Drowsiness-Detection-System/
├── venv/                          # Virtual environment (not tracked by Git)
├── main.py                        # Main application script
├── test.py                        # Test script to verify the Keras model
├── eye_weights.weights.h5         # Eye detection model (download separately)
├── yawn_weights.weights.h5        # Yawn detection model (download separately)
├── alarm.wav                      # Alert sound file
├── requirements.txt               # Strict Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This documentation
```

## 🔧 Configuration

Edit these values at the top of `main.py` to customize the tracking sensitivity:

```python
EYE_CLOSED_THRESHOLD = 0.50        # Sensitivity for eye detection (0-1)
YAWN_THRESHOLD = 0.60              # Sensitivity for yawn detection (0-1)
HEAD_DROP_THRESHOLD = -15.0        # Head drop angle for alertness (degrees)
HISTORY_LENGTH = 5                 # Frames for smoothing predictions
```

## 📊 How It Works

```
Video Frame (Webcam)
    ↓
Face Detection (MediaPipe Face Mesh)
    ↓
Extract Eye & Mouth Crops
    ↓
MobileNetV2 Prediction
    ↓
Smoothing Filter & Memory
    ↓
Fatigue Scoring Engine
    ↓
Trigger Audio/Visual Alarm
```

### Algorithm Details

1. **Face Detection**: MediaPipe FaceMesh detects 468 facial landmarks.
2. **Feature Extraction**: Crops the left eye, right eye, and mouth regions using specific landmark coordinates. Also uses `cv2.solvePnP` to calculate head pitch.
3. **Preprocessing**: Resizes crops to 96×96 and converts to grayscale (simulated IR).
4. **Classification**: Pre-trained MobileNetV2 models predict eye state and yawning probabilities.
5. **Fatigue Scoring Engine**: 
   - Uses a dynamic score that increases based on yawning, head nodding, and continuous closed eyes.
   - Triggers an absolute override alarm if eyes are closed for >30 frames.
   - Automatically cools down (reduces score) during normal behavior.

## 🐛 Troubleshooting

**Error: "AttributeError: module 'mediapipe' has no attribute 'solutions'"**
- You have installed a newer version of MediaPipe where the legacy API was removed. Please strictly run `pip install -r requirements.txt` to install `mediapipe==0.10.14`.

**Error: "Could not find model weights"**
- Ensure the `.h5` files are placed in the root project directory and named exactly as they are in `main.py`.

**Error: "Could not find alarm.wav"**
- Place an audio file named `alarm.wav` in the directory, or the code will silently skip the audio alert without crashing.

## 📝 License

This project is open source. Feel free to use and modify.
