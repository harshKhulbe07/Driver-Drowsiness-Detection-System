# 👁️ Eye Detection & Drowsiness Detection Project

A real-time computer vision application that detects eye state (open/closed) and yawning using MediaPipe and MobileNetV2 neural networks.

## ✨ Features

- **Real-time Eye Detection**: Detects whether eyes are open or closed
- **Yawn Detection**: Identifies yawning behavior
- **Drowsiness Alert**: Tracks fatigue score and alerts user when drowsy
- **Fast Performance**: 30-50 FPS on CPU using MobileNetV2
- **Lightweight Model**: Only 14 MB model size
- **Audio Alert**: Plays sound when drowsiness detected

## 📋 Requirements

- Python 3.8+
- Webcam (for real-time detection)
- 500 MB disk space (for virtual environment + models)

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/eye_detection_project.git
cd eye_detection_project
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
Make sure you have these files in the project directory:
- `eye_weights.weights.h5` - Pre-trained eye detection model
- `yawn_weights.weights.h5` - Pre-trained yawn detection model
- `alarm.wav` - Alert sound file

## 🎮 Usage

### Run Real-time Detection
```bash
python main.py
```

### Control Keys
- `ESC` - Exit application
- `Q` - Quit
- `Space` - Pause/Resume

### Output Display
- **Green eyes**: Eyes open
- **Red eyes**: Eyes closed
- **Yellow mouth**: Yawning
- **Drowsiness score**: Displayed in top-right

## 🏗️ Project Structure

```
eye_detection_project/
├── venv/                          # Virtual environment (don't commit)
├── main.py                         # Main application
├── inference.py                    # Simplified inference
├── realtime.py                     # Alternative implementation
├── test_eye.py                     # Test script
├── eye_weights.weights.h5         # Eye detection model (don't commit)
├── yawn_weights.weights.h5        # Yawn detection model (don't commit)
├── alarm.wav                       # Alert sound
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 📚 Learning Materials

Comprehensive guides for understanding each technology:

- **MOBILENET_GUIDE.md** - MobileNet architecture deep dive
- **TENSORFLOW_GUIDE.md** - TensorFlow framework explained
- **MEDIAPIPE_GUIDE.md** - Face detection with MediaPipe
- **LEARNING_GUIDE.md** - Overview of all technologies
- **CODE_EXPLANATION.md** - Line-by-line code walkthrough
- **STUDY_PLAN.md** - Week-by-week learning curriculum

And corresponding **LAB** and **MASTERPLAN** guides for hands-on experiments!

## 🔧 Configuration

Edit these values in `main.py` to customize behavior:

```python
EYE_CLOSED_THRESHOLD = 0.50      # Sensitivity for eye detection (0-1)
YAWN_THRESHOLD = 0.60             # Sensitivity for yawn detection (0-1)
HEAD_DROP_THRESHOLD = -15.0        # Head drop angle for alertness
HISTORY_LENGTH = 5                 # Frames for smoothing predictions
```

## 📊 How It Works

```
Video Frame (30 FPS)
    ↓
Face Detection (MediaPipe)
    ↓
Extract Eye Regions
    ↓
MobileNetV2 Prediction
    ↓
Smoothing Filter
    ↓
Drowsiness Scoring
    ↓
Alert if Drowsy
```

### Algorithm Details

1. **Face Detection**: MediaPipe FaceMesh detects 468 facial landmarks
2. **Eye Extraction**: Crops eye regions using landmark coordinates
3. **Preprocessing**: Resizes to 96×96, converts to grayscale
4. **Classification**: MobileNetV2 predicts eye state (0-1 probability)
5. **Smoothing**: 5-frame history averaging reduces noise
6. **Fatigue Scoring**: 
   - +1 if eyes closed
   - -2 if eyes open (faster recovery)
   - Alert triggered at score > 20

## 📈 Performance

- **Inference Speed**: 12-35ms per frame (30-80 FPS)
- **Model Size**: 14 MB (eye) + 14 MB (yawn)
- **Memory**: ~100 MB during runtime
- **CPU**: Works on modern CPU (Intel i5+ or better)
- **GPU**: Automatic speedup if NVIDIA GPU available

## 🐛 Troubleshooting

### Installation Issues

**Error: "No module named 'tensorflow'"**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Error: "mediapipe not found"**
```bash
pip install mediapipe
```

### Runtime Issues

**Error: "Could not find model weights"**
- Ensure `.h5` files are in project directory
- Check filename matches in `main.py`

**Error: "Could not find alarm.wav"**
- Create audio file or disable audio in code

**Low FPS or High CPU**
- Close other applications
- Reduce model accuracy threshold for faster inference
- Use GPU if available

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source. Feel free to use and modify.

## 🎓 Learning Resources

The project includes comprehensive educational materials:

### For Beginners
- Start with `LEARNING_GUIDE.md`
- Read `CODE_EXPLANATION.md`
- Follow `STUDY_PLAN.md`

### For Advanced Users
- Deep dive with `MASTERPLAN` files
- Run hands-on experiments in `LAB` files
- Review `CHEATSHEET` files for quick reference

### Key Technologies Covered
- **MediaPipe**: Real-time face detection
- **MobileNetV2**: Lightweight neural networks
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Image processing
- **NumPy**: Numerical operations

## 🔗 Quick Links

- [MediaPipe Docs](https://mediapipe.dev/)
- [TensorFlow Docs](https://www.tensorflow.org/)
- [MobileNet Paper](https://arxiv.org/abs/1704.04861)
- [OpenCV Docs](https://docs.opencv.org/)

## 💡 Tips

### For Better Accuracy
1. Ensure good lighting
2. Look directly at camera
3. Adjust thresholds for your face
4. Retrain models with your data

### For Better Performance
1. Use GPU if available
2. Batch process multiple frames
3. Reduce video resolution if needed
4. Close unnecessary applications

### For Development
1. Use virtual environment (already set up!)
2. Run tests before committing
3. Check code with linter
4. Follow PEP 8 style guide

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review learning materials
3. Check code comments in `main.py`
4. Open an issue on GitHub

---

**Happy coding! 🚀**

Made with ❤️ for Computer Vision enthusiasts.
