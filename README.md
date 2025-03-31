# Deepfake Detector

This project implements a deepfake detection system using a pre-trained deep learning model. It can analyze both images and videos to detect potential deepfakes.

## Features

- Image-based deepfake detection
- Video-based deepfake detection with frame analysis
- Confidence score for each detection
- Support for both CPU and GPU (CUDA)
- Real-time visualization for video analysis

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- PIL (Python Imaging Library)
- NumPy
- tqdm

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd deepfake-detector
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Image Detection

```python
from deepfake_detector import DeepfakeDetector

# Initialize the detector
detector = DeepfakeDetector()

# Detect deepfake in an image
result = detector.detect("path/to/your/image.jpg")
print(result)
```

### Video Detection

```python
from deepfake_detector import DeepfakeDetector

# Initialize the detector
detector = DeepfakeDetector()

# Analyze a video file
results = detector.analyze_video("path/to/your/video.mp4", "output_video.mp4")
print(results)
```

## Output Format

The detector returns results in the following format:

For images:
```python
{
    'is_deepfake': bool,  # True if deepfake detected, False otherwise
    'confidence': float   # Confidence score between 0 and 1
}
```

For videos:
- A list of results for each analyzed frame
- An annotated video file (if output path is provided)

## Notes

- The model uses ResNet50 as the base architecture
- Video analysis processes every 30th frame (1 second at 30fps) to maintain performance
- GPU acceleration is automatically enabled if available
- The model is pre-trained and ready to use without additional training

## Limitations

- The accuracy of detection depends on the quality of the input media
- Some sophisticated deepfakes might not be detected
- Processing large videos may take significant time
- The model's performance may vary based on the hardware used

## License

This project is licensed under the MIT License - see the LICENSE file for details. 