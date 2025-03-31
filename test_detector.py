import os
from deepfake_detector import DeepfakeDetector

def test_image_detection():
    """Test deepfake detection on an image."""
    detector = DeepfakeDetector()
    
    # Test with a sample image
    image_path = "test_image.jpg"
    if os.path.exists(image_path):
        print(f"\nTesting image detection on: {image_path}")
        result = detector.detect(image_path)
        print(f"Result: {result}")
    else:
        print(f"\nTest image not found: {image_path}")
        print("Please provide a test image named 'test_image.jpg'")

def test_video_detection():
    """Test deepfake detection on a video."""
    detector = DeepfakeDetector()
    
    # Test with a sample video
    video_path = "test_video.mp4"
    if os.path.exists(video_path):
        print(f"\nTesting video detection on: {video_path}")
        output_path = "output_video.mp4"
        results = detector.analyze_video(video_path, output_path)
        print(f"Analysis complete. Processed {len(results)} frames.")
        print(f"Output video saved as: {output_path}")
    else:
        print(f"\nTest video not found: {video_path}")
        print("Please provide a test video named 'test_video.mp4'")

if __name__ == "__main__":
    print("Starting Deepfake Detector Tests...")
    test_image_detection()
    test_video_detection()
    print("\nTests completed.") 