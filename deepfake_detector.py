import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import os

class DeepfakeDetector:
    def __init__(self):
        # Initialize the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def _load_model(self):
        try:
            # Load the pre-trained model (using ResNet18 as it's smaller and more reliable)
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            # Modify the last layer for binary classification (real vs fake)
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, 2)
            model.to(self.device)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting to load model from local cache...")
            try:
                # Try loading from local cache
                model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True, force_reload=False)
                num_features = model.fc.in_features
                model.fc = torch.nn.Linear(num_features, 2)
                model.to(self.device)
                return model
            except Exception as e2:
                print(f"Error loading from cache: {e2}")
                raise Exception("Failed to load the model. Please check your internet connection and try again.")
    
    def preprocess_image(self, image_path):
        """Preprocess the input image."""
        try:
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                image = Image.fromarray(image_path).convert('RGB')
            
            # Apply transformations
            image_tensor = self.transform(image)
            return image_tensor.unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise
    
    def detect(self, image_path):
        """Detect if the image is a deepfake."""
        try:
            self.model.eval()
            with torch.no_grad():
                # Preprocess the image
                image_tensor = self.preprocess_image(image_path)
                
                # Get prediction
                outputs = self.model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                
                # Get confidence score
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence = probabilities[0][predicted.item()].item()
                
                return {
                    'is_deepfake': bool(predicted.item()),
                    'confidence': confidence
                }
        except Exception as e:
            print(f"Error during detection: {e}")
            return {
                'is_deepfake': None,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def analyze_video(self, video_path, output_path=None):
        """Analyze a video file for deepfakes."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Initialize video writer if output path is provided
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            results = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Analyze every 30th frame (1 second at 30fps)
                if frame_count % 30 == 0:
                    result = self.detect(frame)
                    results.append(result)
                    
                    # Draw results on frame
                    label = "Deepfake" if result['is_deepfake'] else "Real"
                    color = (0, 0, 255) if result['is_deepfake'] else (0, 255, 0)
                    cv2.putText(frame, f"{label}: {result['confidence']:.2f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                if writer:
                    writer.write(frame)
                
                frame_count += 1
            
            cap.release()
            if writer:
                writer.release()
                
            return results
        except Exception as e:
            print(f"Error analyzing video: {e}")
            return []

def main():
    # Example usage
    detector = DeepfakeDetector()
    
    # Test with an image
    image_path = "test_image.jpg"  # Replace with your image path
    if os.path.exists(image_path):
        result = detector.detect(image_path)
        print(f"Image Analysis Result: {result}")
    
    # Test with a video
    video_path = "test_video.mp4"  # Replace with your video path
    if os.path.exists(video_path):
        output_path = "output_video.mp4"
        results = detector.analyze_video(video_path, output_path)
        print(f"Video Analysis Results: {results}")

if __name__ == "__main__":
    main() 