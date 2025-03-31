import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from deepfake_detector import DeepfakeDetector
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 1rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .real {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .fake {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .result-text {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .confidence-text {
        font-size: 1.5rem;
        margin-top: 1rem;
        font-weight: bold;
    }
    .progress-bar {
        height: 30px;
        background-color: #e9ecef;
        border-radius: 15px;
        overflow: hidden;
        margin: 1rem 0;
    }
    .progress-fill {
        height: 100%;
        transition: width 0.5s ease-in-out;
    }
    .progress-fill.real {
        background-color: #28a745;
    }
    .progress-fill.fake {
        background-color: #dc3545;
    }
    .result-label {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🔍 Deepfake Detector")
st.markdown("""
    This application uses deep learning to detect potential deepfakes in images and videos.
    Upload an image or video file to analyze it.
""")

# Initialize the detector
@st.cache_resource
def load_detector():
    return DeepfakeDetector()

detector = load_detector()

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Image Detection", "Video Detection"])

# Image Detection Tab
with tab1:
    st.header("Image Detection")
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                # Save the uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name)
                    result = detector.detect(tmp_file.name)
                
                # Display results
                st.markdown("### Analysis Results")
                result_class = "fake" if result['is_deepfake'] else "real"
                result_text = "FAKE" if result['is_deepfake'] else "REAL"
                confidence = result['confidence']
                
                st.markdown(f"""
                    <div class="result-box {result_class}">
                        <div class="result-text">{result_text}</div>
                        <div class="result-label">Image Classification</div>
                        <div class="progress-bar">
                            <div class="progress-fill {result_class}" style="width: {confidence*100}%"></div>
                        </div>
                        <div class="confidence-text">Confidence Level: {confidence:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)

# Video Detection Tab
with tab2:
    st.header("Video Detection")
    uploaded_video = st.file_uploader("Choose a video file", type=['mp4', 'avi'])
    
    if uploaded_video is not None:
        # Save the uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.getvalue())
            video_path = tmp_file.name
        
        # Display video information
        st.video(uploaded_video)
        
        if st.button("Analyze Video"):
            with st.spinner("Analyzing video..."):
                # Create output path for processed video
                output_path = "output_video.mp4"
                
                # Analyze the video
                results = detector.analyze_video(video_path, output_path)
                
                # Display results
                st.markdown("### Analysis Results")
                if results:
                    # Calculate statistics
                    avg_confidence = sum(r['confidence'] for r in results) / len(results)
                    fake_frames = sum(1 for r in results if r['is_deepfake'])
                    total_frames = len(results)
                    fake_percentage = (fake_frames / total_frames) * 100
                    
                    # Determine overall result
                    overall_result = "fake" if fake_percentage > 50 else "real"
                    result_text = "FAKE" if overall_result == "fake" else "REAL"
                    
                    st.markdown(f"""
                        <div class="result-box {overall_result}">
                            <div class="result-text">{result_text}</div>
                            <div class="result-label">Video Classification</div>
                            <div class="progress-bar">
                                <div class="progress-fill {overall_result}" style="width: {avg_confidence*100}%"></div>
                            </div>
                            <div class="confidence-text">Confidence Level: {avg_confidence:.1%}</div>
                            <div style="margin-top: 1rem;">
                                <p>Total frames analyzed: {total_frames}</p>
                                <p>Frames detected as deepfake: {fake_frames} ({fake_percentage:.1f}%)</p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Offer download of processed video
                    if os.path.exists(output_path):
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="Download Processed Video",
                                data=f,
                                file_name="processed_video.mp4",
                                mime="video/mp4"
                            )
                else:
                    st.error("Error processing video. Please try again.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using PyTorch and Streamlit</p>
        <p>Note: This is a demonstration model and may not be 100% accurate</p>
    </div>
""", unsafe_allow_html=True) 