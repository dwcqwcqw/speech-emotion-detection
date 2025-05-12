import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from pydub import AudioSegment
import io
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 修改导入方式，使用相对导入
from utils.audio_processor import AudioProcessor
from utils.speech_to_text import SpeechToText
from utils.text_analyzer import TextAnalyzer
from utils.emotion_classifier import EmotionClassifier

# Set page configuration
st.set_page_config(
    page_title="Multimodal Emotion Detection",
    page_icon="🎭",
    layout="wide"
)

# Initialize models and processors
@st.cache_resource
def load_models():
    audio_processor = AudioProcessor()
    speech_to_text = SpeechToText()
    text_analyzer = TextAnalyzer()
    
    model_path = 'data/models/emotion_classifier.pkl'
    if os.path.exists(model_path):
        emotion_classifier = EmotionClassifier(model_path=model_path)
    else:
        emotion_classifier = EmotionClassifier()
        st.warning("Pre-trained model not found. Using untrained model. Please run train_model.py first.")
    
    return audio_processor, speech_to_text, text_analyzer, emotion_classifier

# Function to process audio and predict emotion
def process_audio(audio_file, audio_processor, speech_to_text, text_analyzer, emotion_classifier):
    # Save uploaded audio to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Process audio features
    with st.spinner("Extracting audio features..."):
        audio_features, audio_features_dict = audio_processor.extract_features_for_model(tmp_file_path)
    
    # Transcribe speech to text
    with st.spinner("Transcribing speech..."):
        transcription = speech_to_text.transcribe(tmp_file_path)
    
    # Process text features
    with st.spinner("Analyzing text..."):
        text_features, text_features_dict = text_analyzer.extract_features_for_model(transcription)
    
    # Predict emotion
    with st.spinner("Predicting emotion..."):
        predicted_emotion, confidence_scores = emotion_classifier.predict(audio_features, text_features)
    
    # Clean up temporary file
    os.unlink(tmp_file_path)
    
    return {
        'transcription': transcription,
        'audio_features': audio_features_dict,
        'text_features': text_features_dict,
        'predicted_emotion': predicted_emotion,
        'confidence_scores': confidence_scores
    }

# Function to record audio
def record_audio():
    audio_bytes = st.audio_recorder(pause_threshold=5.0)
    return audio_bytes

# Function to plot emotion confidence scores
def plot_confidence_scores(confidence_scores):
    emotions = list(confidence_scores.keys())
    scores = list(confidence_scores.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(emotions, scores, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#C2C2F0'])
    
    # Add labels and title
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Confidence Score')
    ax.set_title('Emotion Confidence Scores')
    ax.set_ylim(0, 1)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

# Main app
def main():
    st.title("🎭 Multimodal Emotion Detection")
    st.markdown("""
    This application detects emotions from speech by analyzing both **how** something is said (tone, pitch, intensity)
    and **what** is said (text content).
    """)
    
    # Load models
    audio_processor, speech_to_text, text_analyzer, emotion_classifier = load_models()
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("""
    This app uses a multimodal approach to detect emotions in speech:
    
    1. **Audio Processing**: Analyzes prosodic features like pitch, energy, and tempo
    2. **Speech Recognition**: Transcribes speech to text
    3. **Text Analysis**: Analyzes the linguistic content for emotional cues
    4. **Fusion**: Combines both modalities for improved emotion detection
    
    Supported emotions: Happy, Sad, Angry, Anxious, Neutral
    """)
    
    # Choose input method
    input_method = st.radio("Choose input method:", ["Upload Audio", "Record Audio"], horizontal=True)
    
    # File uploader or audio recorder
    audio_file = None
    if input_method == "Upload Audio":
        audio_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])
        if audio_file:
            st.audio(audio_file, format="audio/wav")
    else:
        st.write("Click the microphone to start recording, and click again to stop.")
        audio_bytes = record_audio()
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            # Convert audio bytes to file-like object
            audio_file = io.BytesIO(audio_bytes)
    
    # Process button
    if audio_file and st.button("Analyze Emotion"):
        with st.spinner("Processing audio..."):
            # Show progress
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulate progress
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Process audio file
            results = process_audio(
                audio_file, 
                audio_processor, 
                speech_to_text, 
                text_analyzer, 
                emotion_classifier
            )
            
            # Remove progress bar
            progress_bar.empty()
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Results")
            st.markdown(f"### Detected Emotion: **{results['predicted_emotion'].capitalize()}**")
            
            # Display transcription
            st.subheader("Transcription")
            st.write(results['transcription'])
            
            # Display plot
            st.subheader("Confidence Scores")
            fig = plot_confidence_scores(results['confidence_scores'])
            st.pyplot(fig)
        
        with col2:
            # Display audio features
            st.subheader("Audio Features")
            # Select important audio features to display
            important_audio_features = {
                'pitch_mean': 'Pitch (Mean)',
                'energy_mean': 'Energy (Mean)',
                'tempo': 'Tempo',
                'zero_crossing_rate_mean': 'Zero Crossing Rate'
            }
            
            audio_df = pd.DataFrame({
                'Feature': important_audio_features.values(),
                'Value': [results['audio_features'][k] for k in important_audio_features.keys()]
            })
            st.table(audio_df)
            
            # Display text features
            st.subheader("Text Features")
            # Select important text features to display
            important_text_features = {
                'sentiment_compound': 'Sentiment (Overall)',
                'sentiment_pos': 'Positive Sentiment',
                'sentiment_neg': 'Negative Sentiment',
                'emotion_happy_ratio': 'Happy Words Ratio',
                'emotion_sad_ratio': 'Sad Words Ratio',
                'emotion_angry_ratio': 'Angry Words Ratio',
                'emotion_anxious_ratio': 'Anxious Words Ratio'
            }
            
            text_df = pd.DataFrame({
                'Feature': important_text_features.values(),
                'Value': [results['text_features'][k] for k in important_text_features.keys()]
            })
            st.table(text_df)

if __name__ == "__main__":
    main() 