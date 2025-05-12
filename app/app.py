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

# Modified imports to use the simple speech-to-text
from utils.audio_processor import AudioProcessor
try:
    # Try to import the transformer-based speech-to-text
    from utils.speech_to_text import SpeechToText
    print("Using transformer-based speech-to-text")
except ImportError:
    # Fall back to the simple speech-to-text
    from utils.speech_to_text_simple import SimpleSpeechToText as SpeechToText
    print("Using simplified speech-to-text (mock transcriptions)")

# Import text analyzers and emotion classifiers
try:
    from utils.text_analyzer_extended import TextAnalyzerExtended as TextAnalyzer
    print("Using extended text analyzer with sarcasm detection")
except ImportError:
    from utils.text_analyzer import TextAnalyzer
    print("Using basic text analyzer")

try:
    from utils.emotion_classifier_extended import EmotionClassifierExtended as EmotionClassifier
    print("Using extended emotion classifier with sarcasm detection")
except ImportError:
    from utils.emotion_classifier import EmotionClassifier
    print("Using basic emotion classifier")

# Import the new multimodal analyzer
from utils.multimodal_analyzer import MultimodalAnalyzer

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
    
    # Load text analyzer
    text_analyzer = TextAnalyzer()
    
    # Load emotion classifier
    model_path = 'data/models/emotion_classifier.pkl'
    extended_model_path = 'data/models/emotion_classifier_extended.pkl'
    
    # Try to load extended model first, fall back to basic model
    if os.path.exists(extended_model_path):
        emotion_classifier = EmotionClassifier(model_path=extended_model_path)
    elif os.path.exists(model_path):
        emotion_classifier = EmotionClassifier(model_path=model_path)
    else:
        emotion_classifier = EmotionClassifier()
        st.warning("Pre-trained model not found. Using untrained model. Please run train_model.py first.")
    
    # Initialize the multimodal analyzer
    multimodal_analyzer = MultimodalAnalyzer(
        audio_processor=audio_processor,
        speech_to_text=speech_to_text,
        text_analyzer=text_analyzer,
        emotion_classifier=emotion_classifier
    )
    
    return audio_processor, speech_to_text, text_analyzer, emotion_classifier, multimodal_analyzer

# Function to process audio and predict emotion using multimodal analyzer
def process_audio(audio_file, multimodal_analyzer):
    # Save uploaded audio to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Use the multimodal analyzer for integrated analysis
    with st.spinner("Analyzing audio and text..."):
        results = multimodal_analyzer.analyze(tmp_file_path)
    
    # Clean up temporary file
    os.unlink(tmp_file_path)
    
    return results

# Function to record audio
def record_audio():
    audio_bytes = st.audio_recorder(pause_threshold=5.0)
    return audio_bytes

# Function to plot emotion confidence scores
def plot_confidence_scores(confidence_scores):
    emotions = list(confidence_scores.keys())
    scores = list(confidence_scores.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use a better color palette with consistent colors for emotions
    emotion_colors = {
        'happy': '#FF9999',
        'sad': '#66B2FF',
        'angry': '#FF6666',
        'anxious': '#FFCC99',
        'neutral': '#C2C2F0',
        'sarcastic': '#99FF99'
    }
    
    # Get colors for each emotion, defaulting to gray for unknown emotions
    bar_colors = [emotion_colors.get(emotion, '#CCCCCC') for emotion in emotions]
    
    bars = ax.bar(emotions, scores, color=bar_colors)
    
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

# Function to plot modality weights
def plot_modality_weights(weights):
    modalities = list(weights.keys())
    weight_values = list(weights.values())
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(modalities, weight_values, color=['#9370DB', '#20B2AA'])
    
    # Add labels and title
    ax.set_xlabel('Modality')
    ax.set_ylabel('Weight')
    ax.set_title('Modality Contribution')
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
    and **what** is said (text content) with enhanced integration between modalities.
    """)
    
    # Load models
    audio_processor, speech_to_text, text_analyzer, emotion_classifier, multimodal_analyzer = load_models()
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("""
    This app uses an enhanced multimodal approach to detect emotions in speech:
    
    1. **Audio Processing**: Analyzes prosodic features like pitch, energy, and tempo
    2. **Speech Recognition**: Transcribes speech to text
    3. **Text Analysis**: Analyzes the linguistic content for emotional cues
    4. **Contextual Integration**: Intelligently combines modalities based on context
    5. **Disagreement Detection**: Identifies when speech tone contradicts the words (potential sarcasm)
    
    Supported emotions: Happy, Sad, Angry, Anxious, Neutral, Sarcastic
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
            
            # Process audio file with multimodal analyzer
            results = process_audio(audio_file, multimodal_analyzer)
            
            # Remove progress bar
            progress_bar.empty()
        
        # Display results
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Results")
            st.markdown(f"### Detected Emotion: **{results['emotion'].capitalize()}**")
            
            # Display transcription
            st.subheader("Transcription")
            st.write(results['transcription'])
            
            # Display agreement score if available
            if 'agreement_score' in results:
                agreement = results['agreement_score']
                st.subheader("Speech-Text Agreement")
                
                # Create a colored box for the agreement score
                if agreement > 0.3:
                    st.success(f"Speech tone and text content are in agreement: {agreement:.2f}")
                elif agreement < -0.3:
                    st.error(f"Speech tone and text content show disagreement: {agreement:.2f}")
                    st.info("Disagreement may indicate sarcasm or mixed emotions.")
                else:
                    st.info(f"Neutral relationship between speech tone and text: {agreement:.2f}")
            
            # Display plot
            st.subheader("Confidence Scores")
            fig = plot_confidence_scores(results['confidence_scores'])
            st.pyplot(fig)
            
            # Display modality weights
            if 'modality_weights' in results:
                st.subheader("Modality Contribution")
                weights_fig = plot_modality_weights(results['modality_weights'])
                st.pyplot(weights_fig)
        
        with col2:
            # Display audio features
            st.subheader("Audio Features")
            # Select important audio features to display
            important_audio_features = {
                'pitch_mean': 'Pitch (Mean)',
                'pitch_std': 'Pitch (Variation)',
                'energy_mean': 'Energy (Mean)',
                'energy_std': 'Energy (Variation)',
                'tempo': 'Tempo',
                'zero_crossing_rate_mean': 'Zero Crossing Rate'
            }
            
            # Only display features that exist in the results
            audio_display = {}
            for k, v in important_audio_features.items():
                if k in results['audio_features']:
                    audio_display[k] = v
            
            if audio_display:
                audio_df = pd.DataFrame({
                    'Feature': audio_display.values(),
                    'Value': [results['audio_features'][k] for k in audio_display.keys()]
                })
                st.table(audio_df)
            
            # Display text features
            st.subheader("Text Features")
            # Select important text features to display
            important_text_features = {
                'sentiment_compound': 'Sentiment (Overall)',
                'sentiment_pos': 'Positive Sentiment',
                'sentiment_neg': 'Negative Sentiment'
            }
            
            # Add emotion word ratios if available
            for emotion in ['happy', 'sad', 'angry', 'anxious', 'neutral']:
                key = f'emotion_{emotion}_ratio'
                if key in results['text_features']:
                    important_text_features[key] = f'{emotion.capitalize()} Words Ratio'
            
            # Add sarcasm scores if available
            if 'sarcasm_score_rule' in results['text_features']:
                important_text_features['sarcasm_score_rule'] = 'Sarcasm Score (Rule-based)'
            if 'sarcasm_score_model' in results['text_features']:
                important_text_features['sarcasm_score_model'] = 'Sarcasm Score (Model-based)'
            
            # Only display features that exist in the results
            text_display = {}
            for k, v in important_text_features.items():
                if k in results['text_features']:
                    text_display[k] = v
            
            if text_display:
                text_df = pd.DataFrame({
                    'Feature': text_display.values(),
                    'Value': [results['text_features'][k] for k in text_display.keys()]
                })
                st.table(text_df)

if __name__ == "__main__":
    main() 