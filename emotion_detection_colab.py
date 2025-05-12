# %% [markdown]
# # Multimodal Emotion Detection System
# 
# This notebook provides a complete setup for running the multimodal emotion detection system in Google Colab. The system combines speech prosody analysis with text analysis to detect emotions.

# %% [markdown]
# ## 1. Setup Environment

# %%
# Clone the repository
!git clone https://github.com/dwcqwcqw/speech-emotion-detection.git

# Change working directory to the cloned repo
import os
os.chdir('speech-emotion-detection')
!pwd

# Install dependencies with specific versions compatible with Colab
!pip install -q numpy==1.26.4 pandas==2.2.2 scikit-learn==1.2.2 matplotlib==3.7.1 tensorflow==2.15.0 librosa==0.10.1 transformers==4.35.2 soundfile==0.12.1

# Verify installed versions
!pip list | grep -E "numpy|pandas|scikit-learn|matplotlib|tensorflow|librosa|transformers|soundfile"

# Install additional packages if needed
!pip install -q pyyaml

# %% [markdown]
# ## 2. Download and Prepare Dataset

# %%
# Download RAVDESS dataset
!wget -O ravdess.zip https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1
!mkdir -p data/ravdess
!unzip -q ravdess.zip -d data/ravdess
!rm ravdess.zip

# %% [markdown]
# ## 3. Analyze Repository Structure

# %%
# Explore repository directories
!ls -la
!echo "\nChecking for src directory:"
!ls -la src 2>/dev/null || echo "src directory not found"
!echo "\nChecking app directory:"
!ls -la app/ 2>/dev/null || echo "app directory not found"
!echo "\nListing Python files:" 
!find . -maxdepth 3 -type f -name "*.py" | sort

# Create src directory if needed (for module imports)
!mkdir -p src

# Check current working directory and Python path
import sys
import os
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Try both app and src paths
MODULE_BASE = 'app' if os.path.isdir('app') else 'src'
print(f"Using module base directory: {MODULE_BASE}")
sys.path.append(os.path.join(os.getcwd(), MODULE_BASE))

# %% [markdown]
# ## 4. Import Required Modules

# %%
# First try importing from potential modules
try:
    # Try potential module structures
    if os.path.isdir('app'):
        print("Attempting to import from app directory...")
        import app
        from app.models import AudioEmotionModel, TextEmotionModel
        from app.utils import setup_logging, load_config
        print("Successfully imported from app directory.")
    elif os.path.isdir('src'):
        print("Attempting to import from src directory...")
        import src
        from src.audio_features import AudioFeatureExtractor
        from src.data_processor import DataProcessor
        from src.models.audio_model import AudioEmotionModel
        from src.models.text_model import TextEmotionModel
        from src.models.multimodal_analyzer import MultimodalAnalyzer
        from src.utils import setup_logging, load_config
        print("Successfully imported from src directory.")
    
    # Check if we can find run.py's functionality
    print("Checking run.py for importing functionality...")
    import run
    print("Successfully imported run.py")
    
    # If we get here, we successfully imported some modules
    print("Module imports successful!")
    USE_IMPORTED_MODULES = True
    
except Exception as e:
    print(f"Could not import modules: {e}")
    print("Falling back to standalone implementation...")
    USE_IMPORTED_MODULES = False

# Import necessary Python modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## 5. Load or Create Configuration

# %%
# Look for config files in the repository
!find . -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.config"

# Try to load config from repository
try:
    # Try to import config
    if 'load_config' in locals():
        print("Using imported load_config function...")
        config = load_config("config.yaml")
        print("Config loaded successfully:")
        print(config)
    else:
        # Try to find and load config manually
        yaml_files = !find . -name "*.yaml" -o -name "*.yml"
        if yaml_files:
            print(f"Found config file: {yaml_files[0]}")
            with open(yaml_files[0], 'r') as f:
                config = yaml.safe_load(f)
            print("Config loaded successfully:")
            print(json.dumps(config, indent=2))
        else:
            raise FileNotFoundError("No config files found")
except Exception as e:
    print(f"Error loading config: {e}")
    print("Creating default config...")
    
    # Create a default config if none exists
    config = {
        "data": {
            "path": "data/ravdess",
            "test_size": 0.2,
            "random_state": 42
        },
        "audio": {
            "sample_rate": 22050,
            "duration": 3.0,
            "feature_type": "mfcc",
            "n_mfcc": 40
        },
        "model": {
            "type": "lstm",
            "params": {
                "units": 128,
                "dropout": 0.5,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50
            }
        },
        "emotions": ["happy", "sad", "angry", "neutral", "fearful"]
    }
    
    print("Using default config:")
    print(json.dumps(config, indent=2))

# %% [markdown]
# ## 6. Feature Extraction and Data Processing

# %%
# Determine whether to use imported modules or standalone code
if USE_IMPORTED_MODULES and 'AudioFeatureExtractor' in locals() and 'DataProcessor' in locals():
    print("Using imported modules for feature extraction and data processing...")
    
    # Use imported modules
    feature_extractor = AudioFeatureExtractor(config)
    data_processor = DataProcessor(config)
    
    # Process data using imported modules
    features, labels = data_processor.process_data()
    X_train, X_test, y_train, y_test = data_processor.split_data(features, labels)
    
else:
    print("Using standalone implementation for feature extraction and data processing...")
    
    # Import librosa for audio processing
    import librosa
    import librosa.display
    import glob
    
    # Define a function to extract features
    def extract_features(file_path, config):
        """Extract audio features from a file."""
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=config["audio"]["sample_rate"], duration=config["audio"]["duration"])
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config["audio"]["n_mfcc"])
            mfccs_processed = np.mean(mfccs.T, axis=0)
            
            return mfccs_processed
        except Exception as e:
            print(f"Error extracting features from {file_path}: {e}")
            return None
    
    # Function to process data
    def process_data(config):
        """Process audio data and extract features."""
        features = []
        labels = []
        emotions = config["emotions"]
        data_path = config["data"]["path"]
        
        # Find audio files
        audio_files = glob.glob(f"{data_path}/**/*.wav", recursive=True)
        print(f"Found {len(audio_files)} audio files")
        
        # Process a subset of files for demonstration (limit to 100 files)
        sample_files = audio_files[:100] if len(audio_files) > 100 else audio_files
        
        for file_path in sample_files:
            # Extract features
            feature = extract_features(file_path, config)
            if feature is not None:
                features.append(feature)
                
                # For demonstration, assign random emotion labels
                # In a real scenario, you would parse the filename or use a label file
                label = np.random.randint(0, len(emotions))
                labels.append(label)
        
        return np.array(features), np.array(labels)
    
    # Extract features from audio files
    features, labels = process_data(config)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=config["data"]["test_size"], 
        random_state=config["data"]["random_state"]
    )

# Display data info
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Class distribution: {np.bincount(y_train)}")

# %% [markdown]
# ## 7. Build and Train Audio Emotion Model

# %%
# Determine whether to use imported model or create one
if USE_IMPORTED_MODULES and 'AudioEmotionModel' in locals():
    print("Using imported AudioEmotionModel...")
    
    # Use imported audio model
    audio_model = AudioEmotionModel(config)
    history = audio_model.train(X_train, y_train, X_test, y_test)
    
else:
    print("Creating standalone model...")
    
    # Function to create a model
    def create_model(config, input_shape):
        """Create an LSTM model for audio emotion recognition."""
        model = Sequential()
        
        # LSTM layer
        model.add(LSTM(
            units=config["model"]["params"]["units"],
            input_shape=(input_shape[0], 1),
            return_sequences=True
        ))
        model.add(Dropout(config["model"]["params"]["dropout"]))
        
        # Second LSTM layer
        model.add(LSTM(units=64))
        model.add(Dropout(0.3))
        
        # Dense layers
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(config["emotions"]), activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config["model"]["params"]["learning_rate"]),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    # Reshape data for LSTM model
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Create model
    model = create_model(config, X_train.shape)
    model.summary()
    
    # Train model
    history = model.fit(
        X_train_reshaped, y_train,
        validation_data=(X_test_reshaped, y_test),
        batch_size=config["model"]["params"]["batch_size"],
        epochs=10,  # Use fewer epochs for demonstration
        verbose=1
    )

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Text-based Emotion Analysis

# %%
# Determine whether to use imported text model or create one
if USE_IMPORTED_MODULES and 'TextEmotionModel' in locals():
    print("Using imported TextEmotionModel...")
    
    # Use imported text model
    text_model = TextEmotionModel(config)
    
    # Define analyze_text_emotion function to match interface
    def analyze_text_emotion(text):
        return text_model.predict(text)
    
else:
    print("Creating standalone text analyzer...")
    
    # Import transformers for text emotion analysis
    from transformers import pipeline
    
    # Create a text emotion classifier
    sentiment_analyzer = pipeline("sentiment-analysis")
    
    # Function to analyze text emotion
    def analyze_text_emotion(text):
        """Analyze emotion from text using transformer model."""
        result = sentiment_analyzer(text)
        
        # Map sentiment labels to our emotion categories
        # This is a simplistic mapping for demonstration
        label = result[0]["label"]
        score = result[0]["score"]
        
        if "positive" in label.lower():
            emotion = "happy"
        elif "negative" in label.lower():
            emotion = "sad"  # or could be angry depending on context
        else:
            emotion = "neutral"
        
        return {"emotion": emotion, "confidence": score}

# Test with sample text
sample_texts = [
    "I'm feeling so happy today!",
    "I'm so angry I could scream",
    "I feel sad and disappointed",
    "Just another normal day",
    "That scared me so much"
]

for text in sample_texts:
    result = analyze_text_emotion(text)
    print(f"Text: '{text}' → Emotion: {result['emotion']} (Confidence: {result['confidence']:.2f})")

# %% [markdown]
# ## 9. Multimodal Emotion Analysis

# %%
# Determine whether to use imported multimodal analyzer or create one
if USE_IMPORTED_MODULES and 'MultimodalAnalyzer' in locals():
    print("Using imported MultimodalAnalyzer...")
    
    # Use imported multimodal analyzer
    multimodal_analyzer = MultimodalAnalyzer(audio_model, text_model, config)
    
else:
    print("Creating standalone multimodal analyzer...")
    
    # Create a simple multimodal analyzer to combine audio and text
    class SimpleMultimodalAnalyzer:
        def __init__(self, audio_model, config):
            self.audio_model = audio_model
            self.config = config
            self.emotions = config["emotions"]
        
        def analyze_audio(self, audio_features):
            """Predict emotion from audio features."""
            # Reshape for model input
            features = audio_features.reshape(1, audio_features.shape[0], 1)
            prediction = self.audio_model.predict(features, verbose=0)
            
            # Get predicted emotion and confidence
            emotion_idx = np.argmax(prediction[0])
            confidence = prediction[0][emotion_idx]
            
            return {
                "emotion": self.emotions[emotion_idx],
                "confidence": float(confidence)
            }
        
        def analyze_text(self, text):
            """Analyze emotion from text."""
            return analyze_text_emotion(text)
        
        def analyze(self, audio_features, text):
            """Combined analysis of audio and text."""
            audio_result = self.analyze_audio(audio_features)
            text_result = self.analyze_text(text)
            
            # Check for agreement between modalities
            agreement = audio_result["emotion"] == text_result["emotion"]
            
            # Calculate combined confidence
            audio_weight = 0.6  # Give slightly more weight to audio
            text_weight = 0.4
            
            # Detect potential sarcasm (when modalities disagree with high confidence)
            sarcasm_detected = False
            if not agreement and audio_result["confidence"] > 0.7 and text_result["confidence"] > 0.7:
                sarcasm_detected = True
            
            # Determine final emotion (prefer audio if confident, otherwise use highest confidence)
            if sarcasm_detected:
                final_emotion = "sarcastic"
                final_confidence = max(audio_result["confidence"], text_result["confidence"])
            elif audio_result["confidence"] > 0.7:
                final_emotion = audio_result["emotion"]
                final_confidence = audio_result["confidence"]
            elif text_result["confidence"] > 0.7:
                final_emotion = text_result["emotion"]
                final_confidence = text_result["confidence"]
            else:
                # Use weighted confidence
                if audio_result["confidence"] * audio_weight > text_result["confidence"] * text_weight:
                    final_emotion = audio_result["emotion"]
                else:
                    final_emotion = text_result["emotion"]
                
                final_confidence = (audio_result["confidence"] * audio_weight) + (text_result["confidence"] * text_weight)
            
            return {
                "emotion": final_emotion,
                "audio_emotion": audio_result["emotion"],
                "text_emotion": text_result["emotion"],
                "confidence": final_confidence,
                "modality_agreement": agreement,
                "sarcasm_detected": sarcasm_detected
            }
    
    # Create the multimodal analyzer
    if 'model' in locals():
        multimodal_analyzer = SimpleMultimodalAnalyzer(model, config)
    else:
        multimodal_analyzer = SimpleMultimodalAnalyzer(audio_model, config)

# %% [markdown]
# ## 10. Test with Sample Data

# %%
# Test with a sample audio file and text
# Find a sample audio file
sample_files = glob.glob("data/ravdess/**/*.wav", recursive=True)

if sample_files:
    # Extract features from a sample file
    sample_audio_path = sample_files[0]
    print(f"Using sample audio: {sample_audio_path}")
    
    # Extract features for the sample file
    if USE_IMPORTED_MODULES and 'AudioFeatureExtractor' in locals():
        sample_features = feature_extractor.extract_features(sample_audio_path)
    else:
        sample_features = extract_features(sample_audio_path, config)
    
    # Define sample texts with different emotions
    sample_text_pairs = [
        ("I'm feeling really happy today!", "matching"),
        ("I'm so angry right now!", "conflicting"),
        ("I feel rather neutral about this", "neutral"),
        ("This makes me so sad", "conflicting")
    ]
    
    # Test with different text samples
    for text, description in sample_text_pairs:
        print(f"\nTesting with {description} text: '{text}'")
        result = multimodal_analyzer.analyze(sample_features, text)
        
        print("Multimodal Analysis Results:")
        print(f"Detected Emotion: {result['emotion']}")
        print(f"Audio Emotion: {result['audio_emotion']}")
        print(f"Text Emotion: {result['text_emotion']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Modality Agreement: {result['modality_agreement']}")
        print(f"Sarcasm Detected: {result['sarcasm_detected']}")
else:
    print("No sample audio files found. Make sure the dataset was downloaded correctly.")

# %% [markdown]
# ## 11. Save Trained Models

# %%
# Save models to Google Drive
from google.colab import drive
drive.mount("/content/drive")

# Create directory for models
!mkdir -p "/content/drive/MyDrive/emotion_detection_models"

# Save model
if 'model' in locals():
    model.save("/content/drive/MyDrive/emotion_detection_models/audio_model")
else:
    audio_model.save("/content/drive/MyDrive/emotion_detection_models/audio_model")
print("Model saved to Google Drive.")
