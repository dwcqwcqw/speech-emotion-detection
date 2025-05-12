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

# %% [markdown]
# ## 2. Download and Prepare Dataset

# %%
# Download RAVDESS dataset
!wget -O ravdess.zip https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1
!mkdir -p data/ravdess
!unzip -q ravdess.zip -d data/ravdess
!rm ravdess.zip

# %% [markdown]
# ## 3. Import Required Modules

# %%
# Debug: Check the repository structure
!ls -la
!find . -type d -name "src" -o -name "source" -o -name "lib"

# Check current working directory and Python path
import sys
import os
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Create src directory if it doesn't exist (failsafe)
!mkdir -p src

# Check if we need to create module structure
!test -d src/audio_features.py || test -d src/audio_features || echo "Audio features module not found"
!test -d src/data_processor.py || test -d src/data_processor || echo "Data processor module not found"

# Try different approaches to add the path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.getcwd())

# %%
# Continue only after confirming module structure
# First try to import directly
try:
    from src.audio_features import AudioFeatureExtractor
    from src.data_processor import DataProcessor
    from src.models.audio_model import AudioEmotionModel
    from src.models.text_model import TextEmotionModel
    from src.models.multimodal_analyzer import MultimodalAnalyzer
    from src.utils import setup_logging, load_config
    print("Modules successfully imported!")
except ModuleNotFoundError as e:
    print(f"Module import error: {e}")
    print("\nFallback to alternative import method:")
    # Try alternative approach if the repository structure is different
    import importlib.util
    import glob
    
    # Find Python files
    py_files = glob.glob("**/*.py", recursive=True)
    print(f"Python files found: {py_files}")
    
    # Try to locate modules in a different structure
    audio_features_path = next((path for path in py_files if "audio_features" in path), None)
    data_processor_path = next((path for path in py_files if "data_processor" in path), None)
    audio_model_path = next((path for path in py_files if "audio_model" in path), None)
    text_model_path = next((path for path in py_files if "text_model" in path), None)
    multimodal_path = next((path for path in py_files if "multimodal" in path), None)
    utils_path = next((path for path in py_files if "utils" in path), None)
    
    print(f"Found modules at: {audio_features_path}, {data_processor_path}, {audio_model_path}, {text_model_path}, {multimodal_path}, {utils_path}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 4. Load Configuration

# %%
# Try to find the config file
!find . -name "*.yaml" -o -name "*.yml"

# Load configuration - updated to handle dynamic path finding
try:
    config_path = "config.yaml"
    config = load_config(config_path)
    print(config)
except Exception as e:
    print(f"Error loading config: {e}")
    # Try to find and load config manually
    import yaml
    yaml_files = !find . -name "*.yaml" -o -name "*.yml"
    if yaml_files:
        with open(yaml_files[0], 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {yaml_files[0]}")
        print(config)
    else:
        # Create a minimal default config if none exists
        config = {
            "data": {"path": "data/ravdess"},
            "model": {"type": "cnn", "params": {"units": 64, "dropout": 0.5}}
        }
        print("Using default config:")
        print(config)

# %% [markdown]
# ## 5. Data Processing

# %%
# Process audio data - wrapped in try/except for debugging
try:
    data_processor = DataProcessor(config)
    features, labels = data_processor.process_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = data_processor.split_data(features, labels)

    # Display data info
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")
except Exception as e:
    print(f"Error in data processing: {e}")
    # Create dummy data for testing
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Generate dummy features and labels
    print("Generating dummy data for testing...")
    features = np.random.rand(100, 128)  # 100 samples, 128 features
    labels = np.random.randint(0, 5, 100)  # 5 classes
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    print(f"Dummy training data shape: {X_train.shape}")
    print(f"Dummy testing data shape: {X_test.shape}")
    print(f"Dummy class distribution: {np.bincount(y_train)}")

# %% [markdown]
# ## 6. Train Audio Emotion Model

# %%
# Initialize and train the audio model
audio_model = AudioEmotionModel(config)
history = audio_model.train(X_train, y_train, X_test, y_test)

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
# ## 7. Initialize Text Model

# %%
# Initialize text model
text_model = TextEmotionModel(config)

# %% [markdown]
# ## 8. Setup Multimodal Analysis

# %%
# Initialize the multimodal analyzer
analyzer = MultimodalAnalyzer(audio_model, text_model, config)

# %% [markdown]
# ## 9. Test with Sample Data

# %%
# Test with sample audio
audio_path = "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav"
text = "I'm feeling quite happy today."

# Extract audio features
feature_extractor = AudioFeatureExtractor(config)
audio_features = feature_extractor.extract_features(audio_path)

# Run multimodal analysis
result = analyzer.analyze(audio_features, text)
print("\nMultimodal Analysis Results:")
print(f"Detected Emotion: {result['emotion']}")
print(f"Audio Emotion: {result['audio_emotion']}")
print(f"Text Emotion: {result['text_emotion']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Modality Agreement: {result['modality_agreement']}")
print(f"Sarcasm Detected: {result['sarcasm_detected']}")

# %% [markdown]
# ## 10. Evaluate Model Performance

# %%
# Evaluate the audio model
audio_model.evaluate(X_test, y_test)

# %% [markdown]
# ## 11. Save Trained Models

# %%
# Save models to Google Drive
from google.colab import drive
drive.mount("/content/drive")

# Save audio model
audio_model.save("/content/drive/MyDrive/emotion_detection_models/audio_model")
print("Models saved to Google Drive.")
