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

# Install dependencies
!pip install -q numpy pandas scikit-learn matplotlib tensorflow librosa transformers soundfile

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
import sys
# Add the current directory to path
sys.path.append(os.getcwd())

from src.audio_features import AudioFeatureExtractor
from src.data_processor import DataProcessor
from src.models.audio_model import AudioEmotionModel
from src.models.text_model import TextEmotionModel
from src.models.multimodal_analyzer import MultimodalAnalyzer
from src.utils import setup_logging, load_config

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 4. Load Configuration

# %%
# Load and show configuration
config = load_config("config.yaml")
print(config)

# %% [markdown]
# ## 5. Data Processing

# %%
# Process audio data
data_processor = DataProcessor(config)
features, labels = data_processor.process_data()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = data_processor.split_data(features, labels)

# Display data info
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Class distribution: {np.bincount(y_train)}")

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
