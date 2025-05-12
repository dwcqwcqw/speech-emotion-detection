import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Modified imports to use the simple speech-to-text
from utils.dataset_handler import DatasetHandler
from utils.audio_processor import AudioProcessor
try:
    # Try to import the transformer-based speech-to-text
    from utils.speech_to_text import SpeechToText
    print("Using transformer-based speech-to-text")
except ImportError:
    # Fall back to the simple speech-to-text
    from utils.speech_to_text_simple import SimpleSpeechToText as SpeechToText
    print("Using simplified speech-to-text (mock transcriptions)")

from utils.text_analyzer import TextAnalyzer
from utils.emotion_classifier import EmotionClassifier

def main():
    print("Initializing emotion detection model training...")
    
    # Create dataset handler and download dataset
    dataset_handler = DatasetHandler()
    dataset_handler.download_ravdess()
    
    # Split dataset
    train_df, val_df, test_df = dataset_handler.split_dataset()
    print(f"Dataset split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples")
    
    # Check if dataset is empty
    if len(train_df) == 0:
        print("ERROR: No training data available. Please check the dataset.")
        print("This could be due to:")
        print("1. Failed download or extraction of the dataset")
        print("2. Issues with the metadata file creation")
        print("3. Problems with the audio directory structure")
        print("\nTrying to verify dataset directories...")
        
        # Debug information
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        audio_dir = os.path.join(data_dir, 'audio')
        
        print(f"Data directory exists: {os.path.exists(data_dir)}")
        print(f"Audio directory exists: {os.path.exists(audio_dir)}")
        
        if os.path.exists(audio_dir):
            files = os.listdir(audio_dir)
            print(f"Audio directory contains {len(files)} files")
            if len(files) > 0:
                print(f"Sample files: {files[:5]}")
        
        return  # Exit the function as we can't proceed without data
    
    # Initialize processors
    audio_processor = AudioProcessor()
    speech_to_text = SpeechToText()
    text_analyzer = TextAnalyzer()
    emotion_classifier = EmotionClassifier()
    
    # Extract features from training data
    print("Extracting features from training data...")
    train_audio_features = []
    train_text_features = []
    train_labels = []
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        try:
            # Extract audio features
            audio_feat, _ = audio_processor.extract_features_for_model(row['path'])
            
            # Transcribe speech to text
            transcription = speech_to_text.transcribe(row['path'])
            
            # Extract text features
            text_feat, _ = text_analyzer.extract_features_for_model(transcription)
            
            # Store features and label
            train_audio_features.append(audio_feat)
            train_text_features.append(text_feat)
            train_labels.append(row['emotion'])
        except Exception as e:
            print(f"Error processing {row['filename']}: {str(e)}")
    
    # Check if we have enough training data after processing
    if len(train_audio_features) == 0:
        print("ERROR: No valid training samples could be processed.")
        return
    
    # Train the model
    print("Training model...")
    accuracy = emotion_classifier.train(
        audio_features=train_audio_features,
        text_features=train_text_features,
        labels=train_labels
    )
    
    print(f"Training accuracy: {accuracy:.4f}")
    
    # Check if validation set is empty
    if len(val_df) == 0:
        print("WARNING: No validation data available. Skipping validation.")
        model_path = 'data/models/emotion_classifier.pkl'
        emotion_classifier.save_model(model_path)
        print(f"Model saved to {model_path}")
        return
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_audio_features = []
    val_text_features = []
    val_labels = []
    
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        try:
            # Extract audio features
            audio_feat, _ = audio_processor.extract_features_for_model(row['path'])
            
            # Transcribe speech to text
            transcription = speech_to_text.transcribe(row['path'])
            
            # Extract text features
            text_feat, _ = text_analyzer.extract_features_for_model(transcription)
            
            # Store features and label
            val_audio_features.append(audio_feat)
            val_text_features.append(text_feat)
            val_labels.append(row['emotion'])
        except Exception as e:
            print(f"Error processing {row['filename']}: {str(e)}")
    
    # Check if we have enough validation data after processing
    if len(val_audio_features) == 0:
        print("WARNING: No valid validation samples could be processed.")
        model_path = 'data/models/emotion_classifier.pkl'
        emotion_classifier.save_model(model_path)
        print(f"Model saved to {model_path}")
        return
    
    # Evaluate on validation set
    val_metrics = emotion_classifier.evaluate(
        audio_features=val_audio_features,
        text_features=val_text_features,
        true_labels=val_labels
    )
    
    print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
    print("Per-class metrics:")
    
    for emotion, metrics in val_metrics['per_class_metrics'].items():
        print(f"  {emotion}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
    
    # Save the model
    model_path = 'data/models/emotion_classifier.pkl'
    emotion_classifier.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main() 