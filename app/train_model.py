import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.dataset_handler import DatasetHandler
from app.utils.audio_processor import AudioProcessor
from app.utils.speech_to_text import SpeechToText
from app.utils.text_analyzer import TextAnalyzer
from app.utils.emotion_classifier import EmotionClassifier

def main():
    print("Initializing emotion detection model training...")
    
    # Create dataset handler and download dataset
    dataset_handler = DatasetHandler()
    dataset_handler.download_ravdess()
    
    # Split dataset
    train_df, val_df, test_df = dataset_handler.split_dataset()
    print(f"Dataset split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples")
    
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
    
    # Train the model
    print("Training model...")
    accuracy = emotion_classifier.train(
        audio_features=train_audio_features,
        text_features=train_text_features,
        labels=train_labels
    )
    
    print(f"Training accuracy: {accuracy:.4f}")
    
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