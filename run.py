#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Emotion Detection System")
    parser.add_argument("command", choices=["setup", "train", "evaluate", "app", "demo"],
                        help="Command to run: setup, train, evaluate, app, or demo")
    
    args = parser.parse_args()
    
    # Make sure we're in the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    if args.command == "setup":
        # Install dependencies
        print("Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        # Download NLTK data
        print("Downloading NLTK data...")
        subprocess.run([sys.executable, "-c", "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"])
        
        print("Setup complete.")
        
    elif args.command == "train":
        # Train the model
        print("Training the emotion detection model...")
        subprocess.run([sys.executable, "app/train_model.py"])
        
    elif args.command == "evaluate":
        # Evaluate the model
        print("Evaluating the emotion detection model...")
        subprocess.run([sys.executable, "app/evaluate_model.py"])
        
    elif args.command == "app":
        # Run the web app
        print("Starting the web application...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/app.py"])
        
    elif args.command == "demo":
        # Run a simple demo with a few examples
        print("Running demo...")
        
        # Check if model exists
        if not os.path.exists("data/models/emotion_classifier.pkl"):
            print("Model not found. Please run 'python run.py train' first.")
            return
        
        # Import necessary modules
        sys.path.append(project_root)
        from app.utils.audio_processor import AudioProcessor
        from app.utils.speech_to_text import SpeechToText
        from app.utils.text_analyzer import TextAnalyzer
        from app.utils.emotion_classifier import EmotionClassifier
        
        # Initialize components
        audio_processor = AudioProcessor()
        speech_to_text = SpeechToText()
        text_analyzer = TextAnalyzer()
        emotion_classifier = EmotionClassifier("data/models/emotion_classifier.pkl")
        
        # Load dataset
        from app.utils.dataset_handler import DatasetHandler
        dataset_handler = DatasetHandler()
        _, _, test_df = dataset_handler.split_dataset()
        
        # Select a few examples
        sample_size = min(5, len(test_df))
        samples = test_df.sample(sample_size)
        
        print(f"\nAnalyzing {sample_size} random samples from the test set:\n")
        
        for i, (_, row) in enumerate(samples.iterrows()):
            print(f"Sample {i+1} - True emotion: {row['emotion']}")
            print(f"File: {row['filename']}")
            
            try:
                # Extract audio features
                audio_features, _ = audio_processor.extract_features_for_model(row['path'])
                
                # Transcribe speech
                transcription = speech_to_text.transcribe(row['path'])
                print(f"Transcription: {transcription}")
                
                # Extract text features
                text_features, _ = text_analyzer.extract_features_for_model(transcription)
                
                # Predict emotion
                predicted_emotion, confidence_scores = emotion_classifier.predict(
                    audio_features, text_features
                )
                
                print(f"Predicted emotion: {predicted_emotion}")
                print("Confidence scores:")
                for emotion, score in confidence_scores.items():
                    print(f"  {emotion}: {score:.4f}")
                print()
                
            except Exception as e:
                print(f"Error processing sample: {str(e)}\n")

if __name__ == "__main__":
    main() 