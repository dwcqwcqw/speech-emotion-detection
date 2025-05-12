#!/usr/bin/env python3

import os
import sys
import shutil

def main():
    """
    Fix Python module import issues for Google Colab environment.
    This modifies the source files directly to use absolute imports.
    """
    # Ensure we're running from the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    print("Fixing import issues for Google Colab...")
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/audio", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    
    print(f"Created directory structure in {project_root}")
    print(f"- data/ (exists: {os.path.exists('data')})")
    print(f"- data/audio/ (exists: {os.path.exists('data/audio')})")
    print(f"- data/models/ (exists: {os.path.exists('data/models')})")
    
    # First, ensure that all necessary files exist
    if not os.path.exists("app/utils/speech_to_text_simple.py"):
        print("Error: speech_to_text_simple.py not found. Creating it...")
        # Create the simplified speech-to-text implementation
        with open("app/utils/speech_to_text_simple.py", "w") as f:
            f.write("""import os
import librosa
import numpy as np
import string

class SimpleSpeechToText:
    \"\"\"
    A simplified speech-to-text class that returns a dummy transcription.
    This is used as a fallback when the transformers library is not available.
    
    For a real application, you would need to use a proper ASR system or API.
    \"\"\"
    
    def __init__(self):
        \"\"\"Initialize the simple speech to text converter.\"\"\"
        self.emotions = {
            'happy': ['happy', 'joy', 'excited', 'pleasure', 'delighted'],
            'sad': ['sad', 'unhappy', 'disappointed', 'sorrow', 'grief'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated'], 
            'anxious': ['afraid', 'worried', 'nervous', 'anxious', 'stressed'],
            'neutral': ['fine', 'okay', 'normal', 'neutral', 'calm']
        }
    
    def transcribe(self, file_path, sample_rate=16000):
        \"\"\"
        Generate a simple mock transcription based on audio characteristics.
        
        Args:
            file_path: Path to the audio file
            sample_rate: Sample rate for loading audio
            
        Returns:
            str: A simple mock transcription
        \"\"\"
        # Load audio
        try:
            waveform, sr = librosa.load(file_path, sr=sample_rate)
            
            # Extract basic features
            energy = np.mean(librosa.feature.rms(y=waveform)[0])
            zero_cross = np.mean(librosa.feature.zero_crossing_rate(waveform)[0])
            
            # Pitch calculation (simple version)
            pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr)
            pitch_mean = 0
            pitch_indices = np.where(magnitudes > np.median(magnitudes))[1]
            if len(pitch_indices) > 0:
                pitch_mean = np.mean(pitches[:, pitch_indices])
            
            # Determine likely emotion based on audio features
            likely_emotion = 'neutral'
            
            # High energy and high pitch often correlate with happiness
            if energy > 0.1 and pitch_mean > 200:
                likely_emotion = 'happy'
            # High energy and low pitch might indicate anger
            elif energy > 0.1 and pitch_mean < 150:
                likely_emotion = 'angry'
            # Low energy and low pitch could indicate sadness
            elif energy < 0.05 and pitch_mean < 180:
                likely_emotion = 'sad'
            # High zero crossing rate might indicate anxiety
            elif zero_cross > 0.1:
                likely_emotion = 'anxious'
            
            # Generate a simple sentence based on the detected emotion
            # This is just a mock-up - real ASR would use actual speech recognition
            import random
            words = self.emotions.get(likely_emotion, self.emotions['neutral'])
            
            # Generate mock transcript with emotion words
            transcript = f"I am feeling {random.choice(words)} today because of the {random.choice(['situation', 'weather', 'news', 'event', 'meeting'])}."
            
            return transcript
            
        except Exception as e:
            print(f"Error processing audio file: {str(e)}")
            return "Unable to transcribe the audio."
""")
        print("Created simplified speech-to-text implementation")

    # Write a simpler direct file content instead of using complex replacements
    # This avoids potential syntax errors from string manipulation
    train_model_content = """import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Modified imports to use the simple speech-to-text
from app.utils.dataset_handler import DatasetHandler
from app.utils.audio_processor import AudioProcessor
# Force using the simplified speech-to-text to avoid dependency issues
from app.utils.speech_to_text_simple import SimpleSpeechToText as SpeechToText
print("Using simplified speech-to-text (mock transcriptions)")
from app.utils.text_analyzer import TextAnalyzer
from app.utils.emotion_classifier import EmotionClassifier

def main():
    print("Initializing emotion detection model training...")
    
    # Debug - show current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Ensure we have proper paths for Colab
    # If we're running in Colab, data_dir should be absolute path
    is_colab = 'google.colab' in sys.modules
    
    if is_colab:
        # In Colab, use a path relative to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, 'data')
        print(f"Running in Colab, using data directory: {data_dir}")
    else:
        # Local development
        data_dir = 'data'
    
    # Create dataset handler and download dataset
    dataset_handler = DatasetHandler(data_dir=data_dir)
    dataset_handler.download_ravdess()
    
    # Debug - show audio directory contents after download
    audio_dir = os.path.join(data_dir, 'audio')
    print(f"Audio directory exists after download: {os.path.exists(audio_dir)}")
    if os.path.exists(audio_dir):
        files = os.listdir(audio_dir)
        print(f"Audio directory contains {len(files)} files after download")
        if len(files) > 0:
            print(f"Sample files: {files[:5]}")
    
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
        print("\\nTrying to verify dataset directories...")
        
        # Debug information
        if is_colab:
            # In Colab, we already have the project_root defined
            print(f"Project root: {project_root}")
            print(f"Project files: {os.listdir(project_root)}")
        else:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        
        audio_dir = os.path.join(data_dir, 'audio')
        
        print(f"Data directory exists: {os.path.exists(data_dir)}")
        print(f"Audio directory exists: {os.path.exists(audio_dir)}")
        
        if os.path.exists(audio_dir):
            files = os.listdir(audio_dir)
            print(f"Audio directory contains {len(files)} files")
            if len(files) > 0:
                print(f"Sample files: {files[:5]}")
                
        # Check extraction directory (which might still exist if cleanup failed)
        extracted_path = os.path.join(data_dir, 'Audio_Speech_Actors_01-24')
        if os.path.exists(extracted_path):
            print(f"Extracted directory still exists: {extracted_path}")
            print(f"It contains: {os.listdir(extracted_path)}")
            
            # Try to manually move files if needed
            try:
                print("Attempting to manually move files...")
                moved_count = 0
                for actor_dir in os.listdir(extracted_path):
                    actor_path = os.path.join(extracted_path, actor_dir)
                    if os.path.isdir(actor_path):
                        for audio_file in os.listdir(actor_path):
                            if audio_file.endswith('.wav'):
                                src = os.path.join(actor_path, audio_file)
                                dst = os.path.join(audio_dir, audio_file)
                                if not os.path.exists(dst):
                                    shutil.copy(src, dst)
                                    moved_count += 1
                print(f"Manually moved {moved_count} files")
                
                if moved_count > 0:
                    # Try to regenerate metadata
                    print("Regenerating metadata...")
                    dataset_handler._create_metadata()
                    # Try again to load the dataset
                    train_df, val_df, test_df = dataset_handler.split_dataset()
                    if len(train_df) > 0:
                        print(f"Successfully recovered! Dataset split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples")
                    else:
                        print("Failed to recover dataset despite finding audio files.")
                        return
            except Exception as e:
                print(f"Error during recovery attempt: {str(e)}")
                return
        else:
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
        model_path = os.path.join(data_dir, 'models', 'emotion_classifier.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
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
        model_path = os.path.join(data_dir, 'models', 'emotion_classifier.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
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
    model_path = os.path.join(data_dir, 'models', 'emotion_classifier.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    emotion_classifier.save_model(model_path)
    print(f"Model saved to {model_path}")"""

    # Write the fixed train_model.py file with proper syntax
    with open("app/train_model.py", "w") as f:
        f.write(train_model_content)
    print("Updated app/train_model.py with fixed syntax")

    # The rest of the script remains unchanged
    print("Now creating run scripts for Colab...")
    
    # Create direct run scripts for Colab
    with open("colab_train.py", "w") as f:
        f.write("""#!/usr/bin/env python3

import os
import sys

# Add absolute path to the project
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Create necessary directories
os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
os.makedirs(os.path.join(project_root, "data", "audio"), exist_ok=True)
os.makedirs(os.path.join(project_root, "data", "models"), exist_ok=True)

# Debug info
print(f"Project root: {project_root}")
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")
print(f"Data directories created.")

# Import main function directly
from app.train_model import main

if __name__ == "__main__":
    main()
""")
    print("Created colab_train.py")
    
    with open("colab_evaluate.py", "w") as f:
        f.write("""#!/usr/bin/env python3

import os
import sys

# Add absolute path to the project
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Create necessary directories
os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
os.makedirs(os.path.join(project_root, "data", "audio"), exist_ok=True)
os.makedirs(os.path.join(project_root, "data", "models"), exist_ok=True)

# Debug info
print(f"Project root: {project_root}")
print(f"Working directory: {os.getcwd()}")

# Import main function directly
from app.evaluate_model import main

if __name__ == "__main__":
    main()
""")
    print("Created colab_evaluate.py")
    
    with open("colab_app.py", "w") as f:
        f.write("""#!/usr/bin/env python3

import os
import sys
import subprocess

# Add absolute path to the project
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Create necessary directories
os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
os.makedirs(os.path.join(project_root, "data", "audio"), exist_ok=True)
os.makedirs(os.path.join(project_root, "data", "models"), exist_ok=True)

# Debug info
print(f"Project root: {project_root}")
print(f"Working directory: {os.getcwd()}")

if __name__ == "__main__":
    # Run Streamlit app
    subprocess.run([sys.executable, "-m", "streamlit", "run", 
                   os.path.join(project_root, "app", "app.py")])
""")
    print("Created colab_app.py")
    
    print("\nFix completed for Google Colab!")
    print("In Colab, run the following commands:")
    print("  python colab_train.py      # To train the model")
    print("  python colab_evaluate.py   # To evaluate the model")
    print("  python colab_app.py        # To run the web app")

if __name__ == "__main__":
    main() 