import os
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
try:
    # Try to import the transformer-based speech-to-text
    from app.utils.speech_to_text import SpeechToText
    print("Using transformer-based speech-to-text")
except ImportError:
    # Fall back to the simple speech-to-text
    from app.utils.speech_to_text_simple import SimpleSpeechToText as SpeechToText
    print("Using simplified speech-to-text (mock transcriptions)")

# Try to use extended analyzers and classifiers
try:
    from app.utils.text_analyzer_extended import TextAnalyzerExtended as TextAnalyzer
    from app.utils.emotion_classifier_extended import EmotionClassifierExtended as EmotionClassifier
    print("Using extended text analyzer and emotion classifier with sarcasm detection")
    has_sarcasm_detection = True
except ImportError:
    from app.utils.text_analyzer import TextAnalyzer
    from app.utils.emotion_classifier import EmotionClassifier
    print("Using basic text analyzer and emotion classifier")
    has_sarcasm_detection = False

# Import the multimodal analyzer for training validation
from app.utils.multimodal_analyzer import MultimodalAnalyzer

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
        print("\nTrying to verify dataset directories...")
        
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
    
    # Determine model type based on availability
    if has_sarcasm_detection:
        print("Training with sarcasm detection...")
        emotion_classifier = EmotionClassifier()
        output_model_path = os.path.join(data_dir, 'models', 'emotion_classifier_extended.pkl')
    else:
        print("Training basic emotion classifier...")
        emotion_classifier = EmotionClassifier()
        output_model_path = os.path.join(data_dir, 'models', 'emotion_classifier.pkl')
    
    # Check for sarcasm dataset
    sarcasm_samples = None
    if has_sarcasm_detection:
        try:
            print("Checking for sarcasm dataset...")
            from download_sarcasm_dataset import load_sarcasm_datasets, process_for_training
            
            # Check if sarcasm dataset exists, if not download it
            sarcasm_path = 'data/sarcasm/combined_sarcasm.csv'
            if not os.path.exists(sarcasm_path):
                print("Downloading sarcasm dataset...")
                process_for_training()
            
            # Load sarcasm dataset
            sarcasm_df = pd.read_csv(sarcasm_path)
            print(f"Loaded sarcasm dataset with {len(sarcasm_df)} samples")
            
            # Prepare sarcasm samples (with mock audio features for now)
            sarcasm_samples = []
            
            # Generate simple mock audio features for sarcasm samples
            print("Preparing sarcasm samples...")
            for _, row in tqdm(sarcasm_df.iterrows(), total=len(sarcasm_df)):
                try:
                    # Use neutral audio features as a base for sarcasm
                    # This is a simplification - in real situations, sarcasm has specific prosodic features
                    mock_audio_features = np.zeros(audio_processor.n_features)
                    
                    # Extract text features for sarcasm
                    text = row['text']
                    text_feat, _ = text_analyzer.extract_features_for_model(text)
                    
                    # Store as sarcasm sample
                    sarcasm_samples.append((mock_audio_features, text_feat, 'sarcastic'))
                except Exception as e:
                    print(f"Error processing sarcasm sample: {str(e)}")
                    
            print(f"Prepared {len(sarcasm_samples)} sarcasm samples for training")
            
        except Exception as e:
            print(f"Error loading sarcasm dataset: {str(e)}")
            sarcasm_samples = None
    
    # Extract features from training data
    print("Extracting features from training data...")
    train_audio_features = []
    train_text_features = []
    train_labels = []
    train_text_dicts = []
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        try:
            # Extract audio features
            audio_feat, _ = audio_processor.extract_features_for_model(row['path'])
            
            # Transcribe speech to text
            transcription = speech_to_text.transcribe(row['path'])
            
            # Extract text features
            text_feat, text_dict = text_analyzer.extract_features_for_model(transcription)
            
            # Store features and label
            train_audio_features.append(audio_feat)
            train_text_features.append(text_feat)
            train_labels.append(row['emotion'])
            train_text_dicts.append(text_dict)
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
        labels=train_labels,
        sarcasm_samples=sarcasm_samples
    )
    
    print(f"Training accuracy: {accuracy:.4f}")
    
    # Check if validation set is empty
    if len(val_df) == 0:
        print("WARNING: No validation data available. Skipping validation.")
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        emotion_classifier.save_model(output_model_path)
        print(f"Model saved to {output_model_path}")
        return
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_audio_features = []
    val_text_features = []
    val_labels = []
    val_text_dicts = []
    
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        try:
            # Extract audio features
            audio_feat, _ = audio_processor.extract_features_for_model(row['path'])
            
            # Transcribe speech to text
            transcription = speech_to_text.transcribe(row['path'])
            
            # Extract text features
            text_feat, text_dict = text_analyzer.extract_features_for_model(transcription)
            
            # Store features and label
            val_audio_features.append(audio_feat)
            val_text_features.append(text_feat)
            val_labels.append(row['emotion'])
            val_text_dicts.append(text_dict)
        except Exception as e:
            print(f"Error processing {row['filename']}: {str(e)}")
    
    # Evaluate model on validation set
    print("Evaluating model on validation set...")
    val_metrics = emotion_classifier.evaluate(
        audio_features=val_audio_features,
        text_features=val_text_features,
        true_labels=val_labels,
        text_dicts=val_text_dicts
    )
    
    print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
    
    # Print per-class metrics
    print("\nPer-class validation metrics:")
    for emotion, metrics in val_metrics['per_class_metrics'].items():
        print(f"  {emotion}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1 Score: {metrics['f1']:.4f}")
        print(f"    Support: {metrics['support']}")
    
    # Save the model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    emotion_classifier.save_model(output_model_path)
    print(f"Model saved to {output_model_path}")
    
    # Initialize and test the multimodal analyzer
    print("Testing multimodal analyzer integration...")
    multimodal_analyzer = MultimodalAnalyzer(
        audio_processor=audio_processor,
        speech_to_text=speech_to_text,
        text_analyzer=text_analyzer,
        emotion_classifier=emotion_classifier
    )
    
    # Test on a few samples
    if len(val_df) > 0:
        sample_size = min(3, len(val_df))
        samples = val_df.sample(sample_size)
        
        print(f"\nTesting multimodal integration on {sample_size} samples:")
        for i, (_, row) in enumerate(samples.iterrows()):
            print(f"\nSample {i+1} - True emotion: {row['emotion']}")
            try:
                # Analyze with multimodal analyzer
                result = multimodal_analyzer.analyze(row['path'])
                
                # Display results
                print(f"Predicted emotion: {result['emotion']}")
                print(f"Transcription: {result['transcription']}")
                print(f"Agreement score: {result['agreement_score']:.2f}")
                print(f"Modality weights: Audio={result['modality_weights']['audio']:.2f}, " +
                      f"Text={result['modality_weights']['text']:.2f}")
                
                # Print confidence scores for top 3 emotions
                top_emotions = sorted(
                    result['confidence_scores'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                print("Top confidence scores:")
                for emotion, score in top_emotions:
                    print(f"  {emotion}: {score:.4f}")
                
            except Exception as e:
                print(f"Error in multimodal analysis: {str(e)}")
    
    # Save multimodal analyzer weights
    weights_path = os.path.join(data_dir, 'models', 'multimodal_weights.json')
    multimodal_analyzer.save_weights(weights_path)
    print(f"Multimodal analyzer weights saved to {weights_path}")

if __name__ == "__main__":
    main() 