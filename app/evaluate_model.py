import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import json

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

# Try to use extended analyzers and classifiers
try:
    from utils.text_analyzer_extended import TextAnalyzerExtended as TextAnalyzer
    from utils.emotion_classifier_extended import EmotionClassifierExtended as EmotionClassifier
    print("Using extended text analyzer and emotion classifier with sarcasm detection")
    has_sarcasm_detection = True
except ImportError:
    from utils.text_analyzer import TextAnalyzer
    from utils.emotion_classifier import EmotionClassifier
    print("Using basic text analyzer and emotion classifier")
    has_sarcasm_detection = False

# Import multimodal analyzer for enhanced integration
from utils.multimodal_analyzer import MultimodalAnalyzer

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt

# Function to plot comparison between standard classifier and multimodal analyzer
def plot_comparison(standard_metrics, multimodal_metrics, emotions):
    plt.figure(figsize=(12, 8))
    
    standard_f1 = [standard_metrics['per_class_metrics'][e]['f1'] for e in emotions]
    multimodal_f1 = [multimodal_metrics['per_class_metrics'][e]['f1'] for e in emotions]
    
    x = np.arange(len(emotions))
    width = 0.35
    
    plt.bar(x - width/2, standard_f1, width, label='Standard Classifier', color='cornflowerblue')
    plt.bar(x + width/2, multimodal_f1, width, label='Multimodal Analyzer', color='lightcoral')
    
    plt.xlabel('Emotion')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison: Standard vs. Multimodal Analyzer')
    plt.xticks(x, emotions)
    plt.ylim(0, 1.0)
    
    # Add text labels
    for i, (s, m) in enumerate(zip(standard_f1, multimodal_f1)):
        plt.text(i - width/2, s + 0.02, f"{s:.2f}", ha='center')
        plt.text(i + width/2, m + 0.02, f"{m:.2f}", ha='center')
    
    plt.legend()
    plt.tight_layout()
    return plt

def main():
    print("Evaluating emotion detection model...")
    
    # Check if model exists
    model_path = 'data/models/emotion_classifier.pkl'
    extended_model_path = 'data/models/emotion_classifier_extended.pkl'
    
    # Try to load extended model first, fall back to basic model
    if os.path.exists(extended_model_path):
        selected_model_path = extended_model_path
        print(f"Using extended model from {extended_model_path}")
    elif os.path.exists(model_path):
        selected_model_path = model_path
        print(f"Using basic model from {model_path}")
    else:
        print(f"Error: No model found at {model_path} or {extended_model_path}")
        print("Please run train_model.py first to train the model.")
        return
    
    # Create output directory for evaluation results
    results_dir = 'data/evaluation'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create dataset handler and load test set
    dataset_handler = DatasetHandler()
    _, _, test_df = dataset_handler.split_dataset()
    print(f"Test set size: {len(test_df)} samples")
    
    # Initialize processors
    audio_processor = AudioProcessor()
    speech_to_text = SpeechToText()
    text_analyzer = TextAnalyzer()
    emotion_classifier = EmotionClassifier(model_path=selected_model_path)
    
    # Initialize multimodal analyzer
    multimodal_analyzer = MultimodalAnalyzer(
        audio_processor=audio_processor,
        speech_to_text=speech_to_text,
        text_analyzer=text_analyzer,
        emotion_classifier=emotion_classifier,
        weights_path='data/models/multimodal_weights.json' if os.path.exists('data/models/multimodal_weights.json') else None
    )
    
    # Extract features from test data
    print("Extracting features from test data...")
    test_audio_features = []
    test_text_features = []
    test_labels = []
    test_text_dicts = []
    test_paths = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        try:
            # Extract audio features
            audio_feat, _ = audio_processor.extract_features_for_model(row['path'])
            
            # Transcribe speech to text
            transcription = speech_to_text.transcribe(row['path'])
            
            # Extract text features
            text_feat, text_dict = text_analyzer.extract_features_for_model(transcription)
            
            # Store features and label
            test_audio_features.append(audio_feat)
            test_text_features.append(text_feat)
            test_labels.append(row['emotion'])
            test_text_dicts.append(text_dict)
            test_paths.append(row['path'])
        except Exception as e:
            print(f"Error processing {row['filename']}: {str(e)}")
    
    # Evaluate on test set with standard classifier
    print("Evaluating on test set with standard classifier...")
    standard_metrics = emotion_classifier.evaluate(
        audio_features=test_audio_features,
        text_features=test_text_features,
        true_labels=test_labels,
        text_dicts=test_text_dicts
    )
    
    print(f"Standard classifier accuracy: {standard_metrics['accuracy']:.4f}")
    
    # Evaluate with multimodal analyzer
    print("Evaluating on test set with multimodal analyzer...")
    multimodal_predictions = []
    multimodal_confidences = []
    multimodal_agreements = []
    multimodal_weights = []
    
    for i, path in enumerate(tqdm(test_paths)):
        try:
            # Use multimodal analyzer for prediction
            result = multimodal_analyzer.analyze(path)
            
            # Store prediction and metrics
            multimodal_predictions.append(result['emotion'])
            multimodal_confidences.append(result['confidence_scores'])
            multimodal_agreements.append(result['agreement_score'])
            multimodal_weights.append(result['modality_weights'])
        except Exception as e:
            print(f"Error in multimodal analysis for sample {i}: {str(e)}")
            # Add placeholder values for failed predictions
            multimodal_predictions.append('neutral')
            multimodal_confidences.append({emotion: 0.0 for emotion in emotion_classifier.emotions})
            multimodal_agreements.append(0.0)
            multimodal_weights.append({'audio': 0.5, 'text': 0.5})
    
    # Calculate metrics for multimodal analyzer
    multimodal_accuracy = np.mean([p == t for p, t in zip(multimodal_predictions, test_labels)])
    print(f"Multimodal analyzer accuracy: {multimodal_accuracy:.4f}")
    
    # Calculate per-class metrics for multimodal analyzer
    multimodal_metrics = {
        'accuracy': multimodal_accuracy,
        'per_class_metrics': {}
    }
    
    for emotion in emotion_classifier.emotions:
        true_positives = sum(1 for p, t in zip(multimodal_predictions, test_labels) if p == t == emotion)
        false_positives = sum(1 for p, t in zip(multimodal_predictions, test_labels) if p == emotion and t != emotion)
        false_negatives = sum(1 for p, t in zip(multimodal_predictions, test_labels) if p != emotion and t == emotion)
        
        # Support (number of true instances of this class)
        support = sum(1 for t in test_labels if t == emotion)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        multimodal_metrics['per_class_metrics'][emotion] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    # Print per-class metrics for both approaches
    print("\nPer-class metrics comparison:")
    for emotion in emotion_classifier.emotions:
        standard_f1 = standard_metrics['per_class_metrics'][emotion]['f1'] if emotion in standard_metrics['per_class_metrics'] else 0
        multimodal_f1 = multimodal_metrics['per_class_metrics'][emotion]['f1'] if emotion in multimodal_metrics['per_class_metrics'] else 0
        print(f"  {emotion}:")
        print(f"    Standard F1: {standard_f1:.4f}")
        print(f"    Multimodal F1: {multimodal_f1:.4f}")
        print(f"    Improvement: {(multimodal_f1 - standard_f1):.4f}")
    
    # Compute and plot confusion matrix for multimodal analyzer
    multimodal_cm = confusion_matrix(test_labels, multimodal_predictions)
    cm_plot = plot_confusion_matrix(
        multimodal_cm, 
        classes=sorted(set(test_labels)),
        title='Multimodal Analyzer Confusion Matrix'
    )
    cm_plot.savefig(os.path.join(results_dir, 'multimodal_confusion_matrix.png'))
    
    # Plot comparison of F1 scores
    comparison_plot = plot_comparison(
        standard_metrics,
        multimodal_metrics,
        emotion_classifier.emotions
    )
    comparison_plot.savefig(os.path.join(results_dir, 'f1_comparison.png'))
    
    # Save results as JSON
    evaluation_results = {
        'standard': {
            'accuracy': standard_metrics['accuracy'],
            'per_class_metrics': standard_metrics['per_class_metrics']
        },
        'multimodal': {
            'accuracy': multimodal_metrics['accuracy'],
            'per_class_metrics': multimodal_metrics['per_class_metrics']
        }
    }
    
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Calculate agreement statistics
    print("\nAgreement analysis:")
    agreement_mean = np.mean(multimodal_agreements)
    agreement_std = np.std(multimodal_agreements)
    print(f"  Mean agreement score: {agreement_mean:.4f}")
    print(f"  Agreement std dev: {agreement_std:.4f}")
    
    # Analyze modality weights
    audio_weights = [w['audio'] for w in multimodal_weights]
    text_weights = [w['text'] for w in multimodal_weights]
    
    print("\nModality weight analysis:")
    print(f"  Mean audio weight: {np.mean(audio_weights):.4f}")
    print(f"  Mean text weight: {np.mean(text_weights):.4f}")
    
    # Generate weight plot per emotion
    emotion_weights = {}
    for emotion in set(test_labels):
        indices = [i for i, e in enumerate(test_labels) if e == emotion]
        emotion_weights[emotion] = {
            'audio': np.mean([multimodal_weights[i]['audio'] for i in indices]) if indices else 0,
            'text': np.mean([multimodal_weights[i]['text'] for i in indices]) if indices else 0
        }
    
    # Plot emotion-specific weights
    plt.figure(figsize=(10, 6))
    emotions = list(emotion_weights.keys())
    audio_weights = [emotion_weights[e]['audio'] for e in emotions]
    text_weights = [emotion_weights[e]['text'] for e in emotions]
    
    x = np.arange(len(emotions))
    width = 0.35
    
    plt.bar(x - width/2, audio_weights, width, label='Audio', color='cornflowerblue')
    plt.bar(x + width/2, text_weights, width, label='Text', color='lightcoral')
    
    plt.xlabel('Emotion')
    plt.ylabel('Average Weight')
    plt.title('Average Modality Weights by Emotion')
    plt.xticks(x, emotions)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'modality_weights_by_emotion.png'))
    
    print(f"\nEvaluation results and visualizations saved to {results_dir}")

if __name__ == "__main__":
    main() 