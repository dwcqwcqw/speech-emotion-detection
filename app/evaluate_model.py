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

# 修改导入方式，使用相对导入
from utils.dataset_handler import DatasetHandler
from utils.audio_processor import AudioProcessor
from utils.speech_to_text import SpeechToText
from utils.text_analyzer import TextAnalyzer
from utils.emotion_classifier import EmotionClassifier

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    Plot confusion matrix.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, title="Feature Importance"):
    """
    Plot feature importance from a trained Random Forest model.
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Rearrange feature names so they match the sorted feature importances
    names = [feature_names[i] for i in indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create plot title
    plt.title(title)
    
    # Add bars
    plt.barh(range(len(indices)), importances[indices])
    
    # Add feature names as y-axis labels
    plt.yticks(range(len(indices)), names)
    
    # Add axis labels
    plt.xlabel('Relative Importance')
    
    return fig

def main():
    print("Evaluating emotion detection model...")
    
    # Check if model exists
    model_path = 'data/models/emotion_classifier.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
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
    emotion_classifier = EmotionClassifier(model_path=model_path)
    
    # Extract features from test data
    print("Extracting features from test data...")
    test_audio_features = []
    test_text_features = []
    test_labels = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        try:
            # Extract audio features
            audio_feat, _ = audio_processor.extract_features_for_model(row['path'])
            
            # Transcribe speech to text
            transcription = speech_to_text.transcribe(row['path'])
            
            # Extract text features
            text_feat, _ = text_analyzer.extract_features_for_model(transcription)
            
            # Store features and label
            test_audio_features.append(audio_feat)
            test_text_features.append(text_feat)
            test_labels.append(row['emotion'])
        except Exception as e:
            print(f"Error processing {row['filename']}: {str(e)}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = emotion_classifier.evaluate(
        audio_features=test_audio_features,
        text_features=test_text_features,
        true_labels=test_labels
    )
    
    # Print test metrics
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print("Per-class metrics:")
    
    for emotion, metrics in test_metrics['per_class_metrics'].items():
        print(f"  {emotion}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
    
    # Save test metrics
    with open(os.path.join(results_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    # Make predictions
    print("Generating predictions...")
    predicted_emotions = []
    for audio_feat, text_feat in zip(test_audio_features, test_text_features):
        emotion, _ = emotion_classifier.predict(audio_feat, text_feat)
        predicted_emotions.append(emotion)
    
    # Create confusion matrix
    print("Generating confusion matrix...")
    emotions = emotion_classifier.emotions
    cm_fig = plot_confusion_matrix(
        test_labels, 
        predicted_emotions, 
        classes=emotions, 
        normalize=True, 
        title='Normalized Confusion Matrix'
    )
    cm_fig.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    
    # Generate feature importance plot
    print("Generating feature importance plot...")
    
    # Get feature names from the combined audio and text features
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Access the model's feature names
    audio_feature_example = audio_processor.extract_features(
        audio_processor.load_audio(test_df.iloc[0]['path'])[0],
        audio_processor.load_audio(test_df.iloc[0]['path'])[1]
    )
    text_feature_example = text_analyzer.extract_features("Sample text")
    
    feature_names = list(audio_feature_example.keys()) + list(text_feature_example.keys())
    
    # Create feature importance plot
    importance_fig = plot_feature_importance(
        model_data['model'], 
        feature_names, 
        title="Feature Importance for Emotion Classification"
    )
    importance_fig.savefig(os.path.join(results_dir, 'feature_importance.png'))
    
    print(f"Evaluation results saved to {results_dir}")

if __name__ == "__main__":
    main() 