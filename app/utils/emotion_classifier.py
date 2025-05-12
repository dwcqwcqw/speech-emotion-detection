import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class EmotionClassifier:
    """
    Class for classifying emotions based on both audio and text features.
    Supports training, evaluation, and prediction.
    """
    
    def __init__(self, model_path=None):
        self.emotions = ['happy', 'sad', 'angry', 'anxious', 'neutral']
        self.audio_scaler = StandardScaler()
        self.text_scaler = StandardScaler()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize a new model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42
            )
    
    def preprocess_features(self, audio_features, text_features, train=False):
        """
        Preprocess and combine audio and text features.
        
        Args:
            audio_features: Audio feature vector
            text_features: Text feature vector
            train: Whether to fit the scalers (training) or just transform (inference)
            
        Returns:
            numpy.ndarray: Combined feature vector
        """
        # Normalize features
        if train:
            audio_scaled = self.audio_scaler.fit_transform(audio_features.reshape(1, -1))
            text_scaled = self.text_scaler.fit_transform(text_features.reshape(1, -1))
        else:
            audio_scaled = self.audio_scaler.transform(audio_features.reshape(1, -1))
            text_scaled = self.text_scaler.transform(text_features.reshape(1, -1))
        
        # Combine features
        combined_features = np.concatenate([audio_scaled.flatten(), text_scaled.flatten()])
        
        return combined_features
    
    def train(self, audio_features, text_features, labels):
        """
        Train the emotion classifier.
        
        Args:
            audio_features: List of audio feature vectors
            text_features: List of text feature vectors
            labels: List of emotion labels
            
        Returns:
            float: Training accuracy
        """
        # Prepare training data
        X_train = []
        for audio_feat, text_feat in zip(audio_features, text_features):
            combined = self.preprocess_features(audio_feat, text_feat, train=True)
            X_train.append(combined)
        
        X_train = np.array(X_train)
        y_train = np.array(labels)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate training accuracy
        y_pred = self.model.predict(X_train)
        accuracy = np.mean(y_pred == y_train)
        
        return accuracy
    
    def predict(self, audio_features, text_features):
        """
        Predict emotion based on audio and text features.
        
        Args:
            audio_features: Audio feature vector
            text_features: Text feature vector
            
        Returns:
            tuple: (Predicted emotion, Confidence scores for each emotion)
        """
        # Preprocess features
        combined_features = self.preprocess_features(audio_features, text_features)
        
        # Make prediction
        emotion_probabilities = self.model.predict_proba(combined_features.reshape(1, -1))[0]
        predicted_idx = np.argmax(emotion_probabilities)
        predicted_emotion = self.emotions[predicted_idx]
        
        # Create dictionary of confidence scores
        confidence_scores = {emotion: emotion_probabilities[i] for i, emotion in enumerate(self.emotions)}
        
        return predicted_emotion, confidence_scores
    
    def save_model(self, model_path='data/models/emotion_classifier.pkl'):
        """Save the model, scalers, and emotions list."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'audio_scaler': self.audio_scaler,
            'text_scaler': self.text_scaler,
            'emotions': self.emotions
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, model_path):
        """Load the model, scalers, and emotions list."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.audio_scaler = model_data['audio_scaler']
        self.text_scaler = model_data['text_scaler']
        self.emotions = model_data['emotions']
        
    def evaluate(self, audio_features, text_features, true_labels):
        """
        Evaluate the model on test data.
        
        Args:
            audio_features: List of audio feature vectors
            text_features: List of text feature vectors
            true_labels: List of true emotion labels
            
        Returns:
            dict: Evaluation metrics
        """
        # Prepare test data
        X_test = []
        for audio_feat, text_feat in zip(audio_features, text_features):
            combined = self.preprocess_features(audio_feat, text_feat)
            X_test.append(combined)
        
        X_test = np.array(X_test)
        y_test = np.array(true_labels)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for emotion in self.emotions:
            true_positives = np.sum((y_test == emotion) & (y_pred == emotion))
            false_positives = np.sum((y_test != emotion) & (y_pred == emotion))
            false_negatives = np.sum((y_test == emotion) & (y_pred != emotion))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[emotion] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return {
            'accuracy': accuracy,
            'per_class_metrics': per_class_metrics
        } 