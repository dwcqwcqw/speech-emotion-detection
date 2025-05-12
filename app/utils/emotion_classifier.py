import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import joblib
from collections import Counter

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
            # Initialize a model better suited for imbalanced datasets
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                subsample=0.8,
                random_state=42
            )
    
    def preprocess_features(self, audio_features, text_features, train=False):
        """
        Preprocess and combine audio and text features.
        
        Args:
            audio_features: Audio feature vector or batch of vectors
            text_features: Text feature vector or batch of vectors
            train: Whether to fit the scalers (training) or just transform (inference)
            
        Returns:
            numpy.ndarray: Combined feature vector(s)
        """
        # Check if we have a batch or single sample
        is_batch = len(np.array(audio_features).shape) > 1
        
        # For a single sample, reshape to 2D for scikit-learn
        if not is_batch:
            audio_features = np.array(audio_features).reshape(1, -1)
            text_features = np.array(text_features).reshape(1, -1)
        else:
            audio_features = np.array(audio_features)
            text_features = np.array(text_features)
        
        # Normalize features
        if train:
            audio_scaled = self.audio_scaler.fit_transform(audio_features)
            text_scaled = self.text_scaler.fit_transform(text_features)
        else:
            audio_scaled = self.audio_scaler.transform(audio_features)
            text_scaled = self.text_scaler.transform(text_features)
        
        # Combine features
        combined_features = np.hstack([audio_scaled, text_scaled])
        
        # For single sample, flatten back to 1D for further processing if needed
        if not is_batch and len(combined_features) == 1:
            combined_features = combined_features.flatten()
        
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
        # Print original class distribution
        class_counts = Counter(labels)
        print(f"Original class distribution: {class_counts}")
        
        # Prepare training data
        X_train = []
        for audio_feat, text_feat in zip(audio_features, text_features):
            # Skip samples with invalid features
            if np.any(np.isnan(audio_feat)) or np.any(np.isnan(text_feat)):
                continue
            X_train.append((audio_feat, text_feat))
        
        # Check if we have enough samples
        if len(X_train) < len(self.emotions):
            print("Warning: Not enough valid samples for training!")
            return 0.0
        
        # Unpack the valid samples
        valid_audio_features = [x[0] for x in X_train]
        valid_text_features = [x[1] for x in X_train]
        valid_labels = [labels[i] for i in range(len(labels)) if i < len(X_train)]
        
        # Preprocess features
        X_train = self.preprocess_features(valid_audio_features, valid_text_features, train=True)
        y_train = np.array(valid_labels)
        
        # Calculate class weights to handle imbalance
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {emotion: weight for emotion, weight in zip(np.unique(y_train), class_weights)}
        print(f"Class weights: {class_weight_dict}")
        
        # Apply SMOTE for oversampling
        try:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: {Counter(y_resampled)}")
        except Exception as e:
            print(f"SMOTE failed: {str(e)}. Using original data.")
            X_resampled, y_resampled = X_train, y_train
        
        # Train the model with class weights
        self.model.fit(
            X_resampled, 
            y_resampled,
            sample_weight=[class_weight_dict[label] for label in y_resampled]
        )
        
        # Calculate training accuracy
        y_pred = self.model.predict(X_resampled)
        accuracy = np.mean(y_pred == y_resampled)
        
        # Print feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            feature_importances = self.model.feature_importances_
            print("Top 10 important features:")
            indices = np.argsort(feature_importances)[-10:]
            for i in indices:
                print(f"Feature {i}: {feature_importances[i]:.4f}")
        
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
        # Skip prediction if features are invalid
        if np.any(np.isnan(audio_features)) or np.any(np.isnan(text_features)):
            return 'neutral', {emotion: 0.0 for emotion in self.emotions}
        
        # Preprocess features
        combined_features = self.preprocess_features(audio_features, text_features)
        
        # Reshape for prediction if needed
        if len(combined_features.shape) == 1:
            combined_features = combined_features.reshape(1, -1)
        
        # Make prediction
        emotion_probabilities = self.model.predict_proba(combined_features)[0]
        predicted_idx = np.argmax(emotion_probabilities)
        
        # Map the prediction to emotion label
        predicted_emotion = self.model.classes_[predicted_idx]
        
        # Create dictionary of confidence scores
        confidence_scores = {}
        for i, emotion in enumerate(self.model.classes_):
            confidence_scores[emotion] = emotion_probabilities[i]
            
        # Ensure all emotions have scores
        for emotion in self.emotions:
            if emotion not in confidence_scores:
                confidence_scores[emotion] = 0.0
        
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
        # Filter out invalid samples
        valid_samples = []
        valid_labels = []
        for i, (audio_feat, text_feat) in enumerate(zip(audio_features, text_features)):
            if not (np.any(np.isnan(audio_feat)) or np.any(np.isnan(text_feat))):
                valid_samples.append((audio_feat, text_feat))
                valid_labels.append(true_labels[i])
        
        # Check if we have enough valid samples
        if len(valid_samples) == 0:
            print("Warning: No valid samples for evaluation!")
            return {
                'accuracy': 0.0,
                'per_class_metrics': {emotion: {'precision': 0, 'recall': 0, 'f1': 0} for emotion in self.emotions}
            }
        
        # Unpack the valid samples
        valid_audio_features = [x[0] for x in valid_samples]
        valid_text_features = [x[1] for x in valid_samples]
        
        # Preprocess features
        X_test = self.preprocess_features(valid_audio_features, valid_text_features)
        y_test = np.array(valid_labels)
        
        # Show test class distribution
        print(f"Test set class distribution: {Counter(y_test)}")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for emotion in self.emotions:
            # Skip emotions not in the test set
            if emotion not in y_test and emotion not in y_pred:
                per_class_metrics[emotion] = {
                    'precision': 0,
                    'recall': 0,
                    'f1': 0,
                    'support': 0
                }
                continue
                
            true_positives = np.sum((y_test == emotion) & (y_pred == emotion))
            false_positives = np.sum((y_test != emotion) & (y_pred == emotion))
            false_negatives = np.sum((y_test == emotion) & (y_pred != emotion))
            
            # Support (number of true instances of this class)
            support = np.sum(y_test == emotion)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[emotion] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            }
        
        # Print confusion matrix (in text form)
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        print("\nConfusion Matrix:")
        print("Predicted ->")
        header = "True ↓ " + " ".join([f"{label[:3]:>5}" for label in unique_labels])
        print(header)
        
        for true_label in unique_labels:
            row = [f"{true_label[:3]:>5}"]
            for pred_label in unique_labels:
                count = np.sum((y_test == true_label) & (y_pred == pred_label))
                row.append(f"{count:>5}")
            print(" ".join(row))
        
        return {
            'accuracy': accuracy,
            'per_class_metrics': per_class_metrics
        } 