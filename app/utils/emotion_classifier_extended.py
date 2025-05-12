import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import joblib
from collections import Counter

class EmotionClassifierExtended:
    """
    Extended emotion classifier that includes sarcasm detection.
    Combines audio emotion and text-based sarcasm detection.
    """
    
    def __init__(self, model_path=None):
        # Extended emotion list with sarcasm
        self.emotions = ['happy', 'sad', 'angry', 'anxious', 'neutral', 'sarcastic']
        self.audio_scaler = StandardScaler()
        self.text_scaler = StandardScaler()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize a model better suited for imbalanced datasets with sarcasm
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
    
    def _check_sarcasm_features(self, text_features):
        """
        Check if the text features contain sarcasm-related features.
        Used to determine if we should include sarcasm in prediction.
        """
        sarcasm_keys = [
            'sarcasm_score_rule', 
            'sarcasm_score_model', 
            'sarcastic_words'
        ]
        
        for key in sarcasm_keys:
            if key in text_features:
                return True
        return False
    
    def train(self, audio_features, text_features, labels, sarcasm_samples=None):
        """
        Train the emotion classifier with sarcasm support.
        
        Args:
            audio_features: List of audio feature vectors
            text_features: List of text feature vectors
            labels: List of emotion labels
            sarcasm_samples: Optional list of (audio_feat, text_feat, label) tuples for sarcasm samples
            
        Returns:
            float: Training accuracy
        """
        # Print original class distribution
        class_counts = Counter(labels)
        print(f"Original class distribution: {class_counts}")
        
        # Prepare training data
        X_train = []
        valid_labels = []
        
        for i, (audio_feat, text_feat) in enumerate(zip(audio_features, text_features)):
            # Skip samples with invalid features
            if np.any(np.isnan(audio_feat)) or np.any(np.isnan(text_feat)):
                continue
            
            # Add to valid samples
            X_train.append((audio_feat, text_feat))
            if i < len(labels):
                valid_labels.append(labels[i])
            else:
                valid_labels.append('neutral')  # Default for any extras
        
        # Add sarcasm samples if provided
        if sarcasm_samples:
            print(f"Adding {len(sarcasm_samples)} sarcasm samples")
            for audio_feat, text_feat, _ in sarcasm_samples:
                if not (np.any(np.isnan(audio_feat)) or np.any(np.isnan(text_feat))):
                    X_train.append((audio_feat, text_feat))
                    valid_labels.append('sarcastic')
        
        # Check if we have enough samples
        if len(X_train) < len(self.emotions):
            print("Warning: Not enough valid samples for training!")
            return 0.0
        
        # Unpack the valid samples
        valid_audio_features = [x[0] for x in X_train]
        valid_text_features = [x[1] for x in X_train]
        
        # Preprocess features
        X_train = self.preprocess_features(valid_audio_features, valid_text_features, train=True)
        y_train = np.array(valid_labels)
        
        # Calculate class weights to handle imbalance
        try:
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = {emotion: weight for emotion, weight in zip(np.unique(y_train), class_weights)}
            print(f"Class weights: {class_weight_dict}")
            
            # If sarcastic isn't in the class weights, add it with a high weight
            if 'sarcastic' not in class_weight_dict and 'sarcastic' in self.emotions:
                max_weight = max(class_weight_dict.values()) if class_weight_dict else 2.0
                class_weight_dict['sarcastic'] = max_weight * 1.5
                print(f"Added special weight for sarcastic class: {class_weight_dict['sarcastic']}")
        except Exception as e:
            print(f"Warning: Could not compute class weights: {str(e)}")
            class_weight_dict = {emotion: 1.0 for emotion in self.emotions}
        
        # Apply SMOTE for oversampling
        try:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: {Counter(y_resampled)}")
        except Exception as e:
            print(f"SMOTE failed: {str(e)}. Using original data.")
            X_resampled, y_resampled = X_train, y_train
        
        # Create sample weights array based on class weights
        sample_weights = np.array([class_weight_dict.get(label, 1.0) for label in y_resampled])
        
        # Train the model with sample weights
        self.model.fit(X_resampled, y_resampled, sample_weight=sample_weights)
        
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
    
    def predict(self, audio_features, text_features, text_dict=None):
        """
        Predict emotion based on audio and text features, with sarcasm detection.
        
        Args:
            audio_features: Audio feature vector
            text_features: Text feature vector
            text_dict: Optional dictionary of extracted text features
            
        Returns:
            tuple: (Predicted emotion, Confidence scores for each emotion)
        """
        # Skip prediction if features are invalid
        if np.any(np.isnan(audio_features)) or np.any(np.isnan(text_features)):
            return 'neutral', {emotion: 0.0 for emotion in self.emotions}
        
        # Handle sarcasm detection if we have text dictionary with sarcasm features
        if text_dict and self._check_sarcasm_features(text_dict):
            # If high confidence in sarcasm, override prediction
            sarcasm_score = max(
                text_dict.get('sarcasm_score_rule', 0),
                text_dict.get('sarcasm_score_model', 0)
            )
            
            # If sarcasm score is very high, directly return sarcasm
            if sarcasm_score > 0.85:
                confidence_scores = {emotion: 0.0 for emotion in self.emotions}
                confidence_scores['sarcastic'] = sarcasm_score
                return 'sarcastic', confidence_scores
        
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
        
        # Adjust sarcasm score if we have text features
        if text_dict and self._check_sarcasm_features(text_dict):
            sarcasm_score = max(
                text_dict.get('sarcasm_score_rule', 0),
                text_dict.get('sarcasm_score_model', 0)
            )
            
            # Blend model prediction with rule-based sarcasm score
            confidence_scores['sarcastic'] = max(
                confidence_scores.get('sarcastic', 0),
                sarcasm_score * 0.7  # Weight the rule-based score
            )
            
            # If sarcasm is now the highest confidence, update prediction
            if confidence_scores['sarcastic'] > max(
                score for emotion, score in confidence_scores.items() if emotion != 'sarcastic'
            ):
                predicted_emotion = 'sarcastic'
        
        return predicted_emotion, confidence_scores
    
    def save_model(self, model_path='data/models/emotion_classifier_extended.pkl'):
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
        
        # Ensure 'sarcastic' is in emotions list
        if 'sarcastic' not in self.emotions:
            self.emotions.append('sarcastic')
        
    def evaluate(self, audio_features, text_features, true_labels, text_dicts=None):
        """
        Evaluate the model on test data.
        
        Args:
            audio_features: List of audio feature vectors
            text_features: List of text feature vectors
            true_labels: List of true emotion labels
            text_dicts: Optional list of text feature dictionaries
            
        Returns:
            dict: Evaluation metrics
        """
        # Filter out invalid samples
        valid_samples = []
        valid_labels = []
        valid_text_dicts = []
        
        for i, (audio_feat, text_feat) in enumerate(zip(audio_features, text_features)):
            if not (np.any(np.isnan(audio_feat)) or np.any(np.isnan(text_feat))):
                valid_samples.append((audio_feat, text_feat))
                if i < len(true_labels):
                    valid_labels.append(true_labels[i])
                else:
                    valid_labels.append('neutral')  # Default for any extras
                
                if text_dicts and i < len(text_dicts):
                    valid_text_dicts.append(text_dicts[i])
                else:
                    valid_text_dicts.append(None)
        
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
        
        # Make predictions
        y_pred = []
        for i, (audio_feat, text_feat) in enumerate(zip(valid_audio_features, valid_text_features)):
            text_dict = valid_text_dicts[i] if i < len(valid_text_dicts) else None
            emotion, _ = self.predict(audio_feat, text_feat, text_dict)
            y_pred.append(emotion)
        
        y_test = np.array(valid_labels)
        y_pred = np.array(y_pred)
        
        # Show test class distribution
        print(f"Test set class distribution: {Counter(y_test)}")
        print(f"Prediction distribution: {Counter(y_pred)}")
        
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
        all_labels = list(set(np.unique(y_test)) | set(np.unique(y_pred)))
        all_labels.sort()
        
        print("\nConfusion Matrix:")
        print("Predicted ->")
        header = "True ↓ " + " ".join([f"{label[:3]:>5}" for label in all_labels])
        print(header)
        
        for true_label in all_labels:
            row = [f"{true_label[:3]:>5}"]
            for pred_label in all_labels:
                count = np.sum((y_test == true_label) & (y_pred == pred_label))
                row.append(f"{count:>5}")
            print(" ".join(row))
        
        return {
            'accuracy': accuracy,
            'per_class_metrics': per_class_metrics
        } 