import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

class MultimodalAnalyzer:
    """
    Enhanced integration between speech and text analysis for emotion detection.
    This class combines audio prosody with linguistic content analysis, implementing
    smarter fusion strategies and contextual understanding.
    """
    
    def __init__(self, audio_processor=None, speech_to_text=None, text_analyzer=None, 
                 emotion_classifier=None, weights_path=None):
        """
        Initialize the multimodal analyzer.
        
        Args:
            audio_processor: AudioProcessor instance
            speech_to_text: SpeechToText instance
            text_analyzer: TextAnalyzer or TextAnalyzerExtended instance
            emotion_classifier: EmotionClassifier or EmotionClassifierExtended instance
            weights_path: Path to modality weights file (if None, use default weights)
        """
        self.audio_processor = audio_processor
        self.speech_to_text = speech_to_text
        self.text_analyzer = text_analyzer
        self.emotion_classifier = emotion_classifier
        
        # Default modality weights (can be learned or set from external file)
        self.modality_weights = {
            'audio': 0.6,  # Weight for audio/prosodic features
            'text': 0.4,   # Weight for text/linguistic features
            
            # Context-specific weights based on emotion
            'emotion_weights': {
                'happy': {'audio': 0.65, 'text': 0.35},      # Happiness often more evident in voice
                'sad': {'audio': 0.5, 'text': 0.5},          # Sadness equally conveyed
                'angry': {'audio': 0.7, 'text': 0.3},        # Anger strongly conveyed in voice
                'anxious': {'audio': 0.6, 'text': 0.4},      # Anxiety evident in prosody
                'neutral': {'audio': 0.4, 'text': 0.6},      # Neutral more reliant on text
                'sarcastic': {'audio': 0.3, 'text': 0.7}     # Sarcasm heavily reliant on text
            },
            
            # Feature importance for agreement/disagreement detection
            'feature_importance': {
                'pitch_variation': 0.15,
                'energy_variation': 0.15,
                'speech_rate': 0.10,
                'sentiment_contrast': 0.20,
                'lexical_cues': 0.20,
                'sarcasm_indicators': 0.20
            }
        }
        
        # Load custom weights if provided
        if weights_path and os.path.exists(weights_path):
            self._load_weights(weights_path)
            
        # Feature correlation for contextual understanding
        self.context_correlations = {
            'pitch_high+positive_sentiment': 'happy',
            'pitch_low+negative_sentiment': 'sad',
            'energy_high+negative_sentiment': 'angry',
            'pitch_variation+sentiment_contrast': 'sarcastic'
        }
        
        # Initialize scalers for normalization
        self.audio_confidence_scaler = StandardScaler()
        self.text_confidence_scaler = StandardScaler()
    
    def _load_weights(self, weights_path):
        """Load custom modality weights from file."""
        try:
            with open(weights_path, 'r') as f:
                weights = json.load(f)
                self.modality_weights.update(weights)
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
    
    def save_weights(self, weights_path):
        """Save current modality weights to file."""
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        with open(weights_path, 'w') as f:
            json.dump(self.modality_weights, f, indent=2)
    
    def detect_prosody_text_agreement(self, audio_features, text_features):
        """
        Detect if prosodic features align with linguistic content.
        
        Returns:
            float: Agreement score (-1 to 1, where -1 is complete disagreement,
                  0 is neutral, and 1 is complete agreement)
        """
        agreement_score = 0.0
        
        # Extract key features for alignment analysis
        pitch_mean = audio_features.get('pitch_mean', 0)
        pitch_std = audio_features.get('pitch_std', 0)
        energy_mean = audio_features.get('energy_mean', 0)
        tempo = audio_features.get('tempo', 0)
        
        sentiment = text_features.get('sentiment_compound', 0)
        pos_sentiment = text_features.get('sentiment_pos', 0)
        neg_sentiment = text_features.get('sentiment_neg', 0)
        
        # Check for pitch-sentiment alignment
        # High pitch often aligns with positive emotions, low with negative
        pitch_sentiment_align = 0
        if (pitch_mean > 0.5 and sentiment > 0.2) or (pitch_mean < 0.3 and sentiment < -0.2):
            pitch_sentiment_align = 0.5
        elif (pitch_mean > 0.5 and sentiment < -0.2) or (pitch_mean < 0.3 and sentiment > 0.2):
            pitch_sentiment_align = -0.5
            
        # Energy-sentiment alignment
        energy_sentiment_align = 0
        if (energy_mean > 0.5 and abs(sentiment) > 0.3) or (energy_mean < 0.3 and abs(sentiment) < 0.2):
            energy_sentiment_align = 0.3
        
        # Prosodic variation can indicate sarcasm when combined with sentiment indicators
        sarcasm_indicator = 0
        if pitch_std > 0.2 and abs(pos_sentiment - neg_sentiment) < 0.1:
            sarcasm_indicator = -0.4  # Potential sarcasm (disagreement)
        
        # Compute weighted agreement score
        weights = self.modality_weights['feature_importance']
        agreement_score = (
            weights['pitch_variation'] * pitch_sentiment_align +
            weights['energy_variation'] * energy_sentiment_align +
            weights['sentiment_contrast'] * sarcasm_indicator
        )
        
        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, agreement_score * 2))
    
    def adjust_weights_dynamically(self, audio_features, text_features, preliminary_emotion=None):
        """
        Dynamically adjust modality weights based on features and context.
        
        Returns:
            dict: Updated weights for audio and text modalities
        """
        # Start with default weights
        weights = {
            'audio': self.modality_weights['audio'],
            'text': self.modality_weights['text']
        }
        
        # Use emotion-specific weights if preliminary emotion is available
        if preliminary_emotion and preliminary_emotion in self.modality_weights['emotion_weights']:
            weights = self.modality_weights['emotion_weights'][preliminary_emotion].copy()
        
        # Adjust based on feature quality and agreement
        agreement_score = self.detect_prosody_text_agreement(audio_features, text_features)
        
        # If strong agreement, maintain balance
        # If disagreement, determine which modality seems more reliable in this context
        if agreement_score < -0.5:
            # Strong disagreement - check specifics to determine which to trust more
            
            # Check for sarcasm indicators in text
            sarcasm_score = text_features.get('sarcasm_score_rule', 0) + text_features.get('sarcasm_score_model', 0)
            if sarcasm_score > 0.6:
                # For sarcasm, text cues are more important
                weights['text'] += 0.2
                weights['audio'] -= 0.2
            
            # Check for strong emotional audio signals
            audio_emotion_strength = audio_features.get('pitch_std', 0) + audio_features.get('energy_std', 0)
            if audio_emotion_strength > 0.7:
                weights['audio'] += 0.1
            
            # Check text sentiment strength
            text_emotion_strength = abs(text_features.get('sentiment_compound', 0))
            if text_emotion_strength > 0.7:
                weights['text'] += 0.1
        
        # Normalize weights to sum to 1.0
        total = weights['audio'] + weights['text']
        weights['audio'] /= total
        weights['text'] /= total
        
        return weights
    
    def combine_modality_confidences(self, audio_confidences, text_confidences, weights=None):
        """
        Combine confidence scores from audio and text analysis with weighted fusion.
        
        Args:
            audio_confidences: Dict of emotion confidences from audio
            text_confidences: Dict of emotion confidences from text
            weights: Dict of weights for audio and text modalities
            
        Returns:
            dict: Combined confidence scores for each emotion
        """
        if weights is None:
            weights = {
                'audio': self.modality_weights['audio'],
                'text': self.modality_weights['text']
            }
        
        # Ensure all emotions are present in both confidence sets
        all_emotions = set(list(audio_confidences.keys()) + list(text_confidences.keys()))
        for emotion in all_emotions:
            if emotion not in audio_confidences:
                audio_confidences[emotion] = 0.0
            if emotion not in text_confidences:
                text_confidences[emotion] = 0.0
        
        # Combine confidences with weighted average
        combined_confidences = {}
        for emotion in all_emotions:
            combined_confidences[emotion] = (
                weights['audio'] * audio_confidences.get(emotion, 0.0) +
                weights['text'] * text_confidences.get(emotion, 0.0)
            )
        
        return combined_confidences
    
    def analyze(self, audio_path, text=None):
        """
        Perform integrated multimodal analysis on audio and optional text.
        
        Args:
            audio_path: Path to audio file
            text: Optional text transcription (if None, will be extracted from audio)
            
        Returns:
            dict: Analysis results including emotion, confidences, and features
        """
        # Validate input
        if not self.audio_processor or not self.emotion_classifier:
            raise ValueError("Audio processor and emotion classifier must be initialized")
        
        # Extract audio features
        audio_features, audio_features_dict = self.audio_processor.extract_features_for_model(audio_path)
        
        # Get transcription either from parameter or by extracting from audio
        transcription = text
        if not transcription and self.speech_to_text:
            transcription = self.speech_to_text.transcribe(audio_path)
        
        # If we still don't have text, return audio-only analysis
        if not transcription or not self.text_analyzer:
            # Use only audio for prediction
            predicted_emotion, confidence_scores = self.emotion_classifier.predict(
                audio_features=audio_features,
                text_features=np.zeros(10)  # Placeholder for text features
            )
            
            return {
                'emotion': predicted_emotion,
                'confidence_scores': confidence_scores,
                'audio_features': audio_features_dict,
                'text_features': {},
                'transcription': transcription or "",
                'agreement_score': 0.0,
                'modality_weights': {'audio': 1.0, 'text': 0.0}
            }
        
        # Extract text features
        text_features, text_features_dict = self.text_analyzer.extract_features_for_model(transcription)
        
        # Get preliminary emotion prediction for weight adjustment
        preliminary_emotion, _ = self.emotion_classifier.predict(audio_features, text_features)
        
        # Adjust weights based on context
        weights = self.adjust_weights_dynamically(
            audio_features_dict, 
            text_features_dict,
            preliminary_emotion
        )
        
        # Get agreement score between modalities
        agreement_score = self.detect_prosody_text_agreement(audio_features_dict, text_features_dict)
        
        # Special handling for sarcasm if available in the emotion set
        has_sarcasm = 'sarcastic' in self.emotion_classifier.emotions
        if has_sarcasm and 'sarcasm_score_model' in text_features_dict:
            # For potential sarcasm, do a separate prediction with the text analyzer
            sarcasm_score = max(
                text_features_dict.get('sarcasm_score_rule', 0),
                text_features_dict.get('sarcasm_score_model', 0)
            )
            
            # If strong disagreement between modalities and high sarcasm score,
            # increase likelihood of sarcasm detection
            if agreement_score < -0.3 and sarcasm_score > 0.5:
                weights['text'] += 0.2
                weights['audio'] -= 0.2
                # Normalize weights
                total = weights['audio'] + weights['text']
                weights['audio'] /= total
                weights['text'] /= total
        
        # Make final prediction with the emotion classifier
        predicted_emotion, confidence_scores = self.emotion_classifier.predict(
            audio_features, text_features, text_features_dict
        )
        
        # Return comprehensive results
        return {
            'emotion': predicted_emotion,
            'confidence_scores': confidence_scores,
            'audio_features': audio_features_dict,
            'text_features': text_features_dict,
            'transcription': transcription,
            'agreement_score': agreement_score,
            'modality_weights': weights
        }
        
    def analyze_file_batch(self, file_paths, texts=None):
        """
        Analyze a batch of audio files with optional text transcriptions.
        
        Args:
            file_paths: List of paths to audio files
            texts: Optional list of text transcriptions
            
        Returns:
            list: Analysis results for each file
        """
        results = []
        
        for i, file_path in enumerate(file_paths):
            text = None
            if texts and i < len(texts):
                text = texts[i]
                
            try:
                result = self.analyze(file_path, text)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing file {file_path}: {str(e)}")
                results.append({
                    'error': str(e),
                    'file_path': file_path
                })
                
        return results 