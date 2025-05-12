import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

class TextAnalyzerExtended:
    """
    Extended text analyzer for emotion detection with additional sarcasm detection.
    """
    
    def __init__(self, sarcasm_model_path=None):
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
        
        # Initialize VADER sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Emotion lexicons (simplified)
        self.emotion_lexicons = {
            'happy': ['happy', 'joy', 'delighted', 'pleased', 'excited', 'glad', 'cheerful'],
            'sad': ['sad', 'unhappy', 'depressed', 'down', 'miserable', 'gloomy', 'sorrow'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'rage', 'outraged'],
            'anxious': ['anxious', 'worried', 'nervous', 'tense', 'stressed', 'uneasy', 'afraid'],
            'neutral': ['fine', 'okay', 'neutral', 'indifferent', 'normal', 'ordinary', 'common'],
            'sarcastic': ['obviously', 'surely', 'clearly', 'right', 'oh really', 'wow', 'great', 'nice']
        }
        
        # Load sarcasm detection model if available
        self.sarcasm_model = None
        self.sarcasm_tokenizer = None
        
        self._load_sarcasm_model(sarcasm_model_path)
    
    def _load_sarcasm_model(self, model_path=None):
        """
        Load a pre-trained sarcasm detection model or download one from HuggingFace.
        """
        try:
            # If model_path is provided, load local model
            if model_path and os.path.exists(model_path):
                self.sarcasm_tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.sarcasm_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                # Try loading from HuggingFace
                model_name = "deepset/deberta-v3-base-injection"  # This is a versatile model that can be fine-tuned
                
                print("Loading sarcasm detection model...")
                self.sarcasm_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.sarcasm_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                print("Sarcasm detection model loaded successfully.")
            
            self.has_sarcasm_model = True
        except Exception as e:
            print(f"Warning: Could not load sarcasm detection model: {str(e)}")
            print("Falling back to rule-based sarcasm detection.")
            self.has_sarcasm_model = False
    
    def detect_sarcasm_rule_based(self, text):
        """
        Rule-based approach to detect sarcasm in text.
        Returns a score between 0 and 1.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Sarcasm indicators
        sarcasm_indicators = self.emotion_lexicons['sarcastic']
        sarcasm_phrases = ['yeah right', 'sure thing', 'oh really', 'oh wow', 'as if']
        
        # Sentiment contrast (positive words with negative sentiment or vice versa)
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        sentiment_contrast = False
        
        # Check for sentiment contrast between words and overall sentiment
        happy_words = sum(1 for word in tokens if word in self.emotion_lexicons['happy'])
        sad_words = sum(1 for word in tokens if word in self.emotion_lexicons['sad'])
        
        if (happy_words > 0 and sentiment_scores['compound'] < -0.2) or \
           (sad_words > 0 and sentiment_scores['compound'] > 0.5):
            sentiment_contrast = True
        
        # Check for sarcasm phrases
        phrase_indicator = 0
        for phrase in sarcasm_phrases:
            if phrase in text:
                phrase_indicator = 1
                break
        
        # Check for sarcasm indicators
        word_indicators = sum(1 for word in tokens if word in sarcasm_indicators)
        
        # Check for excessive punctuation (e.g., "Sure!!!") or mixed case ("sUre")
        excessive_punct = sum(1 for char in text if char in '!?') > 2
        mixed_case = any(t.lower() != t and t.upper() != t for t in tokens if len(t) > 3)
        
        # Calculate sarcasm score
        score = 0.0
        if sentiment_contrast:
            score += 0.3
        if phrase_indicator:
            score += 0.3
        if word_indicators > 0:
            score += min(0.2, word_indicators * 0.05)
        if excessive_punct:
            score += 0.1
        if mixed_case:
            score += 0.1
            
        return min(1.0, score)
    
    def detect_sarcasm_model_based(self, text):
        """
        Model-based approach to detect sarcasm using transformer model.
        Returns a score between 0 and 1.
        """
        if not self.has_sarcasm_model:
            return self.detect_sarcasm_rule_based(text)
        
        try:
            # Tokenize and get model prediction
            inputs = self.sarcasm_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.sarcasm_model(**inputs)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Assume last class is sarcasm class (need to adjust based on actual model)
            sarcasm_score = probs[0][1].item()  # Assuming binary classification
            
            return sarcasm_score
        except Exception as e:
            print(f"Error in model-based sarcasm detection: {str(e)}")
            return self.detect_sarcasm_rule_based(text)
    
    def extract_sarcasm_features(self, text):
        """
        Extract features relevant for sarcasm detection.
        """
        # Use rule-based method as fallback
        rule_based_score = self.detect_sarcasm_rule_based(text)
        
        # Try model-based detection if available
        if self.has_sarcasm_model:
            model_score = self.detect_sarcasm_model_based(text)
        else:
            model_score = rule_based_score
        
        # Get sentiment scores
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        
        # Lexical features (sarcasm often uses certain emotion words)
        happy_words = sum(1 for word in nltk.word_tokenize(text.lower()) 
                         if word in self.emotion_lexicons['happy'])
        sad_words = sum(1 for word in nltk.word_tokenize(text.lower()) 
                       if word in self.emotion_lexicons['sad'])
        
        # Return sarcasm features
        return {
            'sarcasm_score_rule': rule_based_score,
            'sarcasm_score_model': model_score,
            'sentiment_pos': sentiment['pos'],
            'sentiment_neg': sentiment['neg'],
            'sentiment_compound': sentiment['compound'],
            'happy_words_count': happy_words,
            'sad_words_count': sad_words
        }
    
    def extract_features(self, text):
        """
        Extract linguistic features from text.
        """
        # Tokenize the text
        tokens = nltk.word_tokenize(text.lower())
        
        # Extract sentiment scores using VADER
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        
        # Count emotion words
        emotion_counts = {}
        for emotion, words in self.emotion_lexicons.items():
            emotion_counts[emotion] = sum(1 for token in tokens if token in words)
        
        # Extract sarcasm features
        sarcasm_features = self.extract_sarcasm_features(text)
        
        # Combine all features
        features = {
            'word_count': len(tokens),
            'sentiment_pos': sentiment['pos'],
            'sentiment_neg': sentiment['neg'],
            'sentiment_neu': sentiment['neu'],
            'sentiment_compound': sentiment['compound']
        }
        
        # Add emotion word counts
        for emotion, count in emotion_counts.items():
            features[f'{emotion}_words'] = count
        
        # Add sarcasm features
        for key, value in sarcasm_features.items():
            features[key] = value
        
        return features
    
    def extract_features_for_model(self, text):
        """
        Extract and process features for model input.
        
        Returns:
            tuple: (Feature vector, Feature dictionary)
        """
        # Extract features
        features_dict = self.extract_features(text)
        
        # Convert to vector
        feature_vector = np.array(list(features_dict.values()))
        
        # Normalize if needed (handled by classifier)
        
        return feature_vector, features_dict 