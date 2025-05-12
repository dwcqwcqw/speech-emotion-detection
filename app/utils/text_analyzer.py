import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import re

class TextAnalyzer:
    """
    Class for analyzing text to extract features for emotion detection.
    """
    
    def __init__(self):
        # Download NLTK resources if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
        
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()
        
        # Emotion word lexicons
        self.emotion_lexicons = {
            'happy': ['happy', 'joy', 'delight', 'ecstatic', 'excited', 'pleased', 'glad', 'content', 'satisfied', 'cheerful'],
            'sad': ['sad', 'unhappy', 'depressed', 'gloomy', 'miserable', 'grief', 'sorrow', 'melancholy', 'downhearted', 'upset'],
            'angry': ['angry', 'furious', 'enraged', 'irate', 'annoyed', 'mad', 'hostile', 'irritated', 'outraged', 'offended'],
            'anxious': ['anxious', 'worried', 'nervous', 'concerned', 'uneasy', 'afraid', 'fearful', 'stressed', 'tense', 'panicked']
        }
    
    def extract_features(self, text):
        """
        Extract features from text for emotion analysis.
        
        Args:
            text: Input text string
            
        Returns:
            dict: Dictionary of text features
        """
        if not text:
            # Return default values if text is empty
            return {
                'word_count': 0,
                'avg_word_length': 0,
                'sentiment_compound': 0,
                'sentiment_pos': 0,
                'sentiment_neg': 0,
                'sentiment_neu': 0,
                'question_count': 0,
                'exclamation_count': 0,
                'capitalized_ratio': 0,
                'emotion_happy_ratio': 0,
                'emotion_sad_ratio': 0,
                'emotion_angry_ratio': 0,
                'emotion_anxious_ratio': 0
            }
        
        # Basic text preprocessing
        text = text.lower()
        words = word_tokenize(text)
        words_no_stop = [word for word in words if word not in self.stop_words and word.isalpha()]
        
        # Basic counts
        word_count = len(words)
        word_lengths = [len(word) for word in words_no_stop] if words_no_stop else [0]
        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
        
        # Sentiment analysis
        sentiment = self.sia.polarity_scores(text)
        
        # Punctuation-based features
        question_count = text.count('?')
        exclamation_count = text.count('!')
        
        # Capitalization
        capitalized_count = sum(1 for char in text if char.isupper())
        capitalized_ratio = capitalized_count / len(text) if text else 0
        
        # Emotion lexicon-based features
        emotion_counts = {}
        for emotion, lexicon in self.emotion_lexicons.items():
            count = sum(1 for word in words_no_stop if word in lexicon)
            emotion_counts[f'emotion_{emotion}_ratio'] = count / word_count if word_count > 0 else 0
        
        # Combine all features
        features = {
            'word_count': word_count,
            'avg_word_length': avg_word_length,
            'sentiment_compound': sentiment['compound'],
            'sentiment_pos': sentiment['pos'],
            'sentiment_neg': sentiment['neg'],
            'sentiment_neu': sentiment['neu'],
            'question_count': question_count,
            'exclamation_count': exclamation_count,
            'capitalized_ratio': capitalized_ratio,
        }
        
        # Add emotion lexicon features
        features.update(emotion_counts)
        
        return features
    
    def extract_features_for_model(self, text):
        """Extract and normalize features for model input."""
        features = self.extract_features(text)
        
        # Convert to vector
        feature_vector = np.array(list(features.values()))
        
        return feature_vector, features 