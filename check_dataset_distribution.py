#!/usr/bin/env python3
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the dataset handler
from app.utils.dataset_handler import DatasetHandler

def check_distribution():
    """Check and visualize the distribution of emotions in the dataset."""
    print("Checking emotion distribution in the dataset...")
    
    # Create dataset handler
    dataset_handler = DatasetHandler()
    
    # Load metadata
    metadata_df = dataset_handler.load_metadata()
    
    if len(metadata_df) == 0:
        print("Error: No data found in metadata.")
        return
    
    # Count emotions
    emotion_counts = metadata_df['emotion'].value_counts()
    total_samples = len(metadata_df)
    
    print(f"Total samples: {total_samples}")
    print("\nEmotion distribution:")
    for emotion, count in emotion_counts.items():
        percentage = (count / total_samples) * 100
        print(f"{emotion}: {count} samples ({percentage:.2f}%)")
    
    # Check distribution by gender
    print("\nEmotion distribution by gender:")
    gender_emotion_counts = metadata_df.groupby(['gender', 'emotion']).size().unstack(fill_value=0)
    print(gender_emotion_counts)
    
    # Check distribution by intensity
    print("\nEmotion distribution by intensity:")
    intensity_emotion_counts = metadata_df.groupby(['intensity', 'emotion']).size().unstack(fill_value=0)
    print(intensity_emotion_counts)
    
    # Visualize the distribution
    plt.figure(figsize=(12, 10))
    
    # Emotion distribution
    plt.subplot(2, 2, 1)
    emotion_counts.plot(kind='bar', color='skyblue')
    plt.title('Emotion Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Gender distribution
    plt.subplot(2, 2, 2)
    gender_emotion_counts.plot(kind='bar')
    plt.title('Emotion Distribution by Gender')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Intensity distribution
    plt.subplot(2, 2, 3)
    intensity_emotion_counts.plot(kind='bar')
    plt.title('Emotion Distribution by Intensity')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Emotion percentage
    plt.subplot(2, 2, 4)
    emotion_percentages = emotion_counts / total_samples * 100
    plt.pie(emotion_percentages, labels=emotion_percentages.index, autopct='%1.1f%%')
    plt.title('Emotion Percentage')
    
    plt.tight_layout()
    plt.savefig('emotion_distribution.png')
    print("\nVisualization saved to 'emotion_distribution.png'")

if __name__ == "__main__":
    check_distribution() 