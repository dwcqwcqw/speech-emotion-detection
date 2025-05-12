#!/usr/bin/env python3
import os
import sys
import pandas as pd
import requests
import json
import csv
import tarfile
import zipfile
from tqdm import tqdm
import nltk

# Add parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def ensure_dir(directory):
    """Ensure directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_file(url, destination):
    """Download a file from URL to destination with progress bar."""
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return
    
    print(f"Downloading {url} to {destination}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def download_news_headlines():
    """Download News Headlines dataset for Sarcasm Detection."""
    # https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection
    dataset_dir = "data/sarcasm/news_headlines"
    ensure_dir(dataset_dir)
    
    sample_data = [
        {"headline": "you won't believe what this politician said", "is_sarcastic": 1},
        {"headline": "study finds regular exercise can improve health", "is_sarcastic": 0},
        {"headline": "shocked to discover water is wet", "is_sarcastic": 1},
        {"headline": "new research shows benefits of drinking water", "is_sarcastic": 0},
        {"headline": "area man reads article, becomes expert overnight", "is_sarcastic": 1},
        {"headline": "scientists discover new species in amazon rainforest", "is_sarcastic": 0},
        {"headline": "breaking: thing that always happens, happens again", "is_sarcastic": 1},
        {"headline": "company announces new product release for next month", "is_sarcastic": 0},
        {"headline": "local man loses pants, life remains unchanged", "is_sarcastic": 1},
        {"headline": "stock market shows signs of recovery after recent dip", "is_sarcastic": 0},
        {"headline": "nation celebrates as celebrity does completely normal thing", "is_sarcastic": 1},
        {"headline": "president signs new legislation to improve infrastructure", "is_sarcastic": 0},
        {"headline": "shocking new study finds that pizza tastes good", "is_sarcastic": 1},
        {"headline": "unemployment rate drops to lowest level in five years", "is_sarcastic": 0},
        {"headline": "weather forecast predicts weather will occur tomorrow", "is_sarcastic": 1}
    ]
    
    # Create sample JSON file
    with open(os.path.join(dataset_dir, "sarcasm_headlines.json"), 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    # Create CSV file for easier processing
    with open(os.path.join(dataset_dir, "sarcasm_headlines.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["headline", "is_sarcastic"])
        writer.writeheader()
        writer.writerows(sample_data)
    
    print(f"Created sample sarcasm news headlines dataset with {len(sample_data)} examples")
    return os.path.join(dataset_dir, "sarcasm_headlines.csv")

def download_reddit_dataset():
    """Download a sample of the Reddit sarcasm dataset."""
    dataset_dir = "data/sarcasm/reddit"
    ensure_dir(dataset_dir)
    
    # Create sample of realistic Reddit sarcasm data
    sample_data = [
        {"comment": "Oh yeah, I'm totally fine with getting up at 5am for this meeting.", "is_sarcastic": 1},
        {"comment": "I agree, we should schedule the next meeting for 9am when everyone is available.", "is_sarcastic": 0},
        {"comment": "Wow, what an absolutely brilliant idea that no one has ever thought of before.", "is_sarcastic": 1},
        {"comment": "That's a good suggestion, we should implement it in the next version.", "is_sarcastic": 0},
        {"comment": "Sure, because adding more bugs is exactly what this software needs.", "is_sarcastic": 1},
        {"comment": "The new update fixed several critical issues in the system.", "is_sarcastic": 0},
        {"comment": "I'm sure everyone will be thrilled to work this weekend.", "is_sarcastic": 1},
        {"comment": "The team is working hard to meet the deadline next week.", "is_sarcastic": 0},
        {"comment": "Right, because that's definitely how programming works.", "is_sarcastic": 1},
        {"comment": "This code implementation is very efficient and well-structured.", "is_sarcastic": 0},
        {"comment": "Oh great, another meeting that could have been an email.", "is_sarcastic": 1},
        {"comment": "The documentation explains all the key aspects of the system.", "is_sarcastic": 0},
        {"comment": "Yeah, I'm sure the client will love waiting another month.", "is_sarcastic": 1},
        {"comment": "We need to be patient with this process as quality takes time.", "is_sarcastic": 0},
        {"comment": "Clearly my favorite part of the day is fixing other people's code.", "is_sarcastic": 1}
    ]
    
    # Create CSV file
    with open(os.path.join(dataset_dir, "reddit_sarcasm.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["comment", "is_sarcastic"])
        writer.writeheader()
        writer.writerows(sample_data)
    
    print(f"Created sample Reddit sarcasm dataset with {len(sample_data)} examples")
    return os.path.join(dataset_dir, "reddit_sarcasm.csv")

def load_sarcasm_datasets():
    """Load and combine all sarcasm datasets."""
    # Download/create datasets if needed
    news_file = download_news_headlines()
    reddit_file = download_reddit_dataset()
    
    # Load datasets
    news_df = pd.read_csv(news_file)
    reddit_df = pd.read_csv(reddit_file)
    
    # Rename columns for consistency
    news_df = news_df.rename(columns={"headline": "text"})
    reddit_df = reddit_df.rename(columns={"comment": "text"})
    
    # Combine datasets
    combined_df = pd.concat([news_df, reddit_df], ignore_index=True)
    
    # Add source column
    combined_df["source"] = ["news"] * len(news_df) + ["reddit"] * len(reddit_df)
    
    # Save combined dataset
    combined_path = "data/sarcasm/combined_sarcasm.csv"
    ensure_dir(os.path.dirname(combined_path))
    combined_df.to_csv(combined_path, index=False)
    
    print(f"Combined sarcasm dataset created with {len(combined_df)} examples")
    print(f"  - Sarcastic: {combined_df['is_sarcastic'].sum()}")
    print(f"  - Non-sarcastic: {len(combined_df) - combined_df['is_sarcastic'].sum()}")
    
    return combined_path

def process_for_training():
    """Process sarcasm datasets for training."""
    # Download or install NLTK resources if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Load combined dataset
    combined_path = load_sarcasm_datasets()
    df = pd.read_csv(combined_path)
    
    # Basic text preprocessing
    df['text_tokenized'] = df['text'].apply(lambda x: ' '.join(nltk.word_tokenize(str(x).lower())))
    
    # Split into train and test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['is_sarcastic'], random_state=42)
    
    # Save processed datasets
    train_df.to_csv("data/sarcasm/train_sarcasm.csv", index=False)
    test_df.to_csv("data/sarcasm/test_sarcasm.csv", index=False)
    
    print(f"Processed sarcasm datasets saved:")
    print(f"  - Training set: {len(train_df)} examples")
    print(f"  - Test set: {len(test_df)} examples")

if __name__ == "__main__":
    process_for_training() 