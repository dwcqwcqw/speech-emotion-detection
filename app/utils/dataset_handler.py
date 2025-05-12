import os
import requests
import zipfile
import tempfile
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

class DatasetHandler:
    """
    Class for downloading and managing the RAVDESS emotional speech dataset.
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.audio_dir = os.path.join(data_dir, 'audio')
        self.metadata_path = os.path.join(data_dir, 'metadata.csv')
        
        # RAVDESS dataset
        self.ravdess_url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        
        # Emotion mapping
        self.emotion_map = {
            '01': 'neutral',
            '02': 'neutral',  # "calm" mapped to neutral
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'anxious',  # "fearful" mapped to anxious
            '07': 'neutral',  # "disgust" mapped to neutral
            '08': 'neutral'   # "surprised" mapped to neutral
        }
    
    def download_ravdess(self):
        """
        Download and extract the RAVDESS dataset.
        """
        if os.path.exists(self.metadata_path) and len(os.listdir(self.audio_dir)) > 0:
            print("Dataset already downloaded and extracted.")
            return
        
        print("Downloading RAVDESS dataset...")
        
        # Download the dataset
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            response = requests.get(self.ravdess_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            
            with tqdm(total=total_size, unit='B', unit_scale=True) as progress_bar:
                for data in response.iter_content(block_size):
                    temp_file.write(data)
                    progress_bar.update(len(data))
            
            temp_file_path = temp_file.name
        
        print("Extracting files...")
        
        # Extract the dataset
        with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        # Move audio files to audio directory
        extracted_path = os.path.join(self.data_dir, 'Audio_Speech_Actors_01-24')
        if os.path.exists(extracted_path):
            for actor_dir in os.listdir(extracted_path):
                actor_path = os.path.join(extracted_path, actor_dir)
                if os.path.isdir(actor_path):
                    for audio_file in os.listdir(actor_path):
                        if audio_file.endswith('.wav'):
                            shutil.move(
                                os.path.join(actor_path, audio_file),
                                os.path.join(self.audio_dir, audio_file)
                            )
            
            # Clean up extraction directory
            shutil.rmtree(extracted_path)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Create metadata
        self._create_metadata()
        
        print("Dataset downloaded and extracted successfully.")
    
    def _create_metadata(self):
        """
        Create metadata CSV file for the dataset.
        Format: 'modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav'
        """
        metadata = []
        
        for filename in os.listdir(self.audio_dir):
            if filename.endswith('.wav'):
                parts = filename.split('-')
                if len(parts) >= 7:
                    emotion_code = parts[2]
                    intensity = parts[3]
                    actor = parts[6].split('.')[0]
                    gender = 'female' if int(actor) % 2 == 0 else 'male'
                    
                    if emotion_code in self.emotion_map:
                        emotion = self.emotion_map[emotion_code]
                        
                        metadata.append({
                            'filename': filename,
                            'path': os.path.join(self.audio_dir, filename),
                            'emotion': emotion,
                            'intensity': 'normal' if intensity == '01' else 'strong',
                            'actor': actor,
                            'gender': gender
                        })
        
        # Create metadata dataframe and save to CSV
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(self.metadata_path, index=False)
    
    def load_metadata(self):
        """
        Load the metadata dataframe.
        """
        if not os.path.exists(self.metadata_path):
            self._create_metadata()
        
        return pd.read_csv(self.metadata_path)
    
    def split_dataset(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split the dataset into training, validation, and test sets.
        
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        # Load metadata
        metadata_df = self.load_metadata()
        
        # Get unique actors for speaker-independent split
        actors = metadata_df['actor'].unique()
        random.Random(random_state).shuffle(actors)
        
        # Split actors into train, val, and test
        n_actors = len(actors)
        n_test = int(n_actors * test_size)
        n_val = int(n_actors * val_size)
        n_train = n_actors - n_test - n_val
        
        train_actors = actors[:n_train]
        val_actors = actors[n_train:n_train+n_val]
        test_actors = actors[n_train+n_val:]
        
        # Split dataframe
        train_df = metadata_df[metadata_df['actor'].isin(train_actors)]
        val_df = metadata_df[metadata_df['actor'].isin(val_actors)]
        test_df = metadata_df[metadata_df['actor'].isin(test_actors)]
        
        return train_df, val_df, test_df 