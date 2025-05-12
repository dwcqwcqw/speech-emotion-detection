import os
import requests
import zipfile
import tempfile
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import sys

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
        
        # Debug information
        is_colab = 'google.colab' in sys.modules
        if is_colab:
            print(f"Running in Colab environment")
            print(f"Data directory: {self.data_dir} (exists: {os.path.exists(self.data_dir)})")
            print(f"Audio directory: {self.audio_dir} (exists: {os.path.exists(self.audio_dir)})")
    
    def download_ravdess(self):
        """
        Download and extract the RAVDESS dataset.
        """
        # Check if we need to download
        if os.path.exists(self.metadata_path) and os.path.getsize(self.metadata_path) > 0 and len(os.listdir(self.audio_dir)) > 0:
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
        print(f"Extracting to {self.data_dir}")
        
        # Extract the dataset
        try:
            with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
                print(f"Extraction completed")
        except Exception as e:
            print(f"Error during extraction: {str(e)}")
            os.unlink(temp_file_path)
            return
        
        # Move audio files to audio directory
        extracted_path = os.path.join(self.data_dir, 'Audio_Speech_Actors_01-24')
        print(f"Looking for extracted files in: {extracted_path}")
        print(f"Extracted path exists: {os.path.exists(extracted_path)}")
        
        if os.path.exists(extracted_path):
            actor_dirs = os.listdir(extracted_path)
            print(f"Found {len(actor_dirs)} actor directories")
            
            audio_files_count = 0
            for actor_dir in actor_dirs:
                actor_path = os.path.join(extracted_path, actor_dir)
                if os.path.isdir(actor_path):
                    files = os.listdir(actor_path)
                    audio_files = [f for f in files if f.endswith('.wav')]
                    audio_files_count += len(audio_files)
                    
                    print(f"Actor {actor_dir}: {len(audio_files)} audio files")
                    
                    for audio_file in audio_files:
                        # Copy instead of move to prevent issues
                        src = os.path.join(actor_path, audio_file)
                        dst = os.path.join(self.audio_dir, audio_file)
                        try:
                            shutil.copy2(src, dst)
                        except Exception as e:
                            print(f"Error copying {src} to {dst}: {str(e)}")
            
            print(f"Total of {audio_files_count} audio files found in extracted directories")
            
            # Verify files were copied
            copied_files = os.listdir(self.audio_dir)
            print(f"Audio directory now contains {len(copied_files)} files")
            
            try:
                # Clean up extraction directory
                shutil.rmtree(extracted_path)
                print("Cleaned up extraction directory")
            except Exception as e:
                print(f"Warning: Could not clean up extraction directory: {str(e)}")
        else:
            print(f"ERROR: Extracted directory not found after extraction")
        
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary file: {str(e)}")
        
        # Create metadata
        self._create_metadata()
        
        print("Dataset downloaded and extracted successfully.")
    
    def _create_metadata(self):
        """
        Create metadata CSV file for the dataset.
        Format: 'modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav'
        """
        metadata = []
        
        print(f"Creating metadata from audio files in {self.audio_dir}")
        
        if not os.path.exists(self.audio_dir):
            print(f"ERROR: Audio directory {self.audio_dir} does not exist")
            return
        
        files = os.listdir(self.audio_dir)
        print(f"Found {len(files)} files in audio directory")
        
        wav_files = [f for f in files if f.endswith('.wav')]
        print(f"Found {len(wav_files)} WAV files")
        
        for filename in wav_files:
            try:
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
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
        
        # Create metadata dataframe and save to CSV
        if metadata:  # Only save if we have data
            metadata_df = pd.DataFrame(metadata)
            metadata_df.to_csv(self.metadata_path, index=False)
            print(f"Created metadata file with {len(metadata)} entries")
        else:
            print("Warning: No audio files found to create metadata")
    
    def load_metadata(self):
        """
        Load the metadata dataframe.
        """
        if not os.path.exists(self.metadata_path) or os.path.getsize(self.metadata_path) == 0:
            print("Metadata file doesn't exist or is empty, creating it now...")
            self._create_metadata()
            
            # If still empty, we have a problem with the dataset
            if not os.path.exists(self.metadata_path) or os.path.getsize(self.metadata_path) == 0:
                # Create a sample dataframe with expected columns but no data
                # This prevents the EmptyDataError but will still show the user that no data was found
                print("Warning: Could not create valid metadata. Creating empty dataframe with expected columns.")
                empty_df = pd.DataFrame(columns=[
                    'filename', 'path', 'emotion', 'intensity', 'actor', 'gender'
                ])
                empty_df.to_csv(self.metadata_path, index=False)
        
        # Print debug info
        print(f"Loading metadata from {self.metadata_path}")
        print(f"Metadata file exists: {os.path.exists(self.metadata_path)}")
        print(f"Metadata file size: {os.path.getsize(self.metadata_path) if os.path.exists(self.metadata_path) else 0} bytes")
        
        try:
            df = pd.read_csv(self.metadata_path)
            
            # Update paths if needed (for compatibility between environments)
            if 'path' in df.columns and len(df) > 0:
                # Ensure paths point to the correct location
                df['path'] = df['filename'].apply(lambda x: os.path.join(self.audio_dir, x))
                print(f"Updated paths in metadata to use audio directory: {self.audio_dir}")
            
            return df
        except pd.errors.EmptyDataError:
            print("Error: Metadata file exists but is empty or improperly formatted")
            # Return an empty DataFrame with the expected columns
            return pd.DataFrame(columns=[
                'filename', 'path', 'emotion', 'intensity', 'actor', 'gender'
            ])
    
    def split_dataset(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split the dataset into training, validation, and test sets.
        
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        # Load metadata
        metadata_df = self.load_metadata()
        
        # Check if we have data
        if len(metadata_df) == 0:
            print("Warning: No data found in metadata. Returning empty dataframes.")
            empty_df = pd.DataFrame(columns=[
                'filename', 'path', 'emotion', 'intensity', 'actor', 'gender'
            ])
            return empty_df, empty_df, empty_df
        
        # Get unique actors for speaker-independent split
        actors = metadata_df['actor'].unique()
        random.Random(random_state).shuffle(actors)
        
        # Split actors into train, val, and test
        n_actors = len(actors)
        n_test = max(1, int(n_actors * test_size))
        n_val = max(1, int(n_actors * val_size))
        n_train = n_actors - n_test - n_val
        
        train_actors = actors[:n_train]
        val_actors = actors[n_train:n_train+n_val]
        test_actors = actors[n_train+n_val:]
        
        # Split dataframe
        train_df = metadata_df[metadata_df['actor'].isin(train_actors)]
        val_df = metadata_df[metadata_df['actor'].isin(val_actors)]
        test_df = metadata_df[metadata_df['actor'].isin(test_actors)]
        
        return train_df, val_df, test_df 