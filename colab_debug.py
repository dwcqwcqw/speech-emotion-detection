#!/usr/bin/env python3
import os
import sys
import zipfile
import tempfile
import requests
import shutil
from tqdm import tqdm

def debug_extraction():
    print("============ COLAB EXTRACTION DEBUGGING ============")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check various paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(f"Project root: {project_root}")
    print(f"Project root contents: {os.listdir(project_root)}")
    
    # Check data directory
    data_dir = os.path.join(project_root, 'data')
    print(f"Data directory: {data_dir}")
    print(f"Data directory exists: {os.path.exists(data_dir)}")
    if os.path.exists(data_dir):
        print(f"Data directory contents: {os.listdir(data_dir)}")
    
    # Search for Audio_Speech_Actors_01-24 anywhere in the filesystem from /content
    search_dir = "/content"
    print(f"Searching for extracted directories in {search_dir}...")
    
    for root, dirs, files in os.walk(search_dir, topdown=True):
        for dirname in dirs:
            if "Actor" in dirname:
                print(f"Found actor directory: {os.path.join(root, dirname)}")
                actor_files = os.listdir(os.path.join(root, dirname))
                wav_files = [f for f in actor_files if f.endswith('.wav')]
                print(f"  Contains {len(wav_files)} WAV files")
                if wav_files:
                    print(f"  Sample files: {wav_files[:3]}")
    
    # Try direct extraction
    print("\nAttempting direct extraction...")
    
    # RAVDESS dataset URL
    ravdess_url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    
    # Set up directories
    data_dir = os.path.join(project_root, 'data')
    audio_dir = os.path.join(data_dir, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    
    print(f"Created audio directory: {audio_dir}")
    
    # Download a small sample (abort after 10MB to save time)
    print("Downloading sample of RAVDESS dataset...")
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        response = requests.get(ravdess_url, stream=True)
        
        # We'll capture the entire file to ensure we get the zip header
        block_size = 1024 * 1024  # 1MB chunks
        total_size = int(response.headers.get('content-length', 0))
        
        # Download full file
        with tqdm(total=total_size, unit='B', unit_scale=True) as progress_bar:
            for data in response.iter_content(block_size):
                temp_file.write(data)
                progress_bar.update(len(data))
        
        temp_file_path = temp_file.name
    
    print(f"Downloaded to temporary file: {temp_file_path}")
    print(f"Temp file size: {os.path.getsize(temp_file_path)} bytes")
    
    # Try extracting directly to audio directory
    try:
        print(f"Extracting directly to audio directory: {audio_dir}")
        with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
            # List contents without extracting first
            entries = zip_ref.namelist()
            print(f"ZIP contains {len(entries)} entries")
            print(f"Sample entries: {entries[:5]}")
            
            # Extract 5 random audio files directly to audio directory
            audio_entries = [e for e in entries if e.endswith('.wav')][:5]
            for entry in audio_entries:
                try:
                    src = zip_ref.open(entry)
                    filename = os.path.basename(entry)
                    dst_path = os.path.join(audio_dir, filename)
                    
                    with open(dst_path, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    
                    print(f"Successfully extracted {filename} to {dst_path}")
                except Exception as e:
                    print(f"Error extracting {entry}: {str(e)}")
    except Exception as e:
        print(f"Error during direct extraction: {str(e)}")
    
    # Check audio directory after extraction
    if os.path.exists(audio_dir):
        files = os.listdir(audio_dir)
        print(f"Audio directory now contains {len(files)} files")
        if files:
            print(f"Sample files: {files[:5]}")
    
    # Clean up
    try:
        os.unlink(temp_file_path)
    except:
        pass
    
    print("============ END DEBUGGING ============")

if __name__ == "__main__":
    debug_extraction() 