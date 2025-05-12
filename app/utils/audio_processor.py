import librosa
import numpy as np
from scipy.stats import zscore

class AudioProcessor:
    """
    A class for extracting acoustic features from audio files for emotion detection.
    """
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path):
        """Load an audio file and return the waveform and sample rate."""
        waveform, sr = librosa.load(file_path, sr=self.sample_rate)
        return waveform, sr
    
    def extract_features(self, waveform, sr):
        """Extract acoustic features relevant for emotion detection."""
        # Duration
        duration = librosa.get_duration(y=waveform, sr=sr)
        
        # Pitch (fundamental frequency) - Use a more robust approach
        try:
            # Calculate the pitch using a safer approach
            pitch_mean = 0
            pitch_std = 0
            
            # Method 1: Using piptrack with safer indexing
            try:
                pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr)
                
                # Instead of indexing directly, which can cause dimension mismatch,
                # use a different approach to get pitch information
                pitch_values = []
                for i in range(pitches.shape[1]):  # Iterate over time
                    indices = magnitudes[:, i] > 0.05
                    if np.any(indices):
                        pitch_values.extend(pitches[indices, i])
                
                if pitch_values:
                    pitch_mean = np.mean(pitch_values)
                    pitch_std = np.std(pitch_values)
                
            except Exception as e:
                print(f"Pitch extraction method 1 failed: {str(e)}")
                
                # Method 2: Try an alternative using harmonic component
                try:
                    f0, voiced_flag, voiced_probs = librosa.pyin(waveform, 
                                                               fmin=librosa.note_to_hz('C2'), 
                                                               fmax=librosa.note_to_hz('C7'),
                                                               sr=sr)
                    pitch_values = f0[voiced_flag]
                    if len(pitch_values) > 0:
                        pitch_mean = np.mean(pitch_values)
                        pitch_std = np.std(pitch_values)
                        
                except Exception as e2:
                    print(f"Pitch extraction method 2 failed: {str(e2)}")
                    # Set defaults if both methods fail
                    pitch_mean = 0
                    pitch_std = 0
                    
        except Exception as e:
            print(f"Complete pitch extraction failed: {str(e)}")
            pitch_mean = 0
            pitch_std = 0
        
        # Energy/Intensity
        rms = librosa.feature.rms(y=waveform)[0]
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        
        # Tempo and rhythm
        try:
            onset_env = librosa.onset.onset_strength(y=waveform, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        except Exception as e:
            print(f"Tempo extraction failed: {str(e)}")
            tempo = 120.0  # Default BPM
        
        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(waveform)[0]
        
        # MFCC (Mel-frequency cepstral coefficients)
        try:
            mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            mfcc_stds = np.std(mfccs, axis=1)
        except Exception as e:
            print(f"MFCC extraction failed: {str(e)}")
            # Set default values
            mfcc_means = np.zeros(13)
            mfcc_stds = np.zeros(13)
        
        # Combine all features
        features = {
            'duration': duration,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'tempo': tempo,
            'spectral_centroid_mean': np.mean(spec_cent),
            'spectral_bandwidth_mean': np.mean(spec_bw),
            'spectral_rolloff_mean': np.mean(rolloff),
            'zero_crossing_rate_mean': np.mean(zcr),
        }
        
        # Add MFCC features
        for i, (mean, std) in enumerate(zip(mfcc_means, mfcc_stds)):
            features[f'mfcc{i+1}_mean'] = mean
            features[f'mfcc{i+1}_std'] = std
        
        return features
    
    def extract_features_for_model(self, file_path):
        """Extract and normalize features for model input."""
        try:
            # Load audio
            waveform, sr = self.load_audio(file_path)
            
            # Extract features
            features = self.extract_features(waveform, sr)
            
            # Convert to vector and normalize
            feature_vector = np.array(list(features.values()))
            
            # Check for NaN values before normalization
            if np.isnan(feature_vector).any():
                print(f"Warning: NaN values detected in features for {file_path}")
                feature_vector = np.nan_to_num(feature_vector)
            
            # Use robust normalization that can handle potential issues
            try:
                normalized_features = zscore(feature_vector, nan_policy='omit')
                
                # Replace any remaining NaNs with 0
                normalized_features = np.nan_to_num(normalized_features)
            except Exception as e:
                print(f"Normalization failed: {str(e)}. Using unnormalized features.")
                normalized_features = feature_vector
            
            return normalized_features, features
            
        except Exception as e:
            print(f"Feature extraction failed for {file_path}: {str(e)}")
            # Return a default feature vector with the expected dimensions
            # Number of features should match the total features in the extract_features method
            default_features = np.zeros(13*2 + 10)  # 13 MFCCs (mean+std) + 10 other features
            empty_features_dict = {}
            return default_features, empty_features_dict 