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
        
        # Pitch (fundamental frequency) using harmonic component
        pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr)
        pitch_mean = np.mean([np.mean(pitches[magnitudes[i] > 0.05]) for i in range(len(magnitudes)) if np.any(magnitudes[i] > 0.05)] or [0])
        pitch_std = np.std([np.mean(pitches[magnitudes[i] > 0.05]) for i in range(len(magnitudes)) if np.any(magnitudes[i] > 0.05)] or [0])
        
        # Energy/Intensity
        rms = librosa.feature.rms(y=waveform)[0]
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        
        # Tempo and rhythm
        onset_env = librosa.onset.onset_strength(y=waveform, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(waveform)[0]
        
        # MFCC (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        
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
        waveform, sr = self.load_audio(file_path)
        features = self.extract_features(waveform, sr)
        
        # Convert to vector and normalize
        feature_vector = np.array(list(features.values()))
        normalized_features = zscore(feature_vector, nan_policy='omit')
        
        # Replace any NaNs with 0
        normalized_features = np.nan_to_num(normalized_features)
        
        return normalized_features, features 