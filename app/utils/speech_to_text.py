import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import numpy as np

class SpeechToText:
    """
    Class for performing Automatic Speech Recognition (ASR) using the Wav2Vec2 model.
    """
    
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
    
    def transcribe(self, file_path, sample_rate=16000):
        """
        Transcribe speech in an audio file to text.
        
        Args:
            file_path: Path to the audio file
            sample_rate: Target sample rate for the model
            
        Returns:
            str: Transcribed text
        """
        # Load audio
        waveform, sr = librosa.load(file_path, sr=sample_rate)
        
        # Convert to appropriate format for model
        inputs = self.processor(
            waveform, 
            sampling_rate=sample_rate, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        
        # Get predicted token IDs
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Convert token IDs to text
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription 