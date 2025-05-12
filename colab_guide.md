# Google Colab Guide - Multimodal Emotion Detection System

This guide will help you set up and run the multimodal emotion detection system in Google Colab.

## 1. Clone the Repository and Set Up the Environment

Create a new notebook in Colab and execute the following code:

```python
# Clone GitHub repository
!git clone https://github.com/dwcqwcqw/speech-emotion-detection.git
%cd speech-emotion-detection

# Install required dependencies
!pip install -r requirements.txt

# Install additional required dependencies
!pip install streamlit pydub librosa torch transformers sklearn nltk scikit-learn matplotlib seaborn tqdm
!pip install imblearn

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

## 2. Fix Colab-Specific Path Issues

Since Colab's environment differs from a local environment, create a Colab-specific configuration file:

```python
%%writefile colab_config.py
import os
import sys

# Ensure the current directory is the project root
project_root = os.getcwd()
sys.path.append(project_root)

# Create necessary directories
os.makedirs('data/models', exist_ok=True)
os.makedirs('data/audio', exist_ok=True)
os.makedirs('data/evaluation', exist_ok=True)
os.makedirs('data/sarcasm', exist_ok=True)
```

## 3. Prepare Datasets

Download and prepare the RAVDESS dataset and sarcasm dataset:

```python
# Run the sarcasm dataset download script
!python download_sarcasm_dataset.py

# View the dataset directory structure
!ls -la data/
!ls -la data/sarcasm/
```

## 4. Train the Model

Use the following code to train the emotion detection model:

```python
# Run the training script
!python app/train_model.py
```

## 5. Evaluate the Model

After training, evaluate the model's performance:

```python
# Run the evaluation script
!python app/evaluate_model.py

# View evaluation results
!ls -la data/evaluation/
```

## 6. Run the Streamlit App in Colab (Optional)

If you want to run the Streamlit application for demonstration in Colab, you can use the following method:

```python
# Install and configure ngrok (for exposing local servers)
!pip install pyngrok
from pyngrok import ngrok

# Start Streamlit application
!streamlit run app/app.py &

# Create a public URL
public_url = ngrok.connect(port=8501)
print(f"Streamlit application can be accessed at: {public_url}")
```

## 7. Custom Inference Using MultimodalAnalyzer

You can use the following code snippet to test the system's inference capability in Colab:

```python
# Import required libraries
import sys
import os
sys.path.append(os.getcwd())

from app.utils.audio_processor import AudioProcessor
from app.utils.speech_to_text_simple import SimpleSpeechToText 
from app.utils.text_analyzer_extended import TextAnalyzerExtended
from app.utils.emotion_classifier_extended import EmotionClassifierExtended
from app.utils.multimodal_analyzer import MultimodalAnalyzer

# Initialize components
audio_processor = AudioProcessor()
speech_to_text = SimpleSpeechToText()
text_analyzer = TextAnalyzerExtended()

# Load trained model
model_path = 'data/models/emotion_classifier_extended.pkl'
emotion_classifier = EmotionClassifierExtended(model_path=model_path)

# Initialize multimodal analyzer
multimodal_analyzer = MultimodalAnalyzer(
    audio_processor=audio_processor,
    speech_to_text=speech_to_text,
    text_analyzer=text_analyzer,
    emotion_classifier=emotion_classifier,
    weights_path='data/models/multimodal_weights.json' if os.path.exists('data/models/multimodal_weights.json') else None
)

# Select an audio file for analysis
from app.utils.dataset_handler import DatasetHandler
dataset_handler = DatasetHandler()
_, _, test_df = dataset_handler.split_dataset()

# Analyze the first test sample
if len(test_df) > 0:
    sample = test_df.iloc[0]
    print(f"Analyzing sample: {sample['filename']} (True emotion: {sample['emotion']})")
    
    result = multimodal_analyzer.analyze(sample['path'])
    
    print(f"Predicted emotion: {result['emotion']}")
    print(f"Transcription: {result['transcription']}")
    print(f"Modality agreement score: {result['agreement_score']:.2f}")
    print(f"Modality weights: Audio={result['modality_weights']['audio']:.2f}, Text={result['modality_weights']['text']:.2f}")
    
    # Print confidence scores
    print("\nEmotion confidence scores:")
    for emotion, score in sorted(result['confidence_scores'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {score:.4f}")
```

## Troubleshooting

If you encounter the following issues, you can try the corresponding solutions:

### Dataset Download Issues

If there are problems downloading or extracting the RAVDESS dataset, you can manually download and upload it to Colab:

```python
# Manually download RAVDESS dataset
!wget -O data/ravdess.zip https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip

# Extract the dataset
!unzip -o data/ravdess.zip -d data/
!mkdir -p data/audio

# Move all WAV files to the data/audio directory
!find data/Audio_Speech_Actors_01-24 -name "*.wav" -exec cp {} data/audio/ \;
```

### Path Errors

If you encounter path-related errors, ensure that the current working directory is the project root:

```python
import os
print(f"Current working directory: {os.getcwd()}")
# If it's not the project root, use the following command to switch
%cd speech-emotion-detection  # Replace with the actual path
```

### Memory Errors

If you encounter out-of-memory errors, you can try the following methods:

1. Switch to a runtime with more RAM in Colab (Runtime > Change runtime type > Select High RAM)
2. Reduce the batch size or use a smaller data subset for training:

```python
# Use a smaller training set for testing
!python app/train_model.py --sample-size 100  # If the script supports this parameter
```

### Dependency Package Conflicts

If you encounter dependency package conflicts, you can try installing in an isolated environment:

```python
!pip install -r requirements.txt --no-deps
```

Then manually install key dependencies:

```python
!pip install torch==1.10.0 transformers==4.11.3 librosa==0.8.1
``` 