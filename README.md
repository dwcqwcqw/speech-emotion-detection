# Multimodal Emotion Detection System

This project combines speech prosody analysis with automatic speech recognition (ASR) to detect human emotions. It analyzes both how something is said (tone, pitch, intensity) and what is said (text content) to accurately identify emotional states such as happiness, anger, sadness, and anxiety.

## Features

- Acoustic feature extraction from speech audio
- Text transcription using ASR
- Combined multimodal emotion classification
- User-friendly web interface for audio upload and analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage

```bash
# Run the web application
streamlit run app/app.py
```

## Dataset

This project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset, which is freely available for academic research.

## License

MIT 