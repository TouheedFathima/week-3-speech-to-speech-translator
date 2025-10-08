# Speech-to-Speech Translator

## Description
This project enables users to record or upload audio, automatically detect whether the speech is in Hindi or English, transcribe it, translate into a selected language, and generate audio output.

## Features
- Record audio (up to 2 minutes)
- Upload audio or video files (.wav, .mp3, .mp4, .mov)
- Automatic Hindi/English detection (backend only)
- Transcription of speech
- Translation to 12 Indian languages
- Audio output in the selected language

## How to Run
1. Clone the repository  
2. Create and activate a virtual environment  
3. Install dependencies: `pip install -r requirements.txt`  
4. Run: `streamlit run app.py`  

## Requirements
- Python >= 3.8  
- Streamlit, Sounddevice, Librosa, Torch, Transformers, Googletrans, gTTS, MoviePy, Joblib

