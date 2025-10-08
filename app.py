import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import torch, joblib, os, time
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
from moviepy import VideoFileClip
from tempfile import NamedTemporaryFile
from googletrans import Translator
from gtts import gTTS

# -----------------------------
# Setup & Config
# -----------------------------
st.set_page_config(page_title="Speech-to-Speech Translator", page_icon="🎙", layout="centered")

os.environ['HF_HOME'] = "E:/huggingface_cache"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
translator = Translator()

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_hindi_asr_model():
    processor = Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec-hindi")
    model = Wav2Vec2ForCTC.from_pretrained("ai4bharat/indicwav2vec-hindi").to(device)
    return processor, model

@st.cache_resource
def load_english_asr_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

@st.cache_resource
def load_language_classifier():
    return joblib.load("hindi_speech_classifier.pkl")

# Lazy-load models only when needed
hindi_processor, hindi_model = None, None
english_asr = None
clf = load_language_classifier()

# -----------------------------
# Utility Functions
# -----------------------------
def extract_embedding(audio, sr=16000, chunk_size=10):
    """Use IndicWav2Vec embedding for language detection."""
    global hindi_processor, hindi_model
    if hindi_processor is None or hindi_model is None:
        hindi_processor, hindi_model = load_hindi_asr_model()

    embeddings = []
    step = sr * chunk_size
    for i in range(0, len(audio), step):
        chunk = audio[i:i+step]
        if len(chunk) == 0:
            continue
        inputs = hindi_processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = hindi_model.wav2vec2(**inputs).last_hidden_state
            emb = outputs.mean(dim=1).cpu().numpy().squeeze()
            embeddings.append(emb)
    return np.mean(embeddings, axis=0)

def predict_audio_language(audio):
    """Detect whether audio is Hindi or English."""
    emb = extract_embedding(audio)
    probs = clf.predict_proba([emb])[0]
    pred_idx = np.argmax(probs)
    return "Hindi" if pred_idx == 1 else "English"

def transcribe_audio(audio, lang):
    """Transcribe using appropriate ASR model."""
    global hindi_processor, hindi_model, english_asr
    if lang == "Hindi":
        if hindi_processor is None or hindi_model is None:
            hindi_processor, hindi_model = load_hindi_asr_model()
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = hindi_model(audio_tensor).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        text = hindi_processor.batch_decode(predicted_ids)[0]
        return text
    else:
        if english_asr is None:
            english_asr = load_english_asr_model()
        result = english_asr(audio)
        return result["text"]

def translate_and_speak(text, src_lang_code, dest_lang_code):
    """Translate text and play audio in the target language."""
    translated = translator.translate(text, src=src_lang_code, dest=dest_lang_code)
    tts = gTTS(text=translated.text, lang=dest_lang_code)
    with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        st.audio(tmp_file.name, format="audio/mp3")
    return translated.text

# -----------------------------
# UI Header
# -----------------------------
st.markdown("""
    <style>
    .main-title {text-align: center; font-size: 35px; font-weight: 700; margin-bottom: 10px;}
    .subtitle {text-align: center; font-size: 18px; margin-bottom: 25px;}
    .result-box {background-color: #f0f2f6; padding: 15px; border-radius: 10px; font-size: 18px; color: #222;}
    </style>
    <div class="main-title">🎙 Speech-to-Speech Translator</div>
    <div class="subtitle">Speak, translate, and listen in your preferred language</div>
""", unsafe_allow_html=True)

# -----------------------------
# Step 1: Record or Upload
# -----------------------------
with st.container():
    st.subheader("Step 1: Record or Upload Your Audio")
    col1, col2 = st.columns(2)

    with col1:
        duration = st.slider("Recording Duration (seconds):", min_value=3, max_value=120, value=10, step=5)
        if st.button("🎧 Record Audio"):
            st.info("Recording... please speak clearly 🎤")
            audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='float32')
            sd.wait()
            st.success("Recording complete ✅")
            st.session_state["audio"] = audio.flatten()

    with col2:
        uploaded_file = st.file_uploader("Or Upload Audio/Video File", type=["wav", "mp3", "mp4", "mov"])
        if uploaded_file is not None:
            with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            if uploaded_file.name.endswith((".mp4", ".mov")):
                st.info("Extracting audio from video...")
                video_clip = VideoFileClip(tmp_path)
                audio_path = tmp_path + "_audio.wav"
                video_clip.audio.write_audiofile(audio_path, fps=16000, nbytes=2, codec='pcm_s16le')
                y, _ = librosa.load(audio_path, sr=16000)
            else:
                y, _ = librosa.load(tmp_path, sr=16000)
            st.success("File uploaded successfully ✅")
            st.session_state["audio"] = y

# -----------------------------
# Step 2 : Detect Language (Backend only)
# -----------------------------
if "audio" in st.session_state:
    detected_lang = predict_audio_language(st.session_state["audio"])
    st.session_state["detected_lang"] = detected_lang

# -----------------------------
# Step 3: Transcription
# -----------------------------
if "detected_lang" in st.session_state:
    st.subheader("Step 2: Converting Speech to Text")
    with st.spinner("Transcribing..."):
        transcribed_text = transcribe_audio(st.session_state["audio"], st.session_state["detected_lang"])
        time.sleep(1)
    st.markdown("#### 🗒 Transcribed Text:")
    st.markdown(f"<div class='result-box'>{transcribed_text}</div>", unsafe_allow_html=True)
    st.session_state["text"] = transcribed_text

# -----------------------------
# Step 4: Translate & Speak
# -----------------------------
if "text" in st.session_state:
    st.subheader("Step 3: Translate Transcribed Speech")
    lang_options = {
        "Tamil": "ta", "Telugu": "te", "Kannada": "kn", "Bengali": "bn",
        "Gujarati": "gu", "Odia": "or", "Malayalam": "ml", "Punjabi": "pa",
        "Assamese": "as", "Marathi": "mr", "Urdu": "ur"
    }

    target_lang = st.selectbox("Choose target language:", list(lang_options.keys()))
    src_lang_code = 'hi' if st.session_state["detected_lang"] == "Hindi" else 'en'
    dest_lang_code = lang_options[target_lang]

    if st.button("🌐 Translate to voice output"):
        with st.spinner("Translating and generating speech..."):
            translated_text = translate_and_speak(st.session_state["text"], src_lang_code, dest_lang_code)
        st.markdown("#### 🎯 Translated Output:")
        st.markdown(f"<div class='result-box'>{translated_text}</div>", unsafe_allow_html=True)
