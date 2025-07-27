# app.py
import streamlit as st # type: ignore
import joblib # type: ignore
import librosa # type: ignore
import numpy as np # type: ignore

st.title("🎧 Stress Detection from Audio")
st.write("Upload a .wav audio file to classify stress level.")

# Load trained model
model = joblib.load("stress_model_audio.pkl")

# Feature extraction
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# File upload
uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file:
    st.audio(uploaded_file)
    features = extract_features(uploaded_file)
    prediction = model.predict([features])[0]
    label = "😌 Relaxed" if prediction == 0 else "😣 Stressed"
    st.success(f"Prediction: **{label}**")
