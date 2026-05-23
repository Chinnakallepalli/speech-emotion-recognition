import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import defaultdict
from utils import extract_feature_logic

# --- PAGE CONFIG ---
st.set_page_config(page_title="Emotion AI Dashboard", page_icon="🎙️", layout="wide")

# --- SIDEBAR FOR PROJECT INFO ---
with st.sidebar:
    st.title("Project Details")
    st.info("""
    **Model:** SVM (RBF Kernel)  
    **Accuracy:** 93%  
    **Features:** MFCC, Chroma, Mel, RMS, ZCR  
    **Datasets:** RAVDESS & TESS
    """)
    st.markdown("---")
    st.write("Developed for Placement Portfolio")

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    m = pickle.load(open('model.pkl', 'rb'))
    s = pickle.load(open('scaler.pkl', 'rb'))
    return m, s

model, scaler = load_assets()

# --- CHECK IF MODEL SUPPORTS PROBABILITY ---
HAS_PROBA = hasattr(model, "predict_proba")
CONFIDENCE_THRESHOLD = 0.45

# --- MAIN INTERFACE ---
st.title("🎙️ Speech Emotion Recognition System")
st.markdown("Upload an audio clip to analyze the underlying emotional state using Machine Learning.")

uploaded_file = st.file_uploader("📤 Drag and drop a WAV file", type=["wav"])

if uploaded_file:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Audio Analysis")
        st.audio(uploaded_file)

        # --- WAVEFORM VISUALIZATION ---
        X, sr = librosa.load(uploaded_file, sr=22050)
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(X, sr=sr, ax=ax, color="#1f77b4")
        ax.set_title("Audio Waveform")
        ax.set_axis_off()
        st.pyplot(fig)

    with col2:
        st.subheader("Prediction Results")
        if st.button("🚀 Run Recognition", use_container_width=True):
            with st.spinner("Analyzing segments..."):
                duration = librosa.get_duration(y=X, sr=sr)
                window_size, step_size = 3.0, 1.5
                windows = np.arange(0, max(0.1, duration - window_size), step_size)

                emotion_scores = defaultdict(float)
                preds = []

                for start in windows:
                    chunk = X[int(start * sr): int((start + window_size) * sr)]
                    feat = extract_feature_logic(chunk, sr)
                    feat_scaled = scaler.transform(feat.reshape(1, -1))

                    if HAS_PROBA:
                        # Confidence-weighted voting
                        proba = model.predict_proba(feat_scaled)[0]
                        confidence = np.max(proba)
                        predicted = model.classes_[np.argmax(proba)]
                        if confidence >= CONFIDENCE_THRESHOLD:
                            emotion_scores[predicted] += confidence
                            preds.append(predicted)
                    else:
                        # Fallback: simple majority vote
                        predicted = model.predict(feat_scaled)[0]
                        emotion_scores[predicted] += 1
                        preds.append(predicted)

                if emotion_scores:
                    final_emotion = max(emotion_scores, key=emotion_scores.get)
                    total_score = sum(emotion_scores.values())
                    conf_pct = (emotion_scores[final_emotion] / total_score) * 100

                    # --- DISPLAY ---
                    st.metric(label="Detected Emotion", value=final_emotion.upper())
                    st.progress(int(conf_pct))
                    st.caption(f"Confidence: {conf_pct:.1f}%")

                    # --- EMOTION FEEDBACK ---
                    if final_emotion == "happy":
                        st.write("😊 The speaker sounds positive and energetic.")
                    elif final_emotion == "sad":
                        st.write("😔 The speaker sounds low-energy or sorrowful.")
                    elif final_emotion == "angry":
                        st.write("😠 High vocal intensity and sharpness detected.")
                    elif final_emotion == "fearful":
                        st.write("😨 Signs of vocal trembling or anxiety detected.")
                    

                    st.balloons()

                elif preds == [] and HAS_PROBA:
                    # All windows were below confidence threshold
                    st.warning("⚠️ No confident prediction found. The audio may be too noisy or unclear.")
                else:
                    st.error("Audio too short.")

# --- FOOTER ---
st.markdown("---")
st.caption("Machine Learning Model trained on 182-dimensional acoustic feature vectors.")
