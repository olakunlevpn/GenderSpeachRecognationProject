"""Interactive Streamlit Web Application for Voice Gender & Age Prediction.

Supports:
- Live microphone recording (st.audio_input) or audio file upload (.wav, .mp3, .ogg, .flac)
- Real-time acoustic feature extraction (MFCCs, Pitch F0 contour, Formants F1-F4)
- Model selector: SVM vs Random Forest vs BiLSTM Deep Learning
- Interactive probability distributions and vocal biomarker charts
"""

import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display

from src.audio.features import extract_features_dict, extract_pitch_f0, extract_formants_lpc
from src.audio.preprocessor import load_and_preprocess_audio
from src.models.classical import ClassicalSpeechPipeline
from src.models.rnn import DeepSpeechPipeline

# Page configuration
st.set_page_config(
    page_title="Voice Gender & Age Analyzer",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Voice Gender & Age Predictor")
st.markdown(
    "Analyze acoustic speech signals (pitch $F_0$, formants $F_1\\text{--}F_4$, MFCCs, prosody) "
    "using **Classical ML (SVM, Random Forest)** and **Deep Learning (BiLSTM + Attention)**."
)

# Sidebar: Model and Configuration
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model Architecture",
    ["SVM (Classical ML)", "Random Forest (Classical ML)", "BiLSTM + Attention (Deep Learning)"],
)

models_dir = Path("saved_models")


@st.cache_resource
def load_models():
    svm_p = None
    rf_p = None
    bilstm_p = None

    svm_path = models_dir / "svm_pipeline.joblib"
    rf_path = models_dir / "rf_pipeline.joblib"
    bilstm_path = models_dir / "bilstm_pipeline.pt"

    if svm_path.is_file():
        svm_p = ClassicalSpeechPipeline.load(svm_path)
    if rf_path.is_file():
        rf_p = ClassicalSpeechPipeline.load(rf_path)
    if bilstm_path.is_file():
        bilstm_p = DeepSpeechPipeline.load(bilstm_path, device="cpu")

    return svm_p, rf_p, bilstm_p


svm_pipeline, rf_pipeline, bilstm_pipeline = load_models()

# Audio Input Section
st.subheader("1. Provide Speech Audio")
input_tab1, input_tab2 = st.tabs(["🎤 Record Microphone", "📁 Upload Audio File"])

audio_bytes = None
source_name = None

with input_tab1:
    recorded_audio = st.audio_input("Record a 2-5 second sentence (e.g., 'The quick brown fox jumps over the lazy dog')")
    if recorded_audio is not None:
        audio_bytes = recorded_audio.read()
        source_name = "Microphone Recording"

with input_tab2:
    uploaded_file = st.file_uploader(
        "Upload audio clip (.wav, .mp3, .ogg, .flac)",
        type=["wav", "mp3", "ogg", "flac"],
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        source_name = uploaded_file.name

if audio_bytes is not None:
    st.audio(audio_bytes, format="audio/wav")

    with st.spinner("Processing audio and extracting acoustic features..."):
        try:
            y, sr = load_and_preprocess_audio(audio_bytes, target_sr=16000)
            duration_sec = len(y) / sr
        except Exception as e:
            st.error(f"Error reading audio file: {e}")
            st.stop()

    col_meta1, col_meta2, col_meta3 = st.columns(3)
    col_meta1.metric("Sample Rate", f"{sr} Hz")
    col_meta2.metric("Duration", f"{duration_sec:.2f} seconds")
    col_meta3.metric("Samples", f"{len(y):,}")

    # Feature Extraction
    features = extract_features_dict(y, sr=sr)

    # Predictions
    st.subheader("2. Model Predictions")

    selected_pipeline = None
    if "SVM" in model_choice:
        selected_pipeline = svm_pipeline
    elif "Random Forest" in model_choice:
        selected_pipeline = rf_pipeline
    else:
        selected_pipeline = bilstm_pipeline

    if selected_pipeline is None:
        st.warning(
            f"Model '{model_choice}' not yet found in `saved_models/`. "
            "Please run the training script first: `python -m src.training.train_all --synthetic`"
        )
    else:
        pred = selected_pipeline.predict_audio(y, sr=sr)

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.markdown("### Predicted Gender")
            gender_label = pred["gender"].capitalize()
            conf = pred.get("gender_confidence", 0.0) * 100
            st.info(f"**{gender_label}** (Confidence: {conf:.1f}%)")

            if "gender_probabilities" in pred:
                df_g = pd.DataFrame([
                    {"Gender": k.capitalize(), "Probability": v}
                    for k, v in pred["gender_probabilities"].items()
                ])
                st.bar_chart(df_g.set_index("Gender"))

        with col_res2:
            st.markdown("### Predicted Age Range")
            age_label = pred["age_range"].replace("_", " ").title()
            conf_a = pred.get("age_confidence", 0.0) * 100
            st.success(f"**{age_label}** (Confidence: {conf_a:.1f}%)")

            if "age_probabilities" in pred:
                df_a = pd.DataFrame([
                    {"Age Range": k.replace("_", " ").title(), "Probability": v}
                    for k, v in pred["age_probabilities"].items()
                ])
                st.bar_chart(df_a.set_index("Age Range"))

    # Acoustic Biomarkers & Visualizations
    st.subheader("3. Acoustic Biomarkers & Spectrogram")

    col_bio1, col_bio2, col_bio3, col_bio4 = st.columns(4)
    f0_mean = features.get("f0_mean", 0.0)
    col_bio1.metric("Mean Pitch ($F_0$)", f"{f0_mean:.1f} Hz" if f0_mean > 0 else "Unvoiced")
    f1 = features.get("formant_f1_mean", 0.0)
    col_bio2.metric("Formant $F_1$", f"{f1:.1f} Hz" if f1 > 0 else "N/A")
    f2 = features.get("formant_f2_mean", 0.0)
    col_bio3.metric("Formant $F_2$", f"{f2:.1f} Hz" if f2 > 0 else "N/A")
    f3 = features.get("formant_f3_mean", 0.0)
    col_bio4.metric("Formant $F_3$", f"{f3:.1f} Hz" if f3 > 0 else "N/A")

    # Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # 1. Waveform
    time_axis = np.linspace(0, duration_sec, len(y))
    ax1.plot(time_axis, y, color="#1f77b4", alpha=0.8, linewidth=0.8)
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Speech Waveform")
    ax1.grid(True, alpha=0.3)

    # 2. Spectrogram + Pitch Contour
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="linear", ax=ax2, cmap="magma")
    ax2.set_ylim(0, 4000)
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_title("Spectrogram (0-4 kHz) with Formants Region")
    fig.colorbar(img, ax=ax2, format="%+2.0f dB")

    st.pyplot(fig)

    # Expandable Details
    with st.expander("🔍 View All 98 Extracted Acoustic Features"):
        st.dataframe(pd.DataFrame(list(features.items()), columns=["Feature", "Value"]))

else:
    st.info("👋 Upload an audio file or record your voice above to start analysis!")
