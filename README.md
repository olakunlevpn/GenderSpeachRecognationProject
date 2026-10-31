# Gender and Age Prediction from Voice / Speech Signals

An end-to-end speech analysis and machine learning project that classifies **Gender** and predicts **Age Range** from voice recordings. The system compares classical machine learning models (**SVM**, **Random Forest**) against deep learning (**BiLSTM with Self-Attention**) and provides an interactive web application ready for free cloud hosting on **Railway**, **Hugging Face Spaces**, or **Render**.

---

## 🚀 Features

- **Rich Acoustic Feature Extraction**:
  - **MFCCs**: 20 coefficients + first derivative ($\Delta$) + second derivative ($\Delta\Delta$).
  - **Pitch ($F_0$) Tracking**: Probabilistic YIN (`librosa.pyin`) for fundamental frequency, jitter, and voiced/unvoiced ratio.
  - **Formant Analysis ($F_1, F_2, F_3, F_4$)**: Linear Predictive Coding (LPC) root polynomial estimation with vocal tract resonance filtering.
  - **Spectral & Energy Biomarkers**: Spectral centroid, bandwidth, rolloff, zero-crossing rate (ZCR), and RMS energy.
- **Model Comparison**:
  - **Classical ML**: RBF-kernel Support Vector Machine (SVM) and Random Forest with feature importance analysis.
  - **Deep Learning**: PyTorch 2-layer Bidirectional LSTM with temporal Self-Attention pooling and multi-task dual heads.
- **Leakage-Free Splitting**:
  - Group-based splitting on `client_id` ensures no speaker appears in both train and evaluation sets.
- **Interactive Streamlit Web Dashboard**:
  - In-browser microphone recording (`st.audio_input`) or file upload (`.wav`, `.mp3`, `.ogg`, `.flac`).
  - Real-time pitch, formant, and waveform visualizations.
  - Probability bar charts for gender and age categories.
- **Free Cloud Deployment**:
  - CPU-only, low-memory Docker container (< 450 MB) designed for free-tier hosting on Railway, Render, and Hugging Face Spaces.

---

## 📁 Repository Structure

```text
├── app/
│   └── streamlit_app.py        # Interactive web application
├── configs/
│   └── config.yaml             # Audio, model, and training hyperparameters
├── data/
│   ├── raw/                    # Mozilla Common Voice raw files
│   ├── synthetic/              # Benchmark synthetic speech dataset
│   └── samples/                # Sample audio files
├── src/
│   ├── audio/
│   │   ├── preprocessor.py     # Resampling, silence trimming, normalization
│   │   └── features.py         # MFCC, F0 (pYIN), formants (LPC), spectral features
│   ├── data/
│   │   └── loader.py           # Mozilla Common Voice loader & group splitter
│   ├── models/
│   │   ├── classical.py        # SVM and Random Forest pipelines
│   │   └── rnn.py              # PyTorch BiLSTM with Attention & multi-task heads
│   ├── training/
│   │   └── train_all.py        # Unified training & evaluation orchestrator
│   └── evaluation/
│       └── benchmark.py        # Side-by-side model comparison metrics
├── tests/
│   └── test_audio_features.py  # Unit tests for preprocessing & feature extraction
├── Dockerfile                  # Containerization for Railway / Render
├── railway.json                # Railway deployment configuration
└── pyproject.toml              # uv Python project definition
```

---

## 🛠️ Quickstart

### 1. Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for fast and deterministic Python environment management:

```bash
# Clone repository and enter directory
cd GenderSpeachRecognationProject

# Install dependencies with uv
uv sync
```

### 2. Run Tests

Verify feature extraction and audio preprocessing:

```bash
uv run pytest
```

### 3. Train Models & Run Comparative Benchmark

You can immediately train all models and generate the comparative benchmark table on a synthetic speech dataset:

```bash
uv run python -m src.training.train_all --synthetic --num-samples 150 --epochs 15
```

To train on real **Mozilla Common Voice** data:
1. Place extracted Common Voice files in `data/raw/` (containing `validated.tsv` and `clips/`).
2. Run:
   ```bash
   uv run python -m src.training.train_all --data-dir data/raw --epochs 25
   ```

Trained models and the comparative benchmark report will be saved to `saved_models/`:
- `svm_pipeline.joblib`
- `rf_pipeline.joblib`
- `bilstm_pipeline.pt`
- `benchmark_comparison.csv`
- `evaluation_metrics.json`

### 4. Launch the Web App Locally

```bash
uv run streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deploying on Free Cloud Hosting (Railway)

### Option A: Railway (One-Click Git Push)

1. Push this repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Gender and age speech prediction system"
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. Log in to [Railway.app](https://railway.com/).
3. Click **"New Project"** -> **"Deploy from GitHub repo"**.
4. Select your repository.
5. Railway will automatically detect `railway.json` and `Dockerfile`, build the CPU-only container, and assign a free HTTPS domain!

### Option B: Hugging Face Spaces (100% Free Forever Alternative)

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) with **Streamlit** SDK.
2. Push this repository directly to the Hugging Face Space repository.
3. Hugging Face provides **16 GB RAM and 2 vCPUs completely free**.

---

## 🔬 Acoustic Feature Details

| Feature | Method | Diagnostic Value |
| :--- | :--- | :--- |
| **MFCC (1-20)** | Discrete Cosine Transform of mel-scaled filterbank | Captures vocal tract shape independent of loudness. |
| **Pitch ($F_0$)** | Probabilistic YIN (`librosa.pyin`) | Discriminates between biological male (~120 Hz) and female (~210 Hz) vocal fold frequency. |
| **Formants ($F_1\text{--}F_4$)** | Linear Predictive Coding (LPC) filter polynomial roots | Reflects physical vocal tract length; male formants are shifted ~15% lower than female formants. |
| **Jitter & Voice Quality** | Relative cycle-to-cycle perturbation & spectral rolloff | Correlates with age-related vocal stability and tissue mass changes. |
