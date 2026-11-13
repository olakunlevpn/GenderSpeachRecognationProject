"""Gender and Age Prediction from Speech: Command-Line Interface."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Gender and Age Prediction from Voice Signals (Classical ML vs Deep Learning)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: train
    train_parser = subparsers.add_parser("train", help="Train and benchmark models")
    train_parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset")
    train_parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset")
    train_parser.add_argument("--num-samples", type=int, default=150, help="Number of synthetic samples")
    train_parser.add_argument("--epochs", type=int, default=15, help="BiLSTM epochs")
    train_parser.add_argument("--save-dir", type=str, default="saved_models", help="Model directory")

    # Subcommand: ui / app
    ui_parser = subparsers.add_parser("app", help="Launch interactive Streamlit web app")
    ui_parser.add_argument("--port", type=int, default=8501, help="Port to serve Streamlit app")

    # Subcommand: predict
    predict_parser = subparsers.add_parser("predict", help="Predict gender and age for an audio file")
    predict_parser.add_argument("audio_path", type=str, help="Path to audio file (.wav, .mp3, etc.)")
    predict_parser.add_argument("--model", type=str, default="svm", choices=["svm", "rf", "bilstm"])

    args = parser.parse_args()

    if args.command == "train":
        from src.training.train_all import run_pipeline

        run_pipeline(
            data_dir=args.data_dir,
            use_synthetic=args.synthetic,
            num_synthetic=args.num_samples,
            epochs=args.epochs,
            save_dir=args.save_dir,
        )
    elif args.command == "app":
        print(f"Launching Streamlit on port {args.port}...")
        subprocess.run(["streamlit", "run", "app/streamlit_app.py", f"--server.port={args.port}"])
    elif args.command == "predict":
        from src.audio.preprocessor import load_and_preprocess_audio

        path = Path(args.audio_path)
        if not path.is_file():
            print(f"Error: file not found: {path}")
            sys.exit(1)

        y, sr = load_and_preprocess_audio(path)

        if args.model == "bilstm":
            from src.models.rnn import DeepSpeechPipeline
            pipe = DeepSpeechPipeline.load("saved_models/bilstm_pipeline.pt")
        elif args.model == "rf":
            from src.models.classical import ClassicalSpeechPipeline
            pipe = ClassicalSpeechPipeline.load("saved_models/rf_pipeline.joblib")
        else:
            from src.models.classical import ClassicalSpeechPipeline
            pipe = ClassicalSpeechPipeline.load("saved_models/svm_pipeline.joblib")

        res = pipe.predict_audio(y, sr=sr)
        print("\n--- Prediction Results ---")
        print(f"Audio file: {path.name}")
        print(f"Predicted Gender: {res['gender'].upper()} (Confidence: {res.get('gender_confidence', 0.0)*100:.1f}%)")
        print(f"Predicted Age Range: {res['age_range']} (Confidence: {res.get('age_confidence', 0.0)*100:.1f}%)")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
