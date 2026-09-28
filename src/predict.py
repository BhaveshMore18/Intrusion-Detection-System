"""
predict.py

Command-line scanner: extract features from a file, run it through the
trained model, print benign/malicious with a confidence score.

Usage:
    python src/predict.py path/to/file.exe
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_features  # noqa: E402
from model import DeepNeuralNetwork  # noqa: E402
from scaler import StandardScaler  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT, "models")


def load_pipeline():
    model_path = os.path.join(MODELS_DIR, "ids_model.npz")
    scaler_path = os.path.join(MODELS_DIR, "scaler.npz")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            "No trained model found. Run `python src/train.py` first."
        )
    model = DeepNeuralNetwork.load(model_path)
    scaler = StandardScaler.load(scaler_path)
    return model, scaler


def scan_file(path: str, model=None, scaler=None):
    if model is None or scaler is None:
        model, scaler = load_pipeline()

    features = extract_features(path).reshape(1, -1)
    features_scaled = scaler.transform(features)
    probability = float(model.predict_proba(features_scaled)[0])
    label = "MALICIOUS" if probability >= 0.5 else "BENIGN"
    return {"path": path, "label": label, "confidence": probability}


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <path-to-file>")
        sys.exit(1)

    target = sys.argv[1]
    if not os.path.isfile(target):
        print(f"No such file: {target}")
        sys.exit(1)

    result = scan_file(target)
    print(f"\nFile:       {result['path']}")
    print(f"Verdict:    {result['label']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")


if __name__ == "__main__":
    main()
