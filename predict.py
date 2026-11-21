"""Predict the species from four Iris measurements in centimetres."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
from tensorflow import keras

CLASSES = ("setosa", "versicolor", "virginica")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for feature in ("sepal_length", "sepal_width", "petal_length", "petal_width"):
        parser.add_argument(feature, type=float)
    args = parser.parse_args()
    with np.errstate(over="ignore"):
        features = np.asarray([list(vars(args).values())], dtype="float32")
    if not np.isfinite(features).all() or (features <= 0).any():
        parser.error("measurements must be positive finite numbers representable as float32")
    model_path = Path(__file__).resolve().parent / "results" / "iris.keras"
    if not model_path.is_file():
        parser.error("model file is missing; run python train.py first")
    model = keras.models.load_model(model_path, compile=False)
    probabilities = model(features, training=False).numpy()[0]
    if not np.isfinite(probabilities).all():
        parser.error("measurements are too large for a valid prediction")
    print(json.dumps({
        "prediction": CLASSES[int(probabilities.argmax())],
        "probabilities": dict(zip(CLASSES, probabilities.tolist())),
    }, indent=2))


if __name__ == "__main__":
    main()
