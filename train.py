"""Train and evaluate a small TensorFlow classifier for Iris measurements."""
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from tensorflow import keras

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SEED = 42
OUTPUT = Path(__file__).resolve().parent / "results"


def create_model(training_features):
    normalizer = keras.layers.Normalization()
    normalizer.adapt(training_features)
    model = keras.Sequential([
        keras.Input(shape=(4,)),
        normalizer,
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate(labels, probabilities, class_names):
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "loss": log_loss(labels, probabilities, labels=[0, 1, 2]),
        "report": classification_report(
            labels, predictions, labels=[0, 1, 2], target_names=class_names,
            output_dict=True, zero_division=0,
        ),
        "matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
    }


def draw_results(history, matrix, names, best_epoch):
    fig, (curve, confusion) = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
    epochs = range(1, len(history["loss"]) + 1)
    curve.plot(epochs, history["loss"], label="Train")
    curve.plot(epochs, history["val_loss"], label="Validation")
    curve.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label="Best epoch")
    curve.set(xlabel="Epoch", ylabel="Cross-entropy", title="Learning curve")
    curve.legend(frameon=False)
    confusion.imshow(matrix, cmap="Blues", vmin=0, vmax=10)
    confusion.set(xticks=range(3), yticks=range(3), xticklabels=names, yticklabels=names,
                  xlabel="Predicted class", ylabel="True class", title="Test confusion matrix")
    confusion.tick_params(axis="x", labelrotation=20)
    for row, counts in enumerate(matrix):
        for column, count in enumerate(counts):
            confusion.text(column, row, str(count), ha="center", va="center",
                           color="white" if count > 5 else "black")
    fig.savefig(OUTPUT / "evaluation.png", dpi=180)
    plt.close(fig)


def main():
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    data = load_iris()
    features = data.data.astype("float32")
    labels = data.target
    names = data.target_names.tolist()
    development, test = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=SEED,
    )
    train, validation = train_test_split(
        development, test_size=0.25, stratify=labels[development], random_state=SEED,
    )
    model = create_model(features[train])
    stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True,
    )
    history = model.fit(
        features[train], labels[train], validation_data=(features[validation], labels[validation]),
        epochs=250, batch_size=16, callbacks=[stopping], verbose=0,
    ).history
    probabilities = model(features[test], training=False).numpy()
    deep = evaluate(labels[test], probabilities, names)
    normalizer = model.layers[0]
    baseline = LogisticRegression(C=1.0, max_iter=1000)
    baseline.fit(normalizer(features[train]).numpy(), labels[train])
    linear = evaluate(labels[test], baseline.predict_proba(normalizer(features[test]).numpy()), names)
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    errors = [
        {"index": int(i), "features_cm": data.data[i].tolist(), "true": names[labels[i]],
         "predicted": names[row.argmax()], "probabilities": row.tolist()}
        for i, row in zip(test, probabilities) if labels[i] != row.argmax()
    ]
    OUTPUT.mkdir(exist_ok=True)
    model.save(OUTPUT / "iris.keras")
    restored = keras.models.load_model(OUTPUT / "iris.keras", compile=False)
    np.testing.assert_allclose(restored(features[test], training=False), probabilities, atol=1e-7)
    metrics = {
        "environment": {"python": platform.python_version(), "tensorflow": tf.__version__,
                        "scikit_learn": sklearn.__version__, "platform": platform.system()},
        "seed": SEED, "split_indices": {"train": train.tolist(), "validation": validation.tolist(),
                                        "test": test.tolist()},
        "best_epoch": best_epoch, "epochs": len(history["loss"]), "history": history,
        "train_at_best": {key: values[best_epoch - 1] for key, values in history.items()},
        "mlp": deep, "logistic_regression": linear, "errors": errors,
        "example": {"features_cm": data.data[0].tolist(),
                    "probabilities": model(features[:1], training=False).numpy()[0].tolist(),
                    "classes": names},
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    draw_results(history, deep["matrix"], names, best_epoch)
    print(f"MLP accuracy: {deep['accuracy']:.4f}; baseline accuracy: {linear['accuracy']:.4f}")
    print(f"Best epoch: {best_epoch}; epochs run: {metrics['epochs']}")


if __name__ == "__main__":
    main()
