"""Train and evaluate two image classifiers on shared EfficientNetV2B0 features."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
SIZE = 224
SPECIES = ("other_iris", "setosa", "versicolor", "virginica")
LOW, HIGH, SPECIES_MIN = 0.40, 0.60, 0.65


def pixels(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    return np.asarray(ImageOps.pad(image, (SIZE, SIZE), color=(127, 127, 127)), dtype=np.uint8)


def build_features(cache):
    inputs = keras.Input(shape=(SIZE, SIZE, 3), dtype="uint8")
    x = keras.layers.Rescaling(1.0)(inputs)
    backbone = keras.applications.EfficientNetV2B0(
        input_shape=(SIZE, SIZE, 3), include_top=False, pooling="avg", weights=None,
    )
    backbone.load_weights(cache / "efficientnetv2_b0.h5")
    backbone.trainable = False
    features = keras.layers.UnitNormalization()(backbone(x, training=False))
    return keras.Model(inputs, features)


def add_classifier(features, classifier, name):
    binary = len(classifier.classes_) == 2
    layer = keras.layers.Dense(1 if binary else len(SPECIES),
                               activation="sigmoid" if binary else "softmax", name=name)
    output = layer(features)
    layer.set_weights([classifier.coef_.T.astype("float32"), classifier.intercept_.astype("float32")])
    return output


def main():
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    with (ROOT / "image_sources.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = np.array([row["label"] for row in rows])
    split = np.array([row["split"] for row in rows])
    for partition in ("train", "validation", "test"):
        if any(np.count_nonzero((split == partition) & (labels == group)) < 3
               for group in (*SPECIES, "not_iris")):
            raise ValueError(f"Insufficient independent samples in {partition}")
    cache = ROOT / "dataset"
    inputs = []
    for row in rows:
        with Image.open(cache / row["filename"]) as image:
            inputs.append(pixels(image))
    inputs = np.stack(inputs)
    feature_model = build_features(cache)
    embeddings = feature_model.predict(inputs, batch_size=24, verbose=0)
    iris = labels != "not_iris"
    train = split == "train"
    genus_model = LogisticRegression(C=18, max_iter=800, class_weight="balanced")
    genus_model.fit(embeddings[train], iris[train].astype(int))
    species_model = LogisticRegression(C=18, max_iter=800, class_weight="balanced")
    species_model.fit(embeddings[train & iris], labels[train & iris])
    assert tuple(species_model.classes_) == SPECIES

    genus_output = add_classifier(feature_model.output, genus_model, "iris")
    species_output = add_classifier(feature_model.output, species_model, "species")
    model = keras.Model(feature_model.input, {"iris": genus_output, "species": species_output})
    model.save(ROOT / "iris_photo.keras")
    model = keras.models.load_model(ROOT / "iris_photo.keras", compile=False)
    output = model.predict(inputs, batch_size=24, verbose=0)
    genus_scores, species_scores = output["iris"][:, 0], output["species"]
    np.testing.assert_allclose(genus_scores, genus_model.predict_proba(embeddings)[:, 1], atol=1e-5)
    np.testing.assert_allclose(species_scores, species_model.predict_proba(embeddings), atol=1e-5)
    names = np.array(SPECIES)[species_scores.argmax(axis=1)]
    evaluation = {"thresholds": {"iris_low": LOW, "iris_high": HIGH, "species": SPECIES_MIN},
                  "species_order": list(SPECIES), "counts": {}, "evaluation": {}}
    for partition in ("train", "validation", "test"):
        mask = split == partition
        evaluation["counts"][partition] = {name: int(np.count_nonzero(mask & (labels == name)))
                                             for name in (*SPECIES, "not_iris")}
        if partition == "train":
            continue
        known = mask & iris
        detected = (genus_scores >= HIGH) & (species_scores.max(axis=1) >= SPECIES_MIN)
        explicit = detected & (names != "other_iris")
        evaluation["evaluation"][partition] = {
            "genus_confusion_matrix_at_0_5": confusion_matrix(
                iris[mask], genus_scores[mask] >= .5, labels=[False, True],
            ).tolist(),
            "genus_uncertain": int(np.count_nonzero(mask & (genus_scores > LOW) & (genus_scores < HIGH))),
            "genus_decided_correct": int(np.count_nonzero(
                mask & ((iris & (genus_scores >= HIGH)) | (~iris & (genus_scores <= LOW))),
            )),
            "species_confusion_matrix_iris_only": confusion_matrix(labels[known], names[known],
                                                                   labels=SPECIES).tolist(),
            "end_to_end_explicit_species_correct": int(np.count_nonzero(mask & explicit & (labels == names))),
            "end_to_end_explicit_species_predictions": int(np.count_nonzero(mask & explicit)),
        }
    (ROOT / "metrics.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(evaluation, indent=2), flush=True)


if __name__ == "__main__":
    main()
