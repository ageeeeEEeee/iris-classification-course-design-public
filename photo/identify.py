"""Choose a local image and predict genus and species."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "iris_photo.keras"
LABELS = {"setosa": "Iris setosa", "versicolor": "Iris versicolor", "virginica": "Iris virginica"}


class Classifier:
    def __init__(self):
        from tensorflow import keras

        metadata = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
        self.model = keras.models.load_model(MODEL, compile=False)
        self.labels = metadata["species_order"]
        self.threshold = metadata["thresholds"]
        self.size = tuple(self.model.input_shape[1:3])

    def predict(self, path):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        data = np.asarray(ImageOps.pad(image, self.size, color=(127, 127, 127)), dtype=np.uint8)
        output = self.model(data[None], training=False)
        iris_score = float(output["iris"][0, 0])
        scores = output["species"][0].numpy()
        top = int(scores.argmax())
        taxon = self.labels[top]
        if iris_score <= self.threshold["iris_low"]:
            return image, {"鸢尾花判断": "不像鸢尾花", "鸢尾得分": round(iris_score, 3), "品种": "—"}
        if iris_score < self.threshold["iris_high"]:
            return image, {"鸢尾花判断": "无法确定", "鸢尾得分": round(iris_score, 3), "品种": "未确定"}
        result = {"鸢尾花判断": "可能是鸢尾花", "鸢尾得分": round(iris_score, 3), "品种": "未确定"}
        if taxon in LABELS and scores[top] >= self.threshold["species"]:
            result["品种"] = LABELS[taxon]
            result["品种得分"] = round(float(scores[top]), 3)
        elif taxon in LABELS:
            result["待复核候选"] = LABELS[taxon]
            result["候选得分"] = round(float(scores[top]), 3)
        return image, result


def show_app():
    import tkinter as tk
    from tkinter import filedialog, ttk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("鸢尾花照片识别")
    root.geometry("540x640")
    root.minsize(420, 570)
    content = ttk.Frame(root, padding=20)
    content.pack(fill="both", expand=True)
    ttk.Label(content, text="鸢尾花照片识别", font=("Microsoft YaHei", 19, "bold")).pack(anchor="w")
    ttk.Label(content, text="选择一张主体清晰的照片。图片在本机处理。").pack(anchor="w", pady=(4, 12))
    preview = ttk.Label(content, anchor="center")
    output = tk.StringVar(value="等待选择图片")
    result_label = ttk.Label(content, textvariable=output, justify="left", font=("Microsoft YaHei", 12))
    classifier = None

    def select():
        nonlocal classifier
        file = filedialog.askopenfilename(filetypes=[
            ("图片", ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")), ("所有文件", "*.*"),
        ])
        if not file:
            return
        output.set("正在识别…")
        preview.configure(image="")
        root.update_idletasks()
        try:
            if classifier is None:
                classifier = Classifier()
            image, result = classifier.predict(file)
            image.thumbnail((480, 320))
            preview.image = ImageTk.PhotoImage(image)
            preview.configure(image=preview.image)
            output.set("\n".join(f"{key}：{value}" for key, value in result.items()))
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            output.set(f"无法处理图片或加载模型。\n{str(exc)[:160]}")

    ttk.Button(content, text="选择图片", command=select).pack(anchor="w")
    preview.pack(fill="x", pady=12)
    result_label.pack(anchor="w")
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="Omit the path to open the image chooser.")
    args = parser.parse_args()
    if args.image:
        try:
            _, result = Classifier().predict(args.image)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        show_app()


if __name__ == "__main__":
    main()
