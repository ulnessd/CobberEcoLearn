#!/usr/bin/env python3
"""
train_bloom_cnn.py

Train a CNN classifier for CobberEcoBloom using a dataset produced by:

    generate_bloom_dataset.py

Typical usage:
    python train_bloom_cnn.py BloomData_2000

Recommended first runs:
    python train_bloom_cnn.py BloomData_2000 --epochs 12 --batch-size 64 --mixed-precision
    python train_bloom_cnn.py BloomData_8000 --epochs 12 --batch-size 64 --mixed-precision

Outputs:
    BloomModel_2000/
        cobber_bloom_model_2000.keras
        cobber_bloom_label_encoder_2000.pkl
        training_history_2000.csv
        training_curve_2000.png
        confusion_matrix_2000.png
        classification_report_2000.txt
        training_summary_2000.txt

Dependencies:
    pip install tensorflow pandas numpy scikit-learn matplotlib

Notes:
    - TensorFlow will use the GPU if your install can see it.
    - The script enables TensorFlow GPU memory growth by default.
    - Use --device cpu if you want to force CPU training.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CobberEcoBloom CNN model.")
    parser.add_argument(
        "dataset_dir",
        help="Dataset directory produced by generate_bloom_dataset.py, e.g. BloomData_2000",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Supervisor CSV path. Default: <dataset_dir>/bloom_supervisor.csv",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default: BloomModel_<N>",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Image size used for training. Should match generation size. Default: 128",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Maximum number of training epochs. Default: 12",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size. Default: 64",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate. Default: 0.001",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="Device preference. Default: auto",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use TensorFlow mixed_float16 policy. Usually helpful on modern NVIDIA GPUs.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache decoded/resized images in RAM. Fast, but uses more memory.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable simple image augmentation during training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed. Default: 17",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early-stopping patience. Default: 4",
    )
    return parser.parse_args()


def infer_dataset_size(dataset_dir: Path, csv_path: Path) -> str:
    match = re.search(r"(\d+)", dataset_dir.name)
    if match:
        return match.group(1)

    try:
        import pandas as pd
        return str(len(pd.read_csv(csv_path)))
    except Exception:
        return "dataset"


def configure_tensorflow(args):
    # Must set this before importing TensorFlow if forcing CPU.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import numpy as np
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)

    gpus = tf.config.list_physical_devices("GPU")

    if args.device == "gpu" and not gpus:
        print("WARNING: --device gpu was requested, but TensorFlow reports no GPU.")
        print("         Training will fail over to CPU unless your environment is fixed.")

    if gpus:
        print("TensorFlow-visible GPUs:")
        for gpu in gpus:
            print(f"  {gpu}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:
                print(f"WARNING: Could not set memory growth on {gpu}: {exc}")

        if args.mixed_precision:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                print("Mixed precision enabled: mixed_float16")
            except Exception as exc:
                print(f"WARNING: Could not enable mixed precision: {exc}")
    else:
        print("TensorFlow-visible GPUs: none")
        if args.mixed_precision:
            print("Mixed precision requested, but no GPU was visible. Continuing anyway.")

    return tf, np


def load_supervisor(dataset_dir: Path, csv_path: Path):
    import pandas as pd

    df = pd.read_csv(csv_path)

    required = {"label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Supervisor CSV is missing required columns: {missing}")

    if "relative_path" not in df.columns and "filename" not in df.columns:
        raise ValueError("Supervisor CSV needs either 'relative_path' or 'filename' column.")

    def resolve_path(row):
        """
        Resolve image paths robustly.

        Early dataset-generator versions wrote relative_path as:
            clear_water/clear_water_000001.png

        while the actual files live under:
            images/clear_water/clear_water_000001.png

        Newer/future generators may write:
            images/clear_water/clear_water_000001.png

        This resolver accepts both forms and also falls back to label/filename.
        """
        candidates = []

        rel = row.get("relative_path", None)
        if isinstance(rel, str) and rel.strip():
            rel_path = Path(rel)
            candidates.append(dataset_dir / rel_path)
            candidates.append(dataset_dir / "images" / rel_path)

        filename = row.get("filename", None)
        label = row.get("label", None)
        if isinstance(filename, str) and isinstance(label, str):
            candidates.append(dataset_dir / "images" / label / filename)
            candidates.append(dataset_dir / label / filename)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Return the most likely path so the error message is helpful.
        if candidates:
            return candidates[0]
        return dataset_dir / "MISSING_PATH"

    df["path"] = df.apply(resolve_path, axis=1).astype(str)

    missing_files = [p for p in df["path"].tolist() if not Path(p).exists()]
    if missing_files:
        preview = "\n".join(missing_files[:5])
        raise FileNotFoundError(f"{len(missing_files)} image files listed in CSV were not found. First few:\n{preview}")

    return df


def build_label_encoder(df, outpath: Path):
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()

    # Fit in a stable, ecological order if all expected labels exist.
    preferred_order = [
        "clear_water",
        "mostly_clear",
        "field_check",
        "mostly_bloom",
        "dense_bloom",
    ]

    labels_in_data = set(df["label"].astype(str).unique())
    if labels_in_data.issubset(set(preferred_order)):
        fit_labels = [x for x in preferred_order if x in labels_in_data]
    else:
        fit_labels = sorted(labels_in_data)

    le.fit(fit_labels)

    with open(outpath, "wb") as f:
        pickle.dump(le, f)

    return le


def make_tf_dataset(tf, paths: List[str], y: List[int], image_size: int, batch_size: int,
                    shuffle: bool, cache: bool):
    ds = tf.data.Dataset.from_tensor_slices((paths, y))

    def load_image(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_png(raw, channels=3)
        img = tf.image.resize(img, [image_size, image_size], method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 10000), seed=17, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(tf, image_size: int, num_classes: int, learning_rate: float, use_augment: bool):
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))

    x = inputs

    if use_augment:
        x = layers.RandomFlip("horizontal_and_vertical")(x)
        x = layers.RandomRotation(0.05)(x)
        x = layers.RandomZoom(0.08)(x)

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(192, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    # Keep final output float32 for numerical stability with mixed precision.
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_history_csv(history, outpath: Path):
    import pandas as pd

    hist = history.history
    df = pd.DataFrame(hist)
    df.insert(0, "epoch", range(1, len(df) + 1))
    df.to_csv(outpath, index=False)


def plot_training_curve(history, outpath: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)

    plt.figure(figsize=(7.5, 5.0))
    plt.plot(epochs, hist.get("loss", []), label="training loss")
    if "val_loss" in hist:
        plt.plot(epochs, hist["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CobberEcoBloom training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

    if "accuracy" in hist:
        acc_path = outpath.with_name(outpath.stem.replace("training_curve", "accuracy_curve") + outpath.suffix)
        plt.figure(figsize=(7.5, 5.0))
        plt.plot(epochs, hist.get("accuracy", []), label="training accuracy")
        if "val_accuracy" in hist:
            plt.plot(epochs, hist["val_accuracy"], label="validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("CobberEcoBloom accuracy curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(acc_path, dpi=160)
        plt.close()


def evaluate_and_save(tf, model, test_ds, le, outdir: Path, size_tag: str):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report

    y_true = []
    y_pred = []

    for batch_images, batch_labels in test_ds:
        preds = model.predict(batch_images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_true.extend(batch_labels.numpy().tolist())

    labels_int = list(range(len(le.classes_)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_int)

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_int,
        target_names=le.classes_,
        digits=4,
        zero_division=0,
    )

    report_path = outdir / f"classification_report_{size_tag}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm)
    ax.set_title("CobberEcoBloom confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=45, ha="right")
    ax.set_yticklabels(le.classes_)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    cm_path = outdir / f"confusion_matrix_{size_tag}.png"
    fig.savefig(cm_path, dpi=160)
    plt.close(fig)

    accuracy = sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
    return accuracy, report_path, cm_path, cm


def write_summary(outpath: Path, lines: List[str]):
    with open(outpath, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def main() -> int:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    csv_path = Path(args.csv).resolve() if args.csv else dataset_dir / "bloom_supervisor.csv"
    if not csv_path.exists():
        raise SystemExit(f"Supervisor CSV not found: {csv_path}")

    size_tag = infer_dataset_size(dataset_dir, csv_path)
    outdir = Path(args.outdir).resolve() if args.outdir else Path.cwd() / f"BloomModel_{size_tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CobberEcoBloom CNN trainer")
    print("=" * 72)
    print(f"Dataset directory:   {dataset_dir}")
    print(f"Supervisor CSV:      {csv_path}")
    print(f"Output directory:    {outdir}")
    print(f"Image size:          {args.image_size} x {args.image_size}")
    print(f"Epochs:              {args.epochs}")
    print(f"Batch size:          {args.batch_size}")
    print(f"Learning rate:       {args.learning_rate}")
    print(f"Device preference:   {args.device}")
    print(f"Mixed precision:     {args.mixed_precision}")
    print(f"Cache in RAM:        {args.cache}")
    print(f"Augmentation:        {not args.no_augment}")
    print("-" * 72)

    # Configure TensorFlow after parsing args, so --device cpu can set CUDA_VISIBLE_DEVICES.
    tf, np = configure_tensorflow(args)

    import pandas as pd

    t0 = time.perf_counter()

    df = load_supervisor(dataset_dir, csv_path)
    le_path = outdir / f"cobber_bloom_label_encoder_{size_tag}.pkl"
    le = build_label_encoder(df, le_path)

    df["label_id"] = le.transform(df["label"].astype(str))

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise SystemExit("Train/val/test split is incomplete. Check bloom_supervisor.csv.")

    print("Dataset split counts:")
    print(f"  train: {len(train_df)}")
    print(f"  val:   {len(val_df)}")
    print(f"  test:  {len(test_df)}")
    print("")
    print("Classes:")
    for i, cls in enumerate(le.classes_):
        print(f"  {i}: {cls}")
    print("-" * 72)

    train_ds = make_tf_dataset(
        tf,
        train_df["path"].tolist(),
        train_df["label_id"].tolist(),
        args.image_size,
        args.batch_size,
        shuffle=True,
        cache=args.cache,
    )
    val_ds = make_tf_dataset(
        tf,
        val_df["path"].tolist(),
        val_df["label_id"].tolist(),
        args.image_size,
        args.batch_size,
        shuffle=False,
        cache=args.cache,
    )
    test_ds = make_tf_dataset(
        tf,
        test_df["path"].tolist(),
        test_df["label_id"].tolist(),
        args.image_size,
        args.batch_size,
        shuffle=False,
        cache=args.cache,
    )

    model = build_model(
        tf,
        image_size=args.image_size,
        num_classes=len(le.classes_),
        learning_rate=args.learning_rate,
        use_augment=not args.no_augment,
    )

    model_path = outdir / f"cobber_bloom_model_{size_tag}.keras"
    best_model_path = outdir / f"cobber_bloom_model_{size_tag}_best.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, args.patience // 2),
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("Model summary:")
    model.summary()
    print("-" * 72)

    train_start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    train_end = time.perf_counter()

    # Save final model as the main file students can load.
    model.save(model_path)

    history_path = outdir / f"training_history_{size_tag}.csv"
    curve_path = outdir / f"training_curve_{size_tag}.png"
    save_history_csv(history, history_path)
    plot_training_curve(history, curve_path)

    print("-" * 72)
    print("Evaluating on test split...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    cm_acc, report_path, cm_path, cm = evaluate_and_save(tf, model, test_ds, le, outdir, size_tag)

    t1 = time.perf_counter()

    train_seconds = train_end - train_start
    total_seconds = t1 - t0
    effective_images = len(train_df) * len(history.history.get("loss", []))
    images_per_sec = effective_images / train_seconds if train_seconds > 0 else float("inf")

    summary_lines = [
        "CobberEcoBloom training summary",
        "=" * 60,
        f"Dataset directory: {dataset_dir}",
        f"Supervisor CSV: {csv_path}",
        f"Output directory: {outdir}",
        "",
        f"Train images: {len(train_df)}",
        f"Validation images: {len(val_df)}",
        f"Test images: {len(test_df)}",
        f"Classes: {', '.join(le.classes_)}",
        "",
        f"Image size: {args.image_size}",
        f"Batch size: {args.batch_size}",
        f"Epochs requested: {args.epochs}",
        f"Epochs completed: {len(history.history.get('loss', []))}",
        f"Learning rate: {args.learning_rate}",
        f"Mixed precision: {args.mixed_precision}",
        f"RAM cache: {args.cache}",
        f"Augmentation: {not args.no_augment}",
        "",
        f"Test loss: {test_loss:.6f}",
        f"Test accuracy from model.evaluate: {test_acc:.6f}",
        f"Test accuracy from confusion matrix: {cm_acc:.6f}",
        "",
        f"Training time: {train_seconds:.3f} s",
        f"Total wall time: {total_seconds:.3f} s",
        f"Effective training images/sec: {images_per_sec:.2f}",
        "",
        f"Saved model: {model_path}",
        f"Saved best model: {best_model_path}",
        f"Saved label encoder: {le_path}",
        f"Saved history: {history_path}",
        f"Saved training curve: {curve_path}",
        f"Saved confusion matrix: {cm_path}",
        f"Saved classification report: {report_path}",
    ]

    summary_path = outdir / f"training_summary_{size_tag}.txt"
    write_summary(summary_path, summary_lines)

    print("")
    print("\n".join(summary_lines))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
