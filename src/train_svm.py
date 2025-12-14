import os
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

from .feature_extraction import extract_hddl_lbp_features

CLASS_NAMES: List[str] = [
    "Happy",
    "Sad",
    "Angry",
]


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    y: List[int] = []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_folder):
            continue

        print(f"[INFO] Loading class '{class_name}' from {class_folder}")

        for fname in os.listdir(class_folder):
            fpath = os.path.join(class_folder, fname)

            if not os.path.isfile(fpath):
                continue
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            img = cv2.imread(fpath)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            try:
                feats = extract_hddl_lbp_features(gray)
                X.append(feats)
                y.append(label_idx)
            except Exception:
                continue

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main() -> None:
    root = get_project_root()
    data_dir = os.path.join(root, "data")
    models_dir = os.path.join(root, "models")

    os.makedirs(models_dir, exist_ok=True)

    print("[INFO] Loading dataset...")
    X, y = load_dataset(data_dir)
    print(f"[INFO] Dataset loaded. Samples: {len(y)}")

    if len(y) == 0:
        return

    unique, counts = np.unique(y, return_counts=True)
    for idx, cnt in zip(unique, counts):
        print(f"{CLASS_NAMES[idx]}: {cnt}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10,
            gamma=0.01,
            class_weight="balanced",
        ),
    )

    print("[INFO] Training model...")
    clf.fit(X_train, y_train)

    print("[INFO] Evaluating model...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Test Accuracy: {acc * 100:.2f}%")

    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    model_path = os.path.join(models_dir, "svm_expression_model.pkl")
    joblib.dump({"model": clf, "classes": CLASS_NAMES}, model_path)
    print(f"[INFO] Model saved to {model_path}")


if __name__ == "__main__":
    main()
