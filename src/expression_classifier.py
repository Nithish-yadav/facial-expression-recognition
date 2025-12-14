# src/expression_classifier.py

import cv2
from typing import List
from deepface import DeepFace


class ExpressionClassifier:
    """
    Real-time expression classifier using DeepFace's pretrained emotion model.

    NOTE:
    - We STILL keep our HDD-LBP + SVM training code in train_svm.py for the report.
    - But for live webcam prediction, we use this deep model because it's
      much more accurate and robust than our small SVM.
    """

    def __init__(self, model_path: str) -> None:
        # model_path is not used here, but kept so main.py doesn't break
        self.model_path = model_path
        # DeepFace loads the emotion model lazily on first call
        self.classes_: List[str] = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
        ]
        print("[INFO] Using DeepFace pretrained emotion model for real-time prediction.")

    def predict_emotion(self, gray_face):
        """
        Takes a grayscale face image (numpy array), returns predicted emotion label (string).
        Uses DeepFace's emotion model.
        """
        # DeepFace expects a 3-channel image; convert gray → BGR
        face_bgr = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)

        try:
            result = DeepFace.analyze(
                img_path=face_bgr,
                actions=["emotion"],
                enforce_detection=False  # IMPORTANT for small / partial faces
            )

            # DeepFace may return a list (newer versions) or dict (older).
            if isinstance(result, list):
                result = result[0]

            dominant_emotion = result.get("dominant_emotion", "neutral")
            # beautify label: "happy" -> "Happy"
            return dominant_emotion.capitalize()

        except Exception as e:
            print(f"[WARN] DeepFace prediction error: {e}")
            return "Neutral"
