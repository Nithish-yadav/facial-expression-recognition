import os
import sys
import cv2
from collections import Counter

from .face_detection import FaceDetector
from .face_tracking import CentroidTracker
from .expression_classifier import ExpressionClassifier
from .utils import draw_labelled_box, save_frame


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_haar_path() -> str:
    root = get_project_root()
    return os.path.join(root, "haarcascades", "haarcascade_frontalface_default.xml")


def get_model_path() -> str:
    root = get_project_root()
    return os.path.join(root, "models", "svm_expression_model.pkl")


def main() -> None:
    cascade_path = get_haar_path()
    model_path = get_model_path()

    face_detector = FaceDetector(cascade_path)
    tracker = CentroidTracker(max_disappeared=15)
    expr_classifier = ExpressionClassifier(model_path=model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)

    root = get_project_root()
    save_folder = os.path.join(root, "saved_frames")

    print("[INFO] Press 's' to save current frame, 'q' to quit.")

 
    WINDOW_SIZE = 7
    emotion_history = {}  

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = face_detector.detect_faces(gray)
        tracked = tracker.update(rects)

        for object_id, bbox in tracked.items():
            x, y, w, h = bbox

            x = max(0, x)
            y = max(0, y)
            w = max(1, w)
            h = max(1, h)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)

            face_gray = gray[y:y2, x:x2]
            if face_gray.size == 0:
                continue

 
            emotion = expr_classifier.predict_emotion(face_gray)

            if object_id not in emotion_history:
                emotion_history[object_id] = []

            emotion_history[object_id].append(emotion)

            if len(emotion_history[object_id]) > WINDOW_SIZE:
                emotion_history[object_id].pop(0)

            
            final_emotion = Counter(emotion_history[object_id]).most_common(1)[0][0]

            label = f"ID {object_id}: {final_emotion}"
            draw_labelled_box(frame, (x, y, w, h), label)

        cv2.imshow("Real-Time Facial Expression Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quitting...")
            break
        elif key == ord("s"):
            save_frame(frame, save_folder)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
