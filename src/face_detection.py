from typing import List, Tuple
import cv2


class FaceDetector:
    """
    Simple Haar Cascade face detector.
    """

    def __init__(self, cascade_path: str) -> None:
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise IOError(f"Failed to load Haar cascade from: {cascade_path}")

    def detect_faces(self, gray_frame) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a grayscale frame.
        Returns list of (x, y, w, h).
        """
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        return list(faces)
