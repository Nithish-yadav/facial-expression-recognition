
import os
from datetime import datetime
import cv2


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def get_timestamp_str() -> str:
    """Return timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_frame(frame, folder_path: str) -> str:
    """
    Save the current frame to the given folder with a timestamp-based filename.
    Returns the full path to the saved file.
    """
    ensure_dir(folder_path)
    filename = f"frame_{get_timestamp_str()}.jpg"
    filepath = os.path.join(folder_path, filename)
    cv2.imwrite(filepath, frame)
    print(f"[INFO] Saved frame to: {filepath}")
    return filepath


def draw_labelled_box(
    frame,
    bbox,
    text: str,
    color=(0, 255, 0),
    thickness: int = 2,
) -> None:
    """
    Draw a bounding box and a label above it.
    bbox: (x, y, w, h)
    """
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    # Background for text
    text_scale = 0.5
    text_thickness = 1
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
    )
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
    cv2.putText(
        frame,
        text,
        (x + 2, y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (0, 0, 0),
        text_thickness,
        cv2.LINE_AA,
    )
