from typing import Dict, Tuple, List
import numpy as np


class CentroidTracker:
    """
    Very simple centroid-based tracker.

    - Input each frame: list of bounding boxes (x, y, w, h)
    - Output: dict {object_id: (x, y, w, h)}
    """

    def __init__(self, max_disappeared: int = 30) -> None:
        # next ID to assign
        self.next_object_id: int = 0
        # object_id -> centroid (cx, cy)
        self.centroids: Dict[int, Tuple[int, int]] = {}
        # object_id -> last bbox (x, y, w, h)
        self.boxes: Dict[int, Tuple[int, int, int, int]] = {}
        # object_id -> number of consecutive frames not seen
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared

    def register(self, bbox: Tuple[int, int, int, int]) -> None:
        x, y, w, h = bbox
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        self.centroids[self.next_object_id] = (cx, cy)
        self.boxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        self.centroids.pop(object_id, None)
        self.boxes.pop(object_id, None)
        self.disappeared.pop(object_id, None)

    def update(
        self, rects: List[Tuple[int, int, int, int]]
    ) -> Dict[int, Tuple[int, int, int, int]]:
        """
        rects: list of bounding boxes (x, y, w, h) for current frame.
        Returns mapping: object_id -> bbox.
        """
        # No detections: mark objects disappeared
        if len(rects) == 0:
            to_deregister = []
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    to_deregister.append(object_id)
            for oid in to_deregister:
                self.deregister(oid)
            return dict(self.boxes)

        # If no existing objects, register all rects
        if len(self.centroids) == 0:
            for rect in rects:
                self.register(rect)
            return dict(self.boxes)

        # Compute centroids for new rects
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int(x + w / 2)
            cY = int(y + h / 2)
            input_centroids[i] = (cX, cY)

        object_ids = list(self.centroids.keys())
        object_centroids = list(self.centroids.values())

        # distance matrix between existing centroids and new centroids
        D = np.linalg.norm(
            np.array(object_centroids)[:, np.newaxis, :] - input_centroids[np.newaxis, :, :],
            axis=2,
        )

        # sort rows by min distance
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        # match existing objects to new rects
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            object_id = object_ids[row]
            self.centroids[object_id] = tuple(input_centroids[col])
            self.boxes[object_id] = rects[col]
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        # any unmatched existing objects → disappeared
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # any unmatched new rects → new objects
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unused_cols:
            self.register(rects[col])

        return dict(self.boxes)
