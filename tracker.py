"""
Minimal centroid-based multi-object tracker, with line-crossing detection.

Bags falling off a nozzle move fast and are only in frame briefly, so
instead of requiring a track to live for many frames before counting
(which works for slow-moving items), we count the instant a tracked
blob's centroid crosses a virtual line -- the standard technique for
industrial item counters on a chute/conveyor.
"""
from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker:
    def __init__(self, max_disappeared=10):
        self.next_id = 0
        self.objects = OrderedDict()       # id -> centroid (x, y)
        self.prev_objects = OrderedDict()  # id -> previous centroid (x, y), for crossing checks
        self.disappeared = OrderedDict()   # id -> frames missing
        self.age = OrderedDict()           # id -> frames alive (tracked continuously)
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.prev_objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.age[self.next_id] = 0
        self.next_id += 1
        return self.next_id - 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.prev_objects[object_id]
        del self.disappeared[object_id]
        del self.age[object_id]

    def update(self, rects):
        """rects: list of (x, y, w, h) bounding boxes detected this frame."""
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects, self.age

        input_centroids = np.array([(x + w // 2, y + h // 2) for (x, y, w, h) in rects])

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(tuple(c))
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.prev_objects[object_id] = self.objects[object_id]
                self.objects[object_id] = tuple(input_centroids[col])
                self.disappeared[object_id] = 0
                self.age[object_id] += 1
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(tuple(input_centroids[col]))

        return self.objects, self.age

    def check_line_crossings(self, line_position_px, axis="horizontal"):
        """
        Returns a list of object_ids that crossed the counting line THIS
        update (i.e. prev centroid was on one side, current centroid is
        on the other side). Call this right after update().
        """
        crossed = []
        for object_id, current in self.objects.items():
            previous = self.prev_objects.get(object_id, current)
            if axis == "horizontal":
                prev_val, curr_val = previous[1], current[1]
            else:
                prev_val, curr_val = previous[0], current[0]

            if prev_val < line_position_px <= curr_val or prev_val > line_position_px >= curr_val:
                crossed.append(object_id)
        return crossed
