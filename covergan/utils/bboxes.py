from math import sqrt
import operator
from typing import Callable

import scipy.ndimage as ndimage
import scipy.spatial as spatial


class BBox(object):
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        """
        (x1, y1) is the upper left corner.
        (x2, y2) is the lower right corner.
        """
        self.x1 = min(x1, x2)
        self.x2 = max(x1, x2)
        self.y1 = min(y1, y2)
        self.y2 = max(y1, y2)

    def ul(self) -> (int, int):
        return self.x1, self.y1

    def lr(self) -> (int, int):
        return self.x2, self.y2

    def width(self) -> int:
        return self.x2 - self.x1

    def height(self) -> int:
        return self.y2 - self.y1

    def wh_ratio(self) -> float:
        return self.width() / self.height()

    def area(self) -> int:
        return self.width() * self.height()

    def taxicab_diagonal(self) -> int:
        # Taxicab distance from (x1, y1) to (x2, y2)
        return self.width() + self.height()

    def min_coord_dist(self, other) -> float:
        dx = min(abs(self.x1 - other.x1), abs(self.x1 - other.x2), abs(self.x2 - other.x1), abs(self.x2 - other.x2))
        dy = min(abs(self.y1 - other.y1), abs(self.y1 - other.y2), abs(self.y2 - other.y1), abs(self.y2 - other.y2))
        return sqrt(dx ** 2 + dy ** 2)

    def overlaps(self, other) -> bool:
        return not ((self.x1 > other.x2)
                    or (self.x2 < other.x1)
                    or (self.y1 > other.y2)
                    or (self.y2 < other.y1))

    def to_pos(self, canvas_size: int) -> (float, float, float, float):
        return (
            self.x1 / canvas_size,
            self.y1 / canvas_size,
            self.width() / canvas_size,
            self.height() / canvas_size
        )

    def split_horizontal(self, ratio: float):
        w1 = round(self.width() * ratio)
        return BBox(self.x1, self.y1, self.x1 + w1, self.y2), BBox(self.x1 + w1 + 1, self.y1, self.x2, self.y2)

    def split_vertical(self, ratio: float):
        h1 = round(self.height() * ratio)
        return BBox(self.x1, self.y1, self.x2, self.y1 + h1), BBox(self.x1, self.y1 + h1 + 1, self.x2, self.y2)

    def recanvas(self, cur_canvas: int, new_canvas: int):
        """
        Rescale coordinates to a different canvas size
        """
        r = new_canvas / cur_canvas
        return BBox(int(self.x1 * r), int(self.y1 * r), int(self.x2 * r), int(self.y2 * r))

    def __str__(self):
        return f"BBox ({self.x1}, {self.y1})-({self.x2}, {self.y2}): w={self.width()}, h={self.height()}"

    def __eq__(self, other) -> bool:
        return (self.x1 == other.x1
                and self.y1 == other.y1
                and self.x2 == other.x2
                and self.y2 == other.y2)

    def __hash__(self):
        return hash((self.x1, self.y1, self.x2, self.y2))


def find_bboxes(img):
    filled = ndimage.morphology.binary_fill_holes(img)
    coded_blobs, num_blobs = ndimage.label(filled)
    data_slices = ndimage.find_objects(coded_blobs)

    bounding_boxes = []
    for s in data_slices:
        dy, dx = s[:2]
        bounding_boxes.append(BBox(dx.start, dy.start, dx.stop, dy.stop))

    return bounding_boxes


def remove_overlaps(bboxes: [BBox]) -> [BBox]:
    """
    Replace overlapping bboxes with the minimal BBox that contains both.
    """
    if not bboxes:
        return []

    corners = []
    ul_corners = [b.ul() for b in bboxes]
    bbox_map = {}  # Corners -> bboxes

    for bbox in bboxes:
        for c in (bbox.ul(), bbox.lr()):
            corners.append(c)
            bbox_map[c] = bbox

    tree = spatial.KDTree(corners)  # Quick nearest-neighbor lookup
    for c in ul_corners:
        bbox = bbox_map[c]
        # Find all points within a taxicab distance of the corner
        indices = tree.query_ball_point(c, bbox_map[c].taxicab_diagonal(), p=1)
        for near_corner in tree.data[indices]:
            near_bbox = bbox_map[tuple(near_corner)]
            if bbox != near_bbox and bbox.overlaps(near_bbox):
                # Expand both bboxes
                bbox.x1 = near_bbox.x1 = min(bbox.x1, near_bbox.x1)
                bbox.y1 = near_bbox.y1 = min(bbox.y1, near_bbox.y1)
                bbox.x2 = near_bbox.x2 = max(bbox.x2, near_bbox.x2)
                bbox.y2 = near_bbox.y2 = max(bbox.y2, near_bbox.y2)
    return list(set(bbox_map.values()))


def filter_bboxes_by_size(bboxes, threshold):
    return list(filter(lambda x: x.width() > threshold and x.height() > threshold, bboxes))


def merge_bboxes(bboxes: [BBox]) -> BBox:
    assert bboxes
    if len(bboxes) == 1:
        return bboxes[0]

    x1 = min(b.x1 for b in bboxes)
    y1 = min(b.y1 for b in bboxes)
    x2 = max(b.x2 for b in bboxes)
    y2 = max(b.y2 for b in bboxes)

    return BBox(x1 - 1, y1 - 1, x2, y2)


def merge_aligned_bboxes(bboxes: [BBox], canvas_size: int) -> [BBox]:
    if not bboxes:
        return []

    shift_threshold = 1 / 4
    dim_threshold = 5 / 12
    canvas_threshold = 1 / 5

    params: [(str, Callable[[BBox], int])] = [
        ("y1", "x1", BBox.height, BBox.width),  # top
        ("y2", "x1", BBox.height, BBox.width),  # bottom
        ("x1", "y1", BBox.width, BBox.height),  # left
        ("x2", "y1", BBox.width, BBox.height),  # right
    ]

    for align_param, sort_param, dim_param_f, dist_param_f in params:
        bboxes = sorted(bboxes, key=operator.attrgetter(align_param))
        result = []

        def split_group(cur):
            cur = sorted(cur, key=operator.attrgetter(sort_param))
            subgroup = [cur[0]]
            for x in cur[1:]:
                if x.min_coord_dist(subgroup[-1]) < canvas_size * canvas_threshold:
                    subgroup.append(x)
                else:
                    result.append(merge_bboxes(subgroup))
                    subgroup = [x]
            if subgroup:
                result.append(merge_bboxes(subgroup))

        cur_group = []
        prev_pos = getattr(bboxes[0], align_param)
        prev_dim = dim_param_f(bboxes[0])
        for b in bboxes:
            b_dim = dim_param_f(b)
            max_dim = max(b_dim, prev_dim)
            valid_shift = abs(getattr(b, align_param) - prev_pos) < max_dim * shift_threshold
            valid_dim_change = abs(b_dim - prev_dim) < max_dim * dim_threshold
            valid_canvas_ratio = (not cur_group) or b_dim < canvas_size * canvas_threshold
            if valid_shift and valid_dim_change and valid_canvas_ratio:
                cur_group.append(b)
            else:
                split_group(cur_group)
                cur_group = [b]
            prev_pos = getattr(b, align_param)
            prev_dim = b_dim
        if cur_group:
            split_group(cur_group)
        bboxes = result

    return bboxes


def crop_bboxes_by_canvas(bboxes: [BBox], canvas_size: int):
    for b in bboxes:
        b.x1 = max(b.x1, 0)
        b.y1 = max(b.y1, 0)
        b.x2 = min(b.x2, canvas_size - 1)
        b.y2 = min(b.y2, canvas_size - 1)


def sort_bboxes_by_area(bboxes: [BBox]) -> [BBox]:
    return sorted(bboxes, key=lambda x: x.area())
