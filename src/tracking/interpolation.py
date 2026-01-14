import math
from threading import Lock
import time
from typing import Dict, Generator, List, Tuple, Union
import uuid

import cv2
import numpy as np
from skimage.transform import AffineTransform, warp
from scipy.ndimage import distance_transform_edt

import supervisely as sly
from supervisely.api.entity_annotation.figure_api import FigureInfo
from supervisely.api.module_api import ApiField
from supervisely.api.video.video_api import VideoInfo
from supervisely import logger
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.point_location import PointLocation, points_to_row_col_list

import src.utils as utils


INTERPOLATION_FRAMES_LIMIT = 200
MIN_GEOMETRIES_BATCH_SIZE = 10


active_interpolations = {}


def _morph_masks_gen(mask1, mask2, N):
    """
    Morphs mask1 into mask2 over N iterations by cropping to minimal bounding boxes,
    performing morphing on cropped masks, and then placing the result back into the full mask.

    Parameters:
    - mask1: numpy.ndarray, initial binary mask.
    - mask2: numpy.ndarray, target binary mask.
    - N: int, number of intermediate masks.

    Returns:
    - inner_masks: list of numpy.ndarray, intermediate binary masks of the same size as the input masks.
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    # Compute bounding boxes
    def get_bbox(mask):
        coords = np.column_stack(np.nonzero(mask))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return np.array([x_min, y_min, x_max, y_max])

    bbox1 = get_bbox(mask1)
    bbox2 = get_bbox(mask2)

    # Determine the combined bounding box over all iterations
    x_min_all = min(bbox1[0], bbox2[0])
    y_min_all = min(bbox1[1], bbox2[1])
    x_max_all = max(bbox1[2], bbox2[2])
    y_max_all = max(bbox1[3], bbox2[3])

    # Expand the bounding box by a margin to account for transformations
    margin = 5  # Adjust as needed
    x_min_all = max(0, x_min_all - margin)
    y_min_all = max(0, y_min_all - margin)
    x_max_all = min(mask1.shape[1], x_max_all + margin)
    y_max_all = min(mask1.shape[0], y_max_all + margin)

    # Crop masks
    mask1_cropped = mask1[y_min_all:y_max_all, x_min_all:x_max_all]
    mask2_cropped = mask2[y_min_all:y_max_all, x_min_all:x_max_all]

    # Adjust bounding boxes to the cropped coordinate system
    bbox1_cropped = bbox1 - np.array([x_min_all, y_min_all, x_min_all, y_min_all])
    bbox2_cropped = bbox2 - np.array([x_min_all, y_min_all, x_min_all, y_min_all])

    for n in range(1, N + 1):
        t = n / (N + 1)
        # Interpolate bounding boxes
        bbox_n = (1 - t) * bbox1_cropped + t * bbox2_cropped

        # Compute transformations for mask1 and mask2 to align to bbox_n
        transform1 = _get_affine_transform(bbox1_cropped, bbox_n)
        transform2 = _get_affine_transform(bbox2_cropped, bbox_n)

        # Warp masks to the interpolated bounding box
        mask1_n = warp(
            mask1_cropped,
            inverse_map=transform1.inverse,
            output_shape=mask1_cropped.shape,
            order=0,
            preserve_range=True,
        )
        mask2_n = warp(
            mask2_cropped,
            inverse_map=transform2.inverse,
            output_shape=mask2_cropped.shape,
            order=0,
            preserve_range=True,
        )

        mask1_n = mask1_n > 0.5
        mask2_n = mask2_n > 0.5

        # Compute SDFs
        sdf1 = distance_transform_edt(~mask1_n) - distance_transform_edt(mask1_n)
        sdf2 = distance_transform_edt(~mask2_n) - distance_transform_edt(mask2_n)

        # Interpolate SDFs
        sdf_n = (1 - t) * sdf1 + t * sdf2
        mask_n_cropped = sdf_n < 0

        # Place the cropped mask back into the full-sized mask
        mask_n_full = np.zeros_like(mask1, dtype=np.uint8)
        mask_n_full[y_min_all:y_max_all, x_min_all:x_max_all] = mask_n_cropped.astype(np.uint8)
        yield mask_n_full


def _get_affine_transform(bbox_from, bbox_to):
    """
    Computes an affine transformation matrix that maps bbox_from to bbox_to.

    Parameters:
    - bbox_from: array-like, [x_min, y_min, x_max, y_max] of the source bounding box.
    - bbox_to: array-like, [x_min, y_min, x_max, y_max] of the target bounding box.

    Returns:
    - transform: skimage.transform.AffineTransform object
    """
    src = np.array(
        [
            [bbox_from[0], bbox_from[1]],  # Top-left
            [bbox_from[2], bbox_from[1]],  # Top-right
            [bbox_from[2], bbox_from[3]],  # Bottom-right
        ]
    )
    dst = np.array(
        [
            [bbox_to[0], bbox_to[1]],  # Top-left
            [bbox_to[2], bbox_to[1]],  # Top-right
            [bbox_to[2], bbox_to[3]],  # Bottom-right
        ]
    )
    # Compute affine transformation
    transform = AffineTransform()
    transform.estimate(src, dst)
    return transform


def _simplify_polygon(polygon: sly.Polygon, epsilon: float = 5) -> sly.Polygon:
    exterior = cv2.approxPolyDP(
        polygon.exterior_np.reshape((-1, 1, 2)), epsilon=epsilon, closed=True
    ).reshape(-1, 2)
    interior = []
    if polygon.interior:
        interior = [
            cv2.approxPolyDP(points.reshape((-1, 1, 2)), epsilon=epsilon, closed=True).reshape(
                -1, 2
            )
            for points in polygon.interior_np
        ]
    return sly.Polygon(
        exterior=[sly.PointLocation(x, y) for x, y in exterior],
        interior=[sly.PointLocation(x, y) for x, y in interior],
    )


def unsupported_geometry_interpolator(this_geom: Geometry, **kwargs):
    msg = f"Interpolation for {this_geom.name()} is not supported."
    raise NotImplementedError(msg)


def _fix_unbound(box: Union[sly.Rectangle, sly.OrientedBBox], frame_hw: Tuple[int, int]) -> sly.Rectangle:
    if isinstance(box, sly.OrientedBBox):
        return sly.OrientedBBox(
            max(0, box.top),
            max(0, box.left),
            min(frame_hw[0], box.bottom),
            min(frame_hw[1], box.right),
            box.angle,
        )
    return sly.Rectangle(
        max(0, box.top), max(0, box.left), min(frame_hw[0], box.bottom), min(frame_hw[1], box.right)
    )


def interpolate_box(
    this_geom: sly.Rectangle,
    dest_geom: sly.Rectangle,
    frames_n: int,
    video_info: VideoInfo,
) -> List[sly.Rectangle]:
    logger.debug("Interpolating box")
    rowdelta = (dest_geom.height - this_geom.height) / (frames_n + 1)
    coldelta = (dest_geom.width - this_geom.width) / (frames_n + 1)
    rowshift = (dest_geom.center.row - this_geom.center.row) / (frames_n + 1)
    colshift = (dest_geom.center.col - this_geom.center.col) / (frames_n + 1)
    created_geometries: List[sly.AnyGeometry] = []
    for i in range(1, frames_n + 1):
        resized: sly.Rectangle = this_geom.resize(
            in_size=(video_info.frame_height, video_info.frame_width),
            out_size=(
                int(video_info.frame_height * (1 + rowdelta * i / this_geom.height)),
                int(video_info.frame_width * (1 + coldelta * i / this_geom.width)),
            ),
        )
        target = int(this_geom.center.row + i * rowshift), int(this_geom.center.col + i * colshift)
        moved: sly.Rectangle = resized.translate(
            target[0] - resized.center.row, target[1] - resized.center.col
        )
        moved = _fix_unbound(moved, (video_info.frame_height, video_info.frame_width))
        created_geometries.append(moved)
    logger.debug("Done interpolating box")
    return created_geometries


def interpolate_bitmap(
    this_geom: sly.Bitmap,
    dest_geom: sly.Bitmap,
    frames_n: int,
    video_info: VideoInfo,
) -> Generator[sly.Bitmap, None, None]:
    logger.debug("Interpolating bitmap")
    this_mask = this_geom.get_mask((video_info.frame_height, video_info.frame_width))
    next_mask = dest_geom.get_mask((video_info.frame_height, video_info.frame_width))
    for mask in _morph_masks_gen(this_mask, next_mask, frames_n):
        yield sly.Bitmap(mask)
    logger.debug("Done interpolating bitmap")


def interpolate_polygon(
    this_geom: sly.Polygon,
    dest_geom: sly.Polygon,
    frames_n: int,
    video_info: VideoInfo,
) -> Generator[List[List[sly.Polygon]], None, None]:
    logger.debug("Interpolating polygon")
    this_mask = this_geom.get_mask((video_info.frame_height, video_info.frame_width))
    next_mask = dest_geom.get_mask((video_info.frame_height, video_info.frame_width))
    for mask in _morph_masks_gen(this_mask, next_mask, frames_n):
        polys = sly.Bitmap(mask).to_contours()
        polys = [_simplify_polygon(poly) for poly in polys]
        yield polys
    logger.debug("Done interpolating polygon")


def interpolate_line(
    this_geom: sly.Polyline,
    dest_geom: sly.Polyline,
    frames_n: int,
    video_info: VideoInfo,
) -> List[sly.Polyline]:
    logger.debug("Interpolating line")
    created_geometries: List[sly.Polyline] = []
    if len(this_geom.exterior) != len(dest_geom.exterior):
        raise ValueError("Cannot interpolate lines with different number of points")
    for i in range(1, frames_n + 1):
        t = i / (frames_n + 1)
        points = []
        for p1, p2 in zip(this_geom.exterior, dest_geom.exterior):
            x = int(p1.row * (1 - t) + p2.row * t)
            y = int(p1.col * (1 - t) + p2.col * t)
            points.append(sly.PointLocation(x, y))
        created_geometries.append(sly.Polyline(exterior=points))
    logger.debug("Done interpolating line")
    return created_geometries


def interpolate_point(
    this_geom: sly.Point,
    dest_geom: sly.Point,
    frames_n: int,
    video_info: VideoInfo,
) -> List[sly.Point]:
    logger.debug("Interpolating point")
    created_geometries: List[sly.Point] = []
    for i in range(1, frames_n + 1):
        t = i / (frames_n + 1)
        x = int(this_geom.row * (1 - t) + dest_geom.row * t)
        y = int(this_geom.col * (1 - t) + dest_geom.col * t)
        created_geometries.append(sly.Point(row=x, col=y))
    logger.debug("Done interpolating point")
    return created_geometries

def interpolate_oriented_bbox(
    this_geom: sly.OrientedBBox,
    dest_geom: sly.OrientedBBox,
    frames_n: int,
    video_info: VideoInfo,
) -> List[sly.OrientedBBox]:
    logger.debug("Interpolating oriented bbox")
    this_geom = normalize_oriented_bbox(this_geom)
    dest_geom = normalize_oriented_bbox(dest_geom)
    rowdelta = (dest_geom.height - this_geom.height) / (frames_n + 1)
    coldelta = (dest_geom.width - this_geom.width) / (frames_n + 1)
    rowshift = (dest_geom.center.row - this_geom.center.row) / (frames_n + 1)
    colshift = (dest_geom.center.col - this_geom.center.col) / (frames_n + 1)
    start_angle = this_geom.angle
    end_angle = dest_geom.angle
    angle_diff = end_angle - start_angle
    angle_diff = (angle_diff + math.pi) % (2*math.pi) - math.pi
    angle_delta = angle_diff / (frames_n + 1)
    created_geometries: List[sly.AnyGeometry] = []
    for i in range(1, frames_n + 1):
        resized: sly.OrientedBBox = this_geom.resize(
            in_size=(video_info.frame_height, video_info.frame_width),
            out_size=(
                int(video_info.frame_height * (1 + rowdelta * i / this_geom.height)),
                int(video_info.frame_width * (1 + coldelta * i / this_geom.width)),
            ),
        )
        target = int(this_geom.center.row + i * rowshift), int(this_geom.center.col + i * colshift)
        moved: sly.OrientedBBox = resized.translate(
            target[0] - resized.center.row, target[1] - resized.center.col
        )
        moved = _fix_unbound(moved, (video_info.frame_height, video_info.frame_width))
        this_angle = (start_angle + i * angle_delta + math.pi) % (2*math.pi) - math.pi
        rotated = sly.OrientedBBox(moved.top, moved.left, moved.bottom, moved.right, this_angle)
        created_geometries.append(rotated)
    logger.debug("Done interpolating oriented bbox")
    return created_geometries


INTERPOLATORS = {
    sly.Rectangle.name(): interpolate_box,
    sly.Bitmap.name(): interpolate_bitmap,
    sly.Polygon.name(): interpolate_polygon,
    sly.Polyline.name(): interpolate_line,
    sly.Point.name(): interpolate_point,
    sly.OrientedBBox.name(): interpolate_oriented_bbox,
}


def interpolate_box_next(this_geom: sly.Rectangle, prev_geom: sly.Rectangle, frames_n: int, video_info: VideoInfo, frames_count: int) -> List[sly.Rectangle]:
    logger.debug("Interpolating box")
    rowdelta = (this_geom.height - prev_geom.height) / frames_n
    coldelta = (this_geom.width - prev_geom.width) / frames_n
    rowshift = (this_geom.center.row - prev_geom.center.row) / frames_n
    colshift = (this_geom.center.col - prev_geom.center.col) / frames_n
    created_geometries: List[sly.AnyGeometry] = []
    for i in range(1, frames_count+1):
        new = sly.Rectangle(
            top=int(this_geom.top - i * rowdelta / 2 + i * rowshift),
            left=int(this_geom.left - i * coldelta / 2 + i * colshift),
            bottom=int(this_geom.bottom + i * rowdelta / 2 + i * rowshift),
            right=int(this_geom.right + i * coldelta / 2 + i * colshift),
        )
        new = _fix_unbound(new, (video_info.frame_height, video_info.frame_width))
        created_geometries.append(new)
    logger.debug("Done interpolating box")
    return created_geometries


def interpolate_line_next(this_geom: sly.Polyline, prev_geom: sly.Polyline, frames_n: int, video_info: VideoInfo, frames_count: int) -> sly.Polyline:
    logger.debug("Interpolating line")
    created_geometries: List[sly.Polyline] = []
    if len(this_geom.exterior) != len(prev_geom.exterior):
        raise ValueError("Cannot interpolate lines with different number of points")
    for i in range(1, frames_count + 1):
        points = []
        for p1, p2 in zip(prev_geom.exterior, this_geom.exterior):
            delta = ((p2.row - p1.row) / frames_n, (p2.col - p1.col) / frames_n)
            x = int(p2.row + i * delta[0])
            y = int(p2.col + i * delta[1])
            points.append(sly.PointLocation(x, y))
        created_geometries.append(sly.Polyline(exterior=points))
    logger.debug("Done interpolating line")
    return created_geometries

def interpolate_point_next(this_geom: sly.Point, prev_geom: sly.Point, frames_n: int, video_info: VideoInfo, frames_count: int) -> List[sly.Point]:
    logger.debug("Interpolating point")
    created_geometries: List[sly.Point] = []
    delta = ((this_geom.row - prev_geom.row) / frames_n, (this_geom.col - prev_geom.col) / frames_n)
    for i in range(1, frames_count + 1):
        x = int(this_geom.row + i * delta[0])
        y = int(this_geom.col + i * delta[1])
        created_geometries.append(sly.Point(row=x, col=y))
    logger.debug("Done interpolating point")
    return created_geometries

def normalize_oriented_bbox(geom: sly.OrientedBBox):
    angle = geom.angle
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    top, left, bottom, right = geom.top, geom.left, geom.bottom, geom.right

    if math.pi / 4 <= angle < 3 * math.pi / 4:
        center_row = (top + bottom) / 2
        center_col = (left + right) / 2
        width = bottom - top
        height = right - left
        top = int(center_row - height / 2)
        bottom = int(center_row + height / 2)
        left = int(center_col - width / 2)
        right = int(center_col + width / 2)
        angle -= math.pi / 2
    
    elif angle >= 3 * math.pi / 4 or angle < -3 * math.pi / 4:
        angle = angle - math.pi if angle >= 3 * math.pi / 4 else angle + math.pi
    
    elif -3 * math.pi / 4 <= angle < -math.pi / 4:
        center_row = (top + bottom) / 2
        center_col = (left + right) / 2
        width = bottom - top
        height = right - left
        top = int(center_row - height / 2)
        bottom = int(center_row + height / 2)
        left = int(center_col - width / 2)
        right = int(center_col + width / 2)
        angle += math.pi / 2
    
    return sly.OrientedBBox(top, left, bottom, right, angle)

def interpolate_oriented_bbox_next(this_geom: sly.OrientedBBox, prev_geom: sly.OrientedBBox, frames_n: int, video_info: VideoInfo, frames_count: int) -> List[sly.OrientedBBox]:
    logger.debug("Interpolating oriented bbox")
    this_geom = normalize_oriented_bbox(this_geom)
    prev_geom = normalize_oriented_bbox(prev_geom)
    rowdelta = (this_geom.height - prev_geom.height) / frames_n
    coldelta = (this_geom.width - prev_geom.width) / frames_n
    rowshift = (this_geom.center.row - prev_geom.center.row) / frames_n
    colshift = (this_geom.center.col - prev_geom.center.col) / frames_n
    start_angle = prev_geom.angle
    end_angle = this_geom.angle
    angle_diff = end_angle - start_angle
    angle_diff = (angle_diff + math.pi) % (2*math.pi) - math.pi
    angle_delta = angle_diff / frames_n
    created_geometries: List[sly.AnyGeometry] = []
    for i in range(1, frames_count + 1):
        top=int(this_geom.top - i * rowdelta / 2 + i * rowshift)
        left=int(this_geom.left - i * coldelta / 2 + i * colshift)
        bottom=int(this_geom.bottom + i * rowdelta / 2 + i * rowshift)
        right=int(this_geom.right + i * coldelta / 2 + i * colshift)
        this_angle = (end_angle + i * angle_delta + math.pi) % (2*math.pi) - math.pi
        if top > bottom or left > right:
            logger.debug("The box has shrinked to zero", extra={
                "tlbr": [top, left, bottom, right],
                "angle": this_angle,
                "deltas": [rowdelta, coldelta]
            })
            created_geometries.extend([None]*(frames_count+1-i))
            break
        new = sly.Rectangle(
            top=top,
            left=left,
            bottom=bottom,
            right=right
        )
        new = _fix_unbound(new, (video_info.frame_height, video_info.frame_width))
        rotated = sly.OrientedBBox(new.top, new.left, new.bottom, new.right, this_angle)
        created_geometries.append(rotated)
    logger.debug("Done interpolating oriented bbox")
    return created_geometries

NEXT_INTERPOLATORS = {
    sly.Rectangle.name(): interpolate_box_next,
    sly.Polyline.name(): interpolate_line_next,
    sly.Point.name(): interpolate_point_next,
    sly.OrientedBBox.name(): interpolate_oriented_bbox_next,
}


def is_geom_type_supported(figure_info: FigureInfo):
    return figure_info.geometry_type in INTERPOLATORS


def flatten_list(l: List):
    flattened_list = []
    for item in l:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


class Interpolator:
    INTERPOLATION_FRAMES_LIMIT = 200

    def __init__(self, api: sly.Api, context: Dict):
        self.api = api
        self.context = context

        self.video_id = context["videoId"]
        self.figure_ids = context["figureIds"]
        self.track_id = context.get("trackId", None)
        self.log_extra = {
            "video_id": self.video_id,
            "figure_ids": self.figure_ids,
            "track_id": self.track_id,
        }
        self.video_info = api.video.get_info_by_id(self.video_id)
        self.dataset_id = self.video_info.dataset_id
        self.figures = api.video.figure.get_by_ids(self.dataset_id, self.figure_ids)
        self.frame_start = min([figure.frame_index for figure in self.figures])
        self.frame_end = self.frame_start

        self.progress_total = 1
        self.progress_current = 0

    def notify_progress(self):
        try:
            self.api.video.notify_progress(
                self.track_id,
                self.video_id,
                frame_start=self.frame_start,
                frame_end=self.frame_end,
                current=self.progress_current,
                total=self.progress_total,
            )
        except Exception:
            logger.warning("Unable to notify video tool", exc_info=True)

    def send_warning(self, message):
        utils.notify_warning(self.api, self.track_id, self.video_id, message)

    def filter_figures(
        self, figures, dest_figures, condition
    ) -> Tuple[List[FigureInfo], List[FigureInfo]]:
        filtered_figures = []
        filtered_dest_figures = []
        for figure, dest_figure in zip(figures, dest_figures):
            if condition((figure, dest_figure)):
                filtered_figures.append(figure)
                filtered_dest_figures.append(dest_figure)
        return filtered_figures, filtered_dest_figures

    def find_destination_figures(self) -> List[Tuple[FigureInfo, FigureInfo]]:
        all_figures: List[FigureInfo] = self.api.video.figure.get_list(
            dataset_id=self.dataset_id,
            filters=[
                {
                    "field": "objectId",
                    "operator": "in",
                    "value": list(set([figure.object_id for figure in self.figures])),
                },
                {
                    "field": "startFrame",
                    "operator": ">=",
                    "value": max(0, self.frame_start - self.INTERPOLATION_FRAMES_LIMIT),
                },
                {
                    "field": "endFrame",
                    "operator": "<=",
                    "value": min(
                        self.frame_end + self.INTERPOLATION_FRAMES_LIMIT,
                        self.video_info.frames_count - 1,
                    ),
                },
            ],
        )
        per_object_figures = {}
        for figure in all_figures:
            per_object_figures.setdefault(figure.object_id, []).append(figure)
        destination_figures = []
        for this_figure in self.figures:
            object_id = this_figure.object_id
            left = None
            right = None
            for figure in per_object_figures.get(object_id, []):
                if figure.frame_index < this_figure.frame_index:
                    if left is None or left.frame_index < figure.frame_index:
                        left = figure
                if figure.frame_index > this_figure.frame_index:
                    if right is None or right.frame_index > figure.frame_index:
                        right = figure
            if left is not None and this_figure.frame_index - left.frame_index < 2:
                left = None
            if right is not None and right.frame_index - this_figure.frame_index < 2:
                right = None
            destination_figures.append((left, right))
        return destination_figures

    def filter_destination_figures(
        self, dest_figures: List[Tuple[FigureInfo, FigureInfo]]
    ) -> Tuple[List[FigureInfo], List[FigureInfo]]:
        figures, dest_figures = self.filter_figures(
            self.figures,
            dest_figures,
            condition=lambda x: x[1][0] is not None or x[1][1] is not None,
        )
        if len(figures) < len(self.figures):
            msg = f"Destination figures not found for {len(self.figures) - len(figures)} figures. Such figures will be skipped"
            logger.info(msg, extra=self.log_extra)
            self.send_warning(msg)

        original_figures_len = len(figures)
        figures, dest_figures = self.filter_figures(
            figures, dest_figures, condition=lambda x: is_geom_type_supported(x[0])
        )
        if len(figures) < original_figures_len:
            msg = "Some geometries are not supported yet. Such figures will be skipped"
            logger.info(msg, extra=self.log_extra)
            self.send_warning(msg)

        return figures, dest_figures

    def _upload(self, object_id: int, batch: List, frame_indexes: List):
        figures_json = []
        for frame_index, geoms in zip(frame_indexes, batch):
            for geom in flatten_list([geoms]):
                figures_json.append(
                    {
                        ApiField.OBJECT_ID: object_id,
                        ApiField.GEOMETRY_TYPE: geom.geometry_name(),
                        ApiField.GEOMETRY: geom.to_json(),
                        ApiField.META: {ApiField.FRAME: frame_index},
                        ApiField.TRACK_ID: self.track_id,
                    }
                )
        figures_keys = [uuid.uuid4() for _ in figures_json]
        key_id_map = sly.KeyIdMap()
        # pylint: disable=protected-access
        self.api.video.figure._append_bulk(
            entity_id=self.video_id,
            figures_json=figures_json,
            figures_keys=figures_keys,
            key_id_map=key_id_map,
        )

    def upload(self, object_id, batch: List, frame_indexes: List):
        try:
            self._upload(object_id, batch, frame_indexes)
        except Exception as e:
            message = f"Unable to upload interpolations: {str(e)}"
            logger.warning(message, exc_info=True, extra=self.log_extra)
            self.send_warning(message)

    def interpolate_direction(self, this_figure, this_geometry, dest_figure):
        if dest_figure is None:
            return
        dest_geometry = sly.deserialize_geometry(dest_figure.geometry_type, dest_figure.geometry)

        direction = 1 if dest_figure.frame_index >= this_figure.frame_index else -1
        frames_n = abs(dest_figure.frame_index - this_figure.frame_index) - 1
        before_this_figure_progress = self.progress_current

        if this_geometry.name() != dest_geometry.name():
            msg = f"Cannot interpolate between {this_geometry.name()} and {dest_geometry.name()}"
            logger.info(msg, extra=self.log_extra)
            self.send_warning(msg)
            self.progress_current += frames_n
            return

        interpolator_func = INTERPOLATORS.get(
            this_geometry.name(), unsupported_geometry_interpolator
        )
        try:
            frame_index = this_figure.frame_index + direction
            cum_batch = []
            for batch in interpolator_func(
                this_geometry,
                dest_geometry,
                frames_n=frames_n,
                video_info=self.video_info,
            ):
                cum_batch.append(batch)
                if len(cum_batch) < MIN_GEOMETRIES_BATCH_SIZE:
                    continue
                frame_indexes = list(
                    range(frame_index, frame_index + len(cum_batch) * direction, direction)
                )
                frame_index = frame_indexes[-1] + direction
                self.upload(this_figure.object_id, cum_batch, frame_indexes)
                self.progress_current += len(cum_batch)
                cum_batch = []
                self.notify_progress()
            if len(cum_batch) > 0:
                frame_indexes = list(
                    range(frame_index, frame_index + len(cum_batch) * direction, direction)
                )
                self.upload(this_figure.object_id, cum_batch, frame_indexes)
                self.progress_current += len(cum_batch)
                self.notify_progress()
        except NotImplementedError:
            msg = f"Interpolation for geometry {this_geometry.name()} is not supported yet. Skipping figure"
            logger.info(msg, extra=self.log_extra)
            self.send_warning(msg)
            self.progress_current += frames_n
        except Exception as e:
            if "Cannot interpolate lines with different number of points" in str(e):
                message = "Cannot interpolate lines with different number of points"
                logger.warning(message)
                self.send_warning(message)
                return
            msg = f"Unexpected Error during interpolation: {str(e)}"
            logger.warning(msg, exc_info=True, extra=self.log_extra)
            self.send_warning(msg)
            this_figure_progress = before_this_figure_progress - self.progress_current
            self.progress_current += frames_n - this_figure_progress
        finally:
            self.notify_progress()

    def interpolate_frames(self):
        self.notify_progress()
        dest_figures: List[Tuple[FigureInfo, FigureInfo]] = self.find_destination_figures()
        figures, dest_figures = self.filter_destination_figures(dest_figures)
        if len(figures) == 0:
            msg = "No valid figures to interpolate"
            logger.info(msg, extra=self.log_extra)
            utils.notify_error(self.api, self.track_id, self.video_id, msg)
            return

        self.frame_start = min([figure.frame_index for figure in figures])
        self.frame_end = self.frame_start
        self.progress_total = 0
        for this_figure, left_right in zip(figures, dest_figures):
            left_figure, right_figure = left_right
            this_from = this_figure.frame_index
            this_to = this_figure.frame_index
            if left_figure is not None:
                self.frame_start = min(self.frame_start, left_figure.frame_index + 1)
                this_from = left_figure.frame_index
            if right_figure is not None:
                self.frame_end = max(self.frame_end, right_figure.frame_index - 1)
                this_to = right_figure.frame_index
            self.progress_total += max(0, this_to - this_from - 1)
        self.notify_progress()

        for this_figure, dest_figures in zip(figures, dest_figures):
            left_dest_figure, right_dest_figure = dest_figures
            this_geometry = sly.deserialize_geometry(
                this_figure.geometry_type, this_figure.geometry
            )

            # left
            if left_dest_figure is not None:
                self.interpolate_direction(this_figure, this_geometry, left_dest_figure)

            # right
            if right_dest_figure is not None:
                self.interpolate_direction(this_figure, this_geometry, right_dest_figure)

        self.progress_current = self.progress_total
        self.notify_progress()


@utils.send_error_data
def interpolate_frames(api, context):
    track_id = context.get("trackId", None)
    if track_id not in active_interpolations:
        active_interpolations[track_id] = Lock()
    if active_interpolations[track_id].locked():
        logger.info(
            "Interpolation is already running. Waiting for it to finish",
            extra={"track_id": track_id},
        )
    with active_interpolations[track_id]:
        Interpolator(api, context).interpolate_frames()

def interpolate_next(api: sly.Api, video_info: VideoInfo, frame_index: int, figures: List[FigureInfo], frames_count: int) -> List[List[Union[Geometry, None]]]:
    MAX_PREVIOUS_FRAMES = 5
    dataset_id = video_info.dataset_id
    all_figures: List[FigureInfo] = api.video.figure.get_list(
        dataset_id=dataset_id,
        filters=[
            {
                "field": "objectId",
                "operator": "in",
                "value": list(set([figure.object_id for figure in figures])),
            },
            {
                "field": "startFrame",
                "operator": ">=",
                "value": max(0, frame_index - MAX_PREVIOUS_FRAMES),
            },
            {
                "field": "endFrame",
                "operator": "<=",
                "value": max(0, frame_index-1),
            },
        ],
    )
    per_object_figures: Dict[int, List[FigureInfo]] = {}
    for figure in all_figures:
        per_object_figures.setdefault(figure.object_id, []).append(figure)
    previous_figures: Dict[int, FigureInfo] = {}
    for this_figure in figures:
        object_id = this_figure.object_id
        left = None
        for figure in per_object_figures.get(object_id, []):
            if figure.geometry_type == this_figure.geometry_type and figure.frame_index <= this_figure.frame_index and (left is None or left.frame_index < figure.frame_index):
                left = figure
        previous_figures[this_figure.id] = left
    figure_predictions = []
    for this_figure in figures:
        left_figure = previous_figures.get(this_figure.id, None)
        this_geometry = sly.deserialize_geometry(this_figure.geometry_type, this_figure.geometry)
        if left_figure is None:
            logger.info(f"No previous figure found, cloining instead of interpolating for figure {this_figure.id}", extra={"figure_id": this_figure.id})
            figure_predictions.append([this_geometry for _ in range(frames_count)])
            continue
        left_geometry = sly.deserialize_geometry(left_figure.geometry_type, left_figure.geometry)
        frames_n = this_figure.frame_index - left_figure.frame_index
        if frames_n < 0:
            logger.debug(f"previous figure is after the target during interpolation")
            figure_predictions.append(None)
            continue
        if frames_n == 0:
            logger.debug("No previous figure found for interpolation, cloning instead")
            figure_predictions.append([this_geometry for _ in range(frames_count)])
            continue
        interpolator_func = NEXT_INTERPOLATORS.get(
            this_geometry.name(), unsupported_geometry_interpolator
        )
        try:
            figure_predictions.append(interpolator_func(this_geometry, left_geometry, frames_n, video_info, frames_count))
        except:
            logger.error("Error during interpolation", exc_info=True)
            figure_predictions.append(None)
    frames_predictions = [[] for _ in range(frames_count)]
    for fig_pred in figure_predictions:
        for frame_offset in range(frames_count):
            if fig_pred is None:
                frames_predictions[frame_offset].append(None)
            else:
                frames_predictions[frame_offset].append(fig_pred[frame_offset])
    return frames_predictions
        

