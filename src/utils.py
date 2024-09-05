import ast
import functools
from typing import Dict, List, Tuple, Type

import uuid
import cv2
import numpy as np

import requests
import supervisely as sly
from supervisely.api.entity_annotation.figure_api import FigureInfo
from supervisely import TinyTimer

import src.globals as g


def get_figure_track_id(figure_id: int) -> str:
    return str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name=str(figure_id)))


class SmartToolInput:
    def __init__(self, crop: List, positive: List, negative: List, visible: bool = True):
        positive = [
            [point.col, point.row] if isinstance(point, sly.Point) else point for point in positive
        ]
        negative = [
            [point.col, point.row] if isinstance(point, sly.Point) else point for point in negative
        ]
        if isinstance(crop, sly.Rectangle):
            crop = [[crop.left, crop.top], [crop.right, crop.bottom]]
        self.crop = crop
        self.positive = positive
        self.negative = negative
        self.visible = visible

    def to_json(self):
        return {
            "crop": self.crop,
            "positive": self.positive,
            "negative": self.negative,
            "visible": self.visible,
        }

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            crop=data["crop"],
            positive=data["positive"],
            negative=data["negative"],
            visible=data["visible"],
        )


class Meta:
    def __init__(
        self,
        smi: SmartToolInput = None,
        object_id: int = None,
        priority: int = 1,
        project_id: int = None,
        tags: List[str] = None,
        tool: str = None,
        track_id: str = None,
        updated_at: str = None,
    ):
        self.smi = smi.to_json() if isinstance(smi, SmartToolInput) else smi
        self.object_id = object_id
        self.priority = priority
        self.project_id = project_id
        self.tags = tags if tags is not None else []
        self.tool = tool
        self.track_id = track_id
        self.updated_at = updated_at

    def to_json(self):
        return {
            "smartToolInput": self.smi,
            "object_id": self.object_id,
            "priority": self.priority,
            "project_id": self.project_id,
            "tags": self.tags,
            "tool": self.tool,
            "track_id": self.track_id,
            "updated_at": self.updated_at,
        }


class Prediction:
    def __init__(self, geometry_data: dict, geometry_type: str, meta: Meta = None):
        self.geometry_data = geometry_data
        self.geometry_type = geometry_type
        self.meta = meta

    def to_json(self):
        meta = {}
        if self.meta is not None:
            meta = self.meta.to_json() if isinstance(self.meta, Meta) else self.meta

        return {
            "data": self.geometry_data,
            "type": self.geometry_type,
            "meta": meta,
        }


def smoothen_mask(mask: sly.Bitmap, img_size) -> sly.Bitmap:
    sly.logger.debug("smoothing mask")
    canv: np.ndarray = mask.get_mask(img_size)
    canv = canv.astype("uint8")
    kernel = np.ones((img_size[0] * 5 // 480, img_size[1] * 5 // 480), np.uint8)

    # pylint: disable=no-member
    canv = cv2.morphologyEx(canv, cv2.MORPH_CLOSE, kernel, iterations=3)
    return sly.Bitmap(canv)


def move_points_relative(
    src_rect: sly.Rectangle, points: List[sly.Point], dst_rect: sly.Rectangle
) -> List[sly.Point]:
    res_points = []
    for point in points:
        w_percent = (point.col - src_rect.left) / src_rect.width
        h_percent = (point.row - src_rect.top) / src_rect.height
        col = int(dst_rect.left + dst_rect.width * w_percent)
        row = int(dst_rect.top + dst_rect.height * h_percent)
        res_points.append(sly.Point(row, col))
    return res_points


def get_smarttool_input(figure_meta: dict) -> SmartToolInput:
    try:
        smi = figure_meta["smartToolInput"]
        return SmartToolInput.from_json(smi)
    except KeyError:
        return None


def split_figures_by_type(figures: List[FigureInfo]) -> Dict[str, List[FigureInfo]]:
    result = {}
    for figure in figures:
        smarttool_input = get_smarttool_input(figure.meta)
        if smarttool_input is not None and smarttool_input.visible:
            result.setdefault(g.GEOMETRY_NAME.SMARTTOOL, []).append(figure)
            continue
        result.setdefault(figure.geometry_type, []).append(figure)
    return result


def figure_from_prediction(
    prediction: Prediction,
    figure_id: int = None,
    object_id: int = None,
    frame_index: int = None,
    tags: List = None,
    track_id: str = None,
) -> FigureInfo:
    area = sly.deserialize_geometry(prediction.geometry_type, prediction.geometry_data).area
    return FigureInfo(
        id=figure_id,
        class_id=None,
        updated_at=None,
        created_at=None,
        entity_id=figure_id,
        object_id=object_id,
        project_id=None,
        dataset_id=None,
        frame_index=frame_index,
        geometry_type=prediction.geometry_type,
        geometry=prediction.geometry_data,
        geometry_meta=None,
        tags=tags,
        meta=prediction.meta.to_json() if prediction.meta is not None else {},
        area=area,
        track_id=track_id,
    )


def time_it(func, *args, **kwargs):
    """Measure time of function execution and return result and time in seconds."""
    tm = TinyTimer()
    result = func(*args, **kwargs)
    return tm.get_sec(), result


def parse_exception(exc: Exception, extra: Dict = None) -> Tuple[Type, str]:
    if isinstance(exc, requests.exceptions.HTTPError):
        try:
            exc_repsponse = exc.response.json()
            if "details" in exc_repsponse and "message" in exc_repsponse["details"]:
                msg = exc_repsponse["details"]["message"]
                if extra is not None:
                    msg = msg.rstrip(".")
                    msg += ". " + ", ".join(f"{k}: {v}" for k, v in extra.items())
                return exc.__class__, msg
            return exc.__class__, str(exc_repsponse)
        except Exception:
            return str(exc)
    return exc.__class__, str(exc)


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as exc:
            api: sly.Api = args[0]
            context = args[1]
            track_id = context.get("trackId", None)
            api.logger.error("An error occured:", exc_info=True)

            api.post(
                "videos.notify-annotation-tool",
                data={
                    "type": "videos:tracking-error",
                    "data": {
                        "trackId": str(track_id),
                        "error": {"message": str(exc)},
                    },
                },
            )
        return value

    return wrapper


def maybe_literal_eval(area):
    if isinstance(area, str):
        return ast.literal_eval(area)
    return area


def get_figure_area(figure_info: FigureInfo) -> float:
    if isinstance(figure_info.area, str):
        return ast.literal_eval(figure_info.area)
    if figure_info.area is None:
        return sly.deserialize_geometry(figure_info.geometry_type, figure_info.geometry).area
    return figure_info.area


def get_figures_center(figures: List[FigureInfo]):
    figure_boxes_centers: List[sly.Rectangle] = [
        sly.deserialize_geometry(figure.geometry_type, figure.geometry).to_bbox().center
        for figure in figures
    ]
    centroid = (
        sum([center.row for center in figure_boxes_centers]) / len(figure_boxes_centers),
        sum([center.col for center in figure_boxes_centers]) / len(figure_boxes_centers),
    )
    return centroid


def detect_movement_anomaly(
    this_center: Tuple[float, float], last_centers: List[Tuple[float, float]]
) -> bool:
    if len(last_centers) < 3:
        return False

    base_multiplier = 5
    variability_weight = 1

    max_distance = None
    for i in range(len(last_centers) - 1):
        distance = np.linalg.norm(np.array(last_centers[i + 1]) - np.array(last_centers[i]))
        if max_distance is None or distance > max_distance:
            max_distance = distance
    last_centers = np.array(last_centers[-3:])
    velocities = [last_centers[i + 1] - last_centers[i] for i in range(len(last_centers) - 1)]
    # distances = [np.linalg.norm(velocity) for velocity in velocities]
    # distances = distances[-30:]

    last_velocity = velocities[-1]
    previous_velocity = velocities[-2]
    acceleration = last_velocity - previous_velocity
    expected_position = last_centers[-1] + last_velocity + 0.5 * acceleration
    # variability = np.std(distances)
    # threshold = (np.mean(distances) + base_multiplier * np.std(distances)) * (
    #     1 + variability_weight * variability
    # )
    threshold = max_distance * base_multiplier
    deviation = np.linalg.norm(np.array(this_center) - expected_position)
    if deviation > threshold:
        sly.logger.debug(
            f"Anomaly detected: deviation {deviation} is greater than threshold {threshold}",
            extra={
                "deviation": deviation,
                "threshold": threshold,
                "expected_position": expected_position,
                "this_center": this_center,
                # "variability": variability,
            },
        )
        return True
    return False
