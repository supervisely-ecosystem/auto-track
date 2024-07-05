import threading
import time
from typing import Dict, List, Literal, Tuple
import functools
import uuid
import requests
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

import supervisely as sly
from supervisely.api.module_api import ApiField
from supervisely.api.entity_annotation.figure_api import FigureInfo

import src.globals as g
import src.utils as utils


def _fix_unbound(rect: utils.Prediction, point: utils.Prediction):
    rect_geom = sly.Rectangle.from_json(rect.geometry_data)
    point_geom = sly.Point.from_json(point.geometry_data)
    if rect_geom.contains_point_location(point_geom):
        return point
    if point_geom.col < rect_geom.left:
        point_geom = sly.Point(point_geom.row, rect_geom.left)
    if point_geom.col > rect_geom.right:
        point_geom = sly.Point(point_geom.row, rect_geom.right)
    if point_geom.row < rect_geom.top:
        point_geom = sly.Point(rect_geom.top, point_geom.col)
    if point_geom.row > rect_geom.bottom:
        point_geom = sly.Point(rect_geom.bottom, point_geom.col)
    return utils.Prediction(
        geometry_data=point_geom.to_json(),
        geometry_type=point.geometry_type,
        meta=point.meta,
    )


def _prepare_data_for_smarttool(
    crop_predictions: List[List[utils.Prediction]],
    pos_predictions: List[List[utils.Prediction]],
    neg_predictions: List[List[utils.Prediction]],
):
    def _to_rect(predictions: List[utils.Prediction]):
        data = predictions[0].geometry_data
        r = sly.Rectangle.from_json(data)
        return {"x": r.left, "y": r.top}, {"x": r.right, "y": r.bottom}

    def _to_points(predictions: List[utils.Prediction]):
        return [
            {
                "x": prediction.geometry_data["points"]["exterior"][0][0],
                "y": prediction.geometry_data["points"]["exterior"][0][1],
            }
            for prediction in predictions
        ]

    crops = [_to_rect(frame_predictions) for frame_predictions in crop_predictions]
    positives = [_to_points(frame_predictions) for frame_predictions in pos_predictions]
    negatives = [_to_points(frame_predictions) for frame_predictions in neg_predictions]
    return crops, positives, negatives


def get_figure_geometry_name(figure: FigureInfo):
    if figure.meta is None:
        return figure.geometry_type
    if figure.meta.get("smartToolInput", None) is None:
        return figure.geometry_type
    return "smarttool"


def _smart_segmentation_with_app(
    api: sly.Api,
    video_id: int,
    task_id: int,
    frame_index: int,
    crop_predictions: List[List[utils.Prediction]],
    pos_predictions: List[List[utils.Prediction]],
    neg_predictions: List[List[utils.Prediction]],
):
    api.logger.debug(
        "Running smarttool inference with app",
        extra={
            "task_id": task_id,
            "video_id": video_id,
            "frame_index": frame_index,
            "frames_count": len(crop_predictions),
        },
    )
    crops, positives, negatives = _prepare_data_for_smarttool(
        crop_predictions, pos_predictions, neg_predictions
    )
    context = {
        "states": [
            {
                "crop": crop,
                "positive": positive,
                "negative": negative,
                "video": {"video_id": video_id, "frame_index": frame_index + i + 1},
                "request_uid": uuid.uuid4().hex,
            }
            for i, (crop, positive, negative) in enumerate(zip(crops, positives, negatives))
        ]
    }
    r_data = api.task.send_request(
        task_id,
        "smart_segmentation_batch",
        data={},
        context=context,
    )
    result = []
    for frame in r_data:
        origin = frame.get("origin")
        bitmap = frame.get("bitmap")
        geometry = sly.Bitmap.from_json(
            {
                "bitmap": {"origin": [origin["x"], origin["y"]], "data": bitmap},
                "shape": "bitmap",
                "geometryType": "bitmap",
            }
        )
        result.append([{"type": geometry.geometry_name(), "data": geometry.to_json()}])
    return result


def _smart_segmentation_by_url(
    api: sly.Api,
    video_id: int,
    nn_url: str,
    frame_index: int,
    crop_predictions: List[List[utils.Prediction]],
    pos_predictions: List[List[utils.Prediction]],
    neg_predictions: List[List[utils.Prediction]],
) -> List[List[Dict]]:
    api.logger.debug(
        "Running smarttool inference by url",
        extra={
            "nn_url": nn_url,
            "video_id": video_id,
            "frame_index": frame_index,
            "frames_count": len(crop_predictions),
        },
    )
    crops, positives, negatives = _prepare_data_for_smarttool(
        crop_predictions, pos_predictions, neg_predictions
    )
    context = {
        "states": [
            {
                "crop": crop,
                "positive": positive,
                "negative": negative,
                "video": {"video_id": video_id, "frame_index": frame_index + i + 1},
                "request_uid": uuid.uuid4().hex,
            }
            for i, (crop, positive, negative) in enumerate(zip(crops, positives, negatives))
        ]
    }
    r = requests.post(
        f"{nn_url}/smart_segmentation_batch",
        json={
            "state": {},
            "context": context,
            "server_address": api.server_address,
            "api_token": api.token,
        },
        timeout=60,
    )
    r.raise_for_status()
    r_data = r.json()
    result = []
    for frame in r_data:
        origin = frame.get("origin")
        bitmap = frame.get("bitmap")
        geometry = sly.Bitmap.from_json(
            {
                "bitmap": {"origin": [origin["x"], origin["y"]], "data": bitmap},
                "shape": "bitmap",
                "geometryType": "bitmap",
            }
        )
        result.append([{"type": geometry.geometry_name(), "data": geometry.to_json()}])
    return result


def predict_by_url(
    api: sly.Api,
    video_id: int,
    nn_url: str,
    frame_index: int,
    frames_count: int,
    geometries_data: List[Dict],
    geometry_type: str,
) -> List[List[utils.Prediction]]:
    """Run inference using the url of the NN model server."""

    api.logger.debug(
        "Running inference by url",
        extra={
            "nn_url": nn_url,
            "video_id": video_id,
            "frame_index": frame_index,
            "frames_count": frames_count,
            "geometry_type": geometry_type,
        },
    )

    geometries = [{"type": geometry_type, "data": geom_data} for geom_data in geometries_data]
    context = {
        "videoId": video_id,
        "frameIndex": frame_index,
        "frames": frames_count,
        "input_geometries": geometries,
    }
    response = requests.post(
        f"{nn_url}/track-api",
        json={"context": context, "server_address": api.server_address, "api_token": api.token},
        timeout=60,
    )
    response.raise_for_status()
    results = response.json()

    results = [
        [
            utils.Prediction(
                geometry_data=prediction["data"],
                geometry_type=prediction["type"],
            )
            for prediction in frame_predictions
        ]
        for frame_predictions in results
    ]
    return results


def predict_with_app(
    api: sly.Api,
    video_id: int,
    task_id: int,
    geometries_data: List[Dict],
    geometry_type: str,
    frame_index: int,
    frames_count: int,
) -> List[List[utils.Prediction]]:
    """Run inference using the NN model Supervisely app session."""
    geometries = [{"type": geometry_type, "data": geometry} for geometry in geometries_data]
    data = {
        "videoId": video_id,
        "frameIndex": frame_index,
        "frames": frames_count,
        "input_geometries": geometries,
        # "direction": "forward"  # optional
    }
    response = api.task.send_request(task_id, "track-api", {}, context=data)

    results = [
        [
            utils.Prediction(
                geometry_data=prediction["data"],
                geometry_type=prediction["type"],
            )
            for prediction in frame_predictions
        ]
        for frame_predictions in response
    ]
    return results


def predict_smarttool(
    api: sly.Api,
    video_id: int,
    track_id: int,
    nn_settings: Dict,
    figure_metas: List[Dict],
    frame_index: int = None,
    frames_count: int = None,
) -> List[List[utils.Prediction]]:
    """Predict smarttool geometries using the specified NN model settings."""
    rect_settings = nn_settings[sly.Rectangle.geometry_name()]
    point_settings = nn_settings[sly.Point.geometry_name()]
    smarttool_settings = nn_settings["smarttool"]

    # prepare bboxes and points for figures
    figure_crops = []
    figure_positives = []
    figure_negatives = []
    positives_figure_index = []
    negatives_figure_index = []
    for f_index, f_meta in enumerate(figure_metas):
        smarttool_input = utils.get_smarttool_input(f_meta)
        smarttool_input_data = smarttool_input.to_json()
        crop = smarttool_input_data["crop"]
        positive = smarttool_input_data["positive"]
        negative = smarttool_input_data["negative"]
        [top, bottom] = sorted([crop[0][1], crop[1][1]])
        [left, right] = sorted([crop[0][0], crop[1][0]])
        crop_rect = sly.Rectangle(top, left, bottom, right)
        figure_crops.append(crop_rect)
        figure_positives.append(positive)
        figure_negatives.append(negative)
        positives_figure_index.extend([f_index for _ in positive])
        negatives_figure_index.extend([f_index for _ in negative])

    # predict bboxes and points for figures for each frame
    with ThreadPoolExecutor(2) as executor:
        if "url" in rect_settings:
            nn_url = rect_settings["url"]
            crop_predictions_task = executor.submit(
                predict_by_url,
                api=api,
                video_id=video_id,
                nn_url=nn_url,
                frame_index=frame_index,
                frames_count=frames_count,
                geometries_data=[crop_rect.to_json() for crop_rect in figure_crops],
                geometry_type=crop_rect.geometry_name(),
            )
        else:
            task_id = rect_settings["task_id"]
            crop_predictions_task = executor.submit(
                predict_with_app,
                api=api,
                video_id=video_id,
                task_id=task_id,
                geometries_data=[crop_rect.to_json() for crop_rect in figure_crops],
                geometry_type=crop_rect.geometry_name(),
                frame_index=frame_index,
                frames_count=frames_count,
            )
        if "url" in point_settings:
            nn_url = point_settings["url"]
            pos_predictions_task = executor.submit(
                predict_by_url,
                api=api,
                video_id=video_id,
                nn_url=nn_url,
                frame_index=frame_index,
                frames_count=frames_count,
                geometries_data=[
                    sly.Point(point[1], point[0]).to_json()
                    for positives in figure_positives
                    for point in positives
                ],
                geometry_type=sly.Point.geometry_name(),
            )
        else:
            task_id = point_settings["task_id"]
            pos_predictions_task = executor.submit(
                predict_with_app,
                api=api,
                video_id=video_id,
                task_id=task_id,
                geometries_data=[
                    sly.Point(point[1], point[0]).to_json()
                    for positives in figure_positives
                    for point in positives
                ],
                geometry_type=sly.Point.geometry_name(),
                frame_index=frame_index,
                frames_count=frames_count,
            )
    crop_predictions = crop_predictions_task.result()
    neg_predictions = []
    for figure_idx in range(len(figure_metas)):
        points = [sly.Point(point[1], point[0]) for point in figure_negatives[figure_idx]]
        src_rect = figure_crops[figure_idx]
        for frame_idx in range(frames_count):
            dst_rect = sly.Rectangle.from_json(
                crop_predictions[frame_idx][figure_idx].geometry_data
            )
            points = utils.move_points_relative(src_rect=src_rect, points=points, dst_rect=dst_rect)
            if len(neg_predictions) <= frame_idx:
                neg_predictions.append([])
            neg_predictions[frame_idx].extend(
                [
                    utils.Prediction(
                        geometry_data=point.to_json(),
                        geometry_type=sly.Point.geometry_name(),
                    )
                    for point in points
                ]
            )
            src_rect = dst_rect
    pos_predictions = pos_predictions_task.result()
    # crop predictions = List[List[utils.Prediction]]
    # crop_predictions[i] - predictions for i-th frame
    # crop_predictions[i][j] - prediction for the j-th figure for the i-th frame
    # same with points predictions except that for each figure there are multiple points

    # predict smarttool for each figure for each frame
    results = [[] for _ in range(frames_count)]
    for figure_idx in range(len(figure_metas)):
        figure_crop_predictions = [[frame[figure_idx]] for frame in crop_predictions]

        i_from = positives_figure_index.index(figure_idx)
        try:
            i_to = positives_figure_index.index(figure_idx + 1)
        except ValueError:
            i_to = len(positives_figure_index)
        figure_pos_predictions = [frame[i_from:i_to] for frame in pos_predictions]
        figure_pos_predictions = [
            [
                _fix_unbound(rect=figure_crop_predictions[frame_idx][0], point=point)
                for point in frame
            ]
            for frame_idx, frame in enumerate(figure_pos_predictions)
        ]

        if figure_idx in negatives_figure_index:
            i_from = negatives_figure_index.index(figure_idx)
            try:
                i_to = negatives_figure_index.index(figure_idx + 1)
            except ValueError:
                i_to = len(negatives_figure_index)
            figure_neg_predictions = [frame[i_from:i_to] for frame in neg_predictions]
        else:
            figure_neg_predictions = [[] for _ in neg_predictions]

        if "url" in smarttool_settings:
            nn_url = smarttool_settings["url"]
            smarttool_predictions = _smart_segmentation_by_url(
                api=api,
                video_id=video_id,
                nn_url=nn_url,
                frame_index=frame_index,
                crop_predictions=figure_crop_predictions,
                pos_predictions=figure_pos_predictions,
                neg_predictions=figure_neg_predictions,
            )
        else:
            task_id = smarttool_settings["task_id"]
            smarttool_predictions = _smart_segmentation_with_app(
                api=api,
                video_id=video_id,
                task_id=task_id,
                frame_index=frame_index,
                crop_predictions=figure_crop_predictions,
                pos_predictions=figure_pos_predictions,
                neg_predictions=figure_neg_predictions,
            )
        for frame_idx, frame_predictions in enumerate(
            zip(
                figure_crop_predictions,
                figure_pos_predictions,
                figure_neg_predictions,
                smarttool_predictions,
            )
        ):
            crops, pos_points, neg_points, smarttools = frame_predictions
            prediction = utils.Prediction(
                geometry_data=smarttools[0]["data"],
                geometry_type=smarttools[0]["type"],
                meta=utils.Meta(
                    smi=utils.SmartToolInput(
                        crop=sly.Rectangle.from_json(crops[0].geometry_data),
                        positive=[sly.Point.from_json(p.geometry_data) for p in pos_points],
                        negative=[sly.Point.from_json(n.geometry_data) for n in neg_points],
                        visible=True,
                    ),
                    track_id=track_id,
                    tool="smart",
                ),
            )
            results[frame_idx].append(prediction)
    return results


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
                        "error": {"message": repr(exc)},
                    },
                },
            )
        return value

    return wrapper


class Track:

    class Timeline:
        def __init__(
            self,
            track: "Track",
            object_id: int,
            start_frame_index: int,
            end_frame_index: int,
            start_frame_figures: List[FigureInfo],
            can_continue: bool,
        ):
            self.track = track
            self.object_id = object_id  # not unique
            self.start_frame_index = start_frame_index
            self.end_frame_index = end_frame_index
            self.start_frame_figures = start_frame_figures
            self.can_continue = can_continue
            self.progress = {}
            self.predicted_figures_ids: Dict[int, List[int]] = {}

        def get_batch(self, batch_size: int) -> Tuple[int, int, List[FigureInfo]]:
            """Get the next batch of frames to track."""
            batch_from = self.progress.get("frame_index", None)
            if batch_from is None:
                return (
                    self.start_frame_index,
                    min(self.start_frame_index + batch_size, self.end_frame_index),
                    self.start_frame_figures,
                )
            return (
                batch_from,
                min(batch_from + batch_size, self.end_frame_index),
                self.progress["figures"],
            )

        def add_predictions(self, frame_index: int, figures: List[FigureInfo]):
            """Add predicted figures for the specified frame index."""
            self.predicted_figures_ids.setdefault(frame_index, []).extend(
                [figure.id for figure in figures]
            )
            if "frame_index" not in self.progress or frame_index >= self.progress["frame_index"]:
                self.progress = {
                    "frame_index": frame_index,
                    "figures": figures,
                }

        def cut(self, frame_index):
            """Delete figures predicted by this timeline after the specified frame index
            and set end_frame_index."""
            figures_to_delete = []
            frames_to_delete = []
            for pred_frame_index, figure_ids in self.predicted_figures_ids.items():
                if pred_frame_index > frame_index:
                    figures_to_delete.extend(figure_ids)
                    frames_to_delete.append(pred_frame_index)
            for frame in frames_to_delete:
                self.predicted_figures_ids.pop(frame, None)
            threading.Thread(target=self.track.remove_figures, args=(figures_to_delete,)).start()
            self.end_frame_index = frame_index

        def continue_timeline(
            self,
            frame_index,
            frames_count,
            object_info=None,
            figures=None,
            no_object_tags_tag_id=None,
        ):
            """Continue tracking after the specified frame index for the specified number of
            frames.
            If frame_index is > end_frame_index or frame_index+frames_count < end_frame_index,
            then do nothing"""
            self.track.logger.debug(
                "Continue timeline",
                extra={
                    **self.log_data(),
                    "frame_index": frame_index,
                    "frames_count": frames_count,
                },
            )
            if not self.can_continue:
                return
            if frame_index > self.end_frame_index:
                return
            if frame_index + frames_count <= self.end_frame_index:
                return

            end_frame_index = frame_index + frames_count
            # get tag if of 'no-objects' tag meta
            if no_object_tags_tag_id is None:
                no_object_tag_metas = self.track.api.video.object.tag.get_list(
                    project_id=self.track.project_id,
                    filters=[
                        {"field": "name", "operator": "=", "value": "no-objects"},
                    ],
                )  # should be of length 1 or 0
                no_object_tags_tag_id = (
                    no_object_tag_metas[0].id if len(no_object_tag_metas) > 0 else None
                )
            if object_info is None:
                object_info = self.track.api.video.object.get_info_by_id(self.object_id)
            frames_with_no_objects = set()
            for tag in object_info.tags:
                if no_object_tags_tag_id == tag["tagId"]:
                    range_from, range_to = tag["frameRange"]
                    for i in range(range_from, range_to + 1):
                        frames_with_no_objects.add(i)

            # get figures for the specified object and frame range
            if figures is None:
                figures: List[FigureInfo] = self.track.api.video.figure.get_list(
                    dataset_id=self.track.dataset_id,
                    filters=[
                        {"field": "objectId", "operator": "=", "value": self.object_id},
                        # {"field": "startFrame", "operator": "=", "value": start_frame_index},
                        # {"field": "endFrame", "operator": "=", "value": start_frame_index + frames_count},
                    ],
                )
            figures = [
                figure
                for figure in figures
                if figure.frame_index > self.start_frame_index
                and figure.frame_index <= end_frame_index
                and figure.object_id == self.object_id
            ]
            for figure in figures:
                track_id = figure.track_id
                if track_id is None:
                    end_frame_index = min(end_frame_index, figure.frame_index - 1)
            for frame_index in range(self.start_frame_index + 1, end_frame_index + 1):
                if frame_index in frames_with_no_objects:
                    end_frame_index = frame_index - 1
                    break

            self.end_frame_index = min(end_frame_index, self.track.video_info.frames_count - 1)

            self.track.logger.debug(
                "Timeline continued", extra={**self.log_data(), **self.track.logger_extra}
            )

        def log_data(self):
            return {
                "object_id": self.object_id,
                "start_frame_index": self.start_frame_index,
                "end_frame_index": self.end_frame_index,
                "start_frame_figures": [figure.id for figure in self.start_frame_figures],
                "progress": {
                    "frame_index": self.progress.get("frame_index", None),
                    "figures": [f.id for f in self.progress.get("figures", [])],
                },
                "can_continue": self.can_continue,
            }

    def __init__(
        self,
        track_id: str,
        session_id: str,
        api: sly.Api,
        video_id: int,
        object_ids: List[int],
        frame_index: int,
        frames_count: int,
        nn_settings: Dict,
        user_id: int = None,
        cloud_token: str = None,
        cloud_action_id: str = None,
    ):
        self.track_id = track_id
        self.session_id = session_id
        self.api = api
        if session_id is not None:
            self.api.headers.update({"x-toolbox-session-id": session_id})
        self.video_id = video_id
        self.object_ids = list(set(object_ids))
        self.nn_settings = nn_settings
        self.frame_index = frame_index
        self.frames_count = frames_count
        self.init_frames_count = frames_count
        self.logger = self.api.logger
        self.logger_extra = {
            "video_id": self.video_id,
            "track_id": self.track_id,
        }
        self.user_id = user_id
        self.cloud_token = cloud_token
        self.cloud_action_id = cloud_action_id

        self.global_stop_indicatior = False
        self.figure_progress = 0
        self.total_figures = 0
        self.batch_size = 8
        self.updates = []
        self._lock = Lock()

        self.video_info = api.video.get_info_by_id(video_id)
        self.dataset_id = self.video_info.dataset_id
        self.project_id = self.video_info.project_id
        self.timelines = self._init_timelines()
        self.update_progress()
        self.logger.debug(
            "Track inited",
            extra={
                "timelines": [
                    {
                        "start_frame_index": timeline.start_frame_index,
                        "end_frame_index": timeline.end_frame_index,
                        "start_figures": [figure.id for figure in timeline.start_frame_figures],
                    }
                    for timeline in self.timelines
                ]
            },
        )

    def remove_figures(self, figure_ids: List[int]):
        """Try to remove figures from"""
        try:
            self.api.video.figure.remove_batch(figure_ids)
        # pylint: disable=broad-except
        except Exception as e:
            self.logger.warning(
                "Unable to remove figures in batch will try to remove one by one: %s",
                e,
                extra={
                    **self.logger_extra,
                    "figure_ids": figure_ids,
                },
            )
        for figure_id in figure_ids:
            try:
                self.api.video.figure.remove(figure_id)
            # pylint: disable=broad-except
            except Exception:
                pass
                # self.logger.warning(
                #     "Unable to remove figure: %s",
                #     e,
                #     extra={**self.logger_extra, "figure_id": figure_id},
                # )

    def init_timelines(
        self,
        object_id: int,
        start_frame_index: int,
        frames_count: int,
        object_info=None,
        figures=None,
        no_object_tags_tag_id=None,
        only_one_timeline=False,
    ) -> List[Timeline]:
        """Initialize timelines for the specified object.
        This function will
        1. get all figures for the specified object and frame range
        2. find key figure that limits tracking (annotated by user)
        3. find frame with 'no objects' tag that limits tracking
        4. delete all figures that have the same track_id as the current track
        5. create Timeline object for each key figure"""

        # get tag if of 'no-objects' tag meta
        if no_object_tags_tag_id is None:
            no_object_tag_metas = self.api.video.object.tag.get_list(
                project_id=self.project_id,
                filters=[
                    {"field": "name", "operator": "=", "value": "no-objects"},
                ],
            )  # should be of length 1 or 0
            no_object_tags_tag_id = (
                no_object_tag_metas[0].id if len(no_object_tag_metas) > 0 else None
            )
        if object_info is None:
            object_info = self.api.video.object.get_info_by_id(object_id)
        frames_with_no_objects = set()
        for tag in object_info.tags:
            if no_object_tags_tag_id == tag["tagId"]:
                range_from, range_to = tag["frameRange"]
                for i in range(range_from, range_to + 1):
                    frames_with_no_objects.add(i)

        # get figures for the specified object and frame range
        if figures is None:
            figures: List[FigureInfo] = self.api.video.figure.get_list(
                dataset_id=self.dataset_id,
                filters=[
                    {"field": "objectId", "operator": "=", "value": object_id},
                    # {"field": "startFrame", "operator": "=", "value": start_frame_index},
                    # {"field": "endFrame", "operator": "=", "value": start_frame_index + frames_count},
                ],
            )
        figures = [
            figure
            for figure in figures
            if figure.frame_index >= start_frame_index
            and figure.frame_index <= start_frame_index + frames_count
            and figure.object_id == object_id
        ]

        # find key figures and figures with the same track_id to delete it
        key_figures = {}
        to_delete_figure_ids = set()
        for figure in figures:
            track_id = figure.track_id

            # add key figures
            if figure.frame_index == start_frame_index:
                key_figures.setdefault(figure.frame_index, []).append(figure)
            elif track_id is None:
                key_figures.setdefault(figure.frame_index, []).append(figure)

            # delete figures with the same track_id
            elif track_id == self.track_id:
                to_delete_figure_ids.add(figure.id)

        # define timeline for each key figure
        timelines = []
        for i, key_figure_frame_index in enumerate(sorted(key_figures.keys())):
            # define end frame index for timeline
            timeline_end_frame_index = min(
                self.frame_index + self.frames_count, self.video_info.frames_count - 1
            )
            can_continue = True
            reason = "default self.frame_index + self.frames_count"
            for frame_index in range(key_figure_frame_index + 1, timeline_end_frame_index):
                if frame_index in frames_with_no_objects:
                    timeline_end_frame_index = frame_index - 1
                    can_continue = False
                    reason = "frame with 'no objects' tag found"
                    break
                if frame_index in key_figures:
                    timeline_end_frame_index = frame_index - 1
                    can_continue = False
                    reason = "key figure found"
                    break
            figures = key_figures[key_figure_frame_index]
            timeline = self.Timeline(
                track=self,
                object_id=object_id,
                start_frame_index=key_figure_frame_index,
                end_frame_index=timeline_end_frame_index,
                start_frame_figures=figures,
                can_continue=can_continue,
            )
            timelines.append(timeline)
            self.logger.debug(
                "Timeline inited",
                extra={
                    **timeline.log_data(),
                    "reason": reason,
                },
            )
            if only_one_timeline:
                break

        # delete tracked figures
        threading.Thread(target=self.remove_figures, args=(list(to_delete_figure_ids),)).start()

        return timelines

    def _init_timelines(self) -> List[Timeline]:
        """Initialize timelines for track objects.
        See init_timeline for more details."""
        object_infos = {
            object_info.id: object_info
            for object_info in self.api.video.object.get_list(
                dataset_id=self.dataset_id,
                filters=[{"field": "id", "operator": "in", "value": self.object_ids}],
            )
        }
        no_object_tags = self.api.video.object.tag.get_list(
            project_id=self.project_id,
            filters=[
                {"field": "name", "operator": "=", "value": "no-objects"},
            ],
        )
        no_object_tag_id = no_object_tags[0].id if no_object_tags else None
        figures: List[FigureInfo] = self.api.video.figure.get_list(
            dataset_id=self.dataset_id,
            filters=[
                {"field": "objectId", "operator": "in", "value": self.object_ids},
                # {"field": "startFrame", "operator": "=", "value": start_frame_index},
                # {"field": "endFrame", "operator": "=", "value": start_frame_index + frames_count},
            ],
        )
        figures = [
            figure
            for figure in figures
            if figure.frame_index >= self.frame_index
            and figure.frame_index <= self.frame_index + self.frames_count
        ]
        figures_dict = {}
        for figure in figures:
            figures_dict.setdefault(figure.object_id, []).append(figure)

        timelines: List[Track.Timeline] = []
        for object_id in self.object_ids:
            timelines.extend(
                self.init_timelines(
                    object_id=object_id,
                    start_frame_index=self.frame_index,
                    frames_count=self.frames_count,
                    figures=figures_dict.get(object_id, []),
                    object_info=object_infos[object_id],
                    no_object_tags_tag_id=no_object_tag_id,
                )
            )

        self.logger.debug(
            "%s Timelines inited",
            len(timelines),
            extra={
                "timelines": [
                    {
                        "object_id": timeline.object_id,
                        "start_frame_index": timeline.start_frame_index,
                        "end_frame_index": timeline.end_frame_index,
                        "figures_count": len(timeline.start_frame_figures),
                    }
                    for timeline in timelines
                ]
            },
        )

        return timelines

    def put_update(
        self,
        object_ids: List[int],
        frame_index: int,
        frames_count: int,
        update_type: Literal["track", "continue", "delete"] = "track",
    ):
        self.notify(task="Continue tracking", pos_increment=0)
        """Put update to the updates list."""
        with self._lock:
            self.updates.append(
                {
                    "object_ids": object_ids,
                    "frame_index": frame_index,
                    "frames_count": frames_count,
                    "type": update_type,
                }
            )

    def pop_updates(self):
        """pop updates"""
        with self._lock:
            if len(self.updates) == 0:
                return []
            updates = [upd for upd in self.updates]
            self.updates = []
            return updates

    def notify(
        self,
        stop: bool = False,
        task: str = "not defined",
        pos_increment: int = 1,
    ):
        """Notify labeling tool about tracking progress."""
        self.figure_progress += pos_increment

        if stop:
            pos = self.total_figures
        else:
            pos = self.figure_progress

        fstart = self.frame_index
        fend = min(self.frame_index + self.frames_count, self.video_info.frames_count - 1)

        self.logger.debug("Task: %s, Notification status: %s/%s", task, pos, self.total_figures)

        self.global_stop_indicatior = self.api.video.notify_progress(
            self.track_id,
            self.video_id,
            fstart,
            fend,
            pos,
            self.total_figures,
        )
        if self.global_stop_indicatior and self.figure_progress < self.total_figures:
            self.logger.info("Tracking stoped by user.")

    def _create_and_notify_batch(
        self,
        object_ids: List,
        frame_indexes: List,
        predictions_json: List[Dict],
        geometry_type: str,
    ):
        def _get_figures_to_delete(object_ids, frame_indexes):
            figures: List[FigureInfo] = self.api.video.figure.get_list(
                dataset_id=self.dataset_id,
                filters=[
                    {"field": "objectId", "operator": "in", "value": object_ids},
                    # {"field": "startFrame", "operator": ">=", "value": start_frame_index},
                    # {"field": "endFrame", "operator": "<=", "value": start_frame_index + frames_count},
                ],
            )
            figures = [fig for fig in figures if fig.frame_index in frame_indexes]
            figures = [fig for fig in figures if fig.track_id is not None]
            return figures

        figures_to_delete = _get_figures_to_delete(object_ids, frame_indexes)
        if figures_to_delete:
            threading.Thread(
                target=self.remove_figures, args=([fig.id for fig in figures_to_delete],)
            ).start()

        figures_json = [
            {
                ApiField.OBJECT_ID: object_id,
                ApiField.GEOMETRY_TYPE: prediction_json["type"],
                ApiField.GEOMETRY: prediction_json["data"],
                ApiField.META: {ApiField.FRAME: frame_index, **prediction_json["meta"]},
                ApiField.TRACK_ID: self.track_id,
            }
            for object_id, frame_index, prediction_json in zip(
                object_ids, frame_indexes, predictions_json
            )
        ]
        sly.logger.debug(
            "Upload predictions batch",
            extra={
                "video_id": self.video_id,
                "track_id": self.track_id,
                "geometry_type": geometry_type,
                "object_ids": object_ids,
                "frame_range": (frame_indexes[0], frame_indexes[-1]),
                "predictions_count": len(figures_json),
            },
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
        sly.logger.debug(
            "Figures added",
            extra={
                "video_id": self.video_id,
                "track_id": self.track_id,
                "geometry_type": geometry_type,
                "object_ids": object_ids,
                "frame_range": (frame_indexes[0], frame_indexes[-1]),
                "predictions_count": len(figures_json),
            },
        )
        # notify lableing tool
        self.notify(task="Figures upload", pos_increment=len(figures_json))

        return [key_id_map.get_figure_id(figure_key) for figure_key in figures_keys]

    def upload_predictions(
        self,
        frame_index: int,
        predictions: List[List[utils.Prediction]],
        object_ids: List[int],
        geometry_type: str,
    ) -> List[List[int]]:
        """Upload predictions to the video and notify labeling tool about progress.
        predictions: List[List[Prediction]] List of lists for geometries on frame.
        """
        frame_indexes = list(
            range(
                frame_index + 1,
                frame_index + 2 + len(predictions),
            )
        )
        # flatten predictions
        all_predictions = []
        for frame_idx, frame_predictions in zip(frame_indexes, predictions):
            for object_id, prediction in zip(object_ids, frame_predictions):
                all_predictions.append(
                    {
                        "object_id": object_id,
                        "frame_index": frame_idx,
                        "prediction": prediction,
                    }
                )
        # upload predictions in batches
        all_figure_ids = []
        for batch in sly.batched(all_predictions, batch_size=50):
            batch_figure_ids = self._create_and_notify_batch(
                object_ids=[pred["object_id"] for pred in batch],
                frame_indexes=[pred["frame_index"] for pred in batch],
                predictions_json=[pred["prediction"].to_json() for pred in batch],
                geometry_type=geometry_type,
            )
            all_figure_ids.extend(batch_figure_ids)

        # return ids of created figures in the same order as input
        per_frame_figure_ids = []
        j = 0
        for i, frame_predictions in enumerate(predictions):
            per_frame_figure_ids.append([])
            for _ in range(len(frame_predictions)):
                per_frame_figure_ids[i].append(all_figure_ids[j])
                j += 1
        return per_frame_figure_ids

    def run_geometry(
        self,
        geometry_type: str,
        figures: List[FigureInfo],
        frame_index: int,
        frames_count: int,
    ) -> List[List[FigureInfo]]:
        """Run tracking for the specified geometry type."""
        sly.logger.debug("Tracking geometry type %s", geometry_type)
        if geometry_type not in self.nn_settings:
            raise KeyError(f"No neural network settings for geometry type {geometry_type}")
        if geometry_type == "smarttool":
            predictions = predict_smarttool(
                api=self.api,
                video_id=self.video_id,
                track_id=self.track_id,
                nn_settings=self.nn_settings,
                figure_metas=[figure.meta for figure in figures],
                frame_index=frame_index,
                frames_count=frames_count,
            )
        elif "url" in self.nn_settings[geometry_type]:
            url = self.nn_settings[geometry_type]["url"]
            predictions = predict_by_url(
                api=self.api,
                video_id=self.video_id,
                frame_index=frame_index,
                frames_count=frames_count,
                nn_url=url,
                geometries_data=[figure.geometry for figure in figures],
                geometry_type=geometry_type,
            )
        else:
            task_id = self.nn_settings[geometry_type]["task_id"]
            predictions = predict_with_app(
                api=self.api,
                video_id=self.video_id,
                task_id=task_id,
                geometries_data=[figure.geometry for figure in figures],
                geometry_type=geometry_type,
                frame_index=frame_index,
                frames_count=frames_count,
            )
        uploaded_figure_ids = self.upload_predictions(
            frame_index=frame_index,
            predictions=predictions,
            object_ids=[figure.object_id for figure in figures],
            geometry_type=geometry_type,
        )
        result = []
        for frame_predictions, frame_uploaded_figure_ids in zip(predictions, uploaded_figure_ids):
            result.append([])
            for i, prediction, uploaded_figure_id, src_figure in zip(
                range(len(frame_predictions)), frame_predictions, frame_uploaded_figure_ids, figures
            ):
                result[-1].append(
                    utils.figure_from_prediction(
                        prediction=prediction,
                        figure_id=uploaded_figure_id,
                        object_id=src_figure.object_id,
                        frame_index=frame_index + 1 + i,
                        track_id=self.track_id,
                    )
                )
        return result

    def _iteration(
        self, figures_by_type: Dict[str, List[FigureInfo]], frame_index: int, frames_count: int
    ) -> Dict[str, List[List[FigureInfo]]]:
        """Run tracking iteration for the specified figures.
        Returns dictionary with predictions and figure ids by geometry type."""

        geom_types = list(figures_by_type.keys())
        with ThreadPoolExecutor(len(geom_types)) as executor:
            tasks_by_geom_type = {}
            for geom_type in geom_types:
                task = executor.submit(
                    self.run_geometry,
                    geometry_type=geom_type,
                    figures=figures_by_type[geom_type],
                    frame_index=frame_index,
                    frames_count=frames_count,
                )
                tasks_by_geom_type[geom_type] = task
            results_by_geom_type = {
                geom_type: task.result() for geom_type, task in tasks_by_geom_type.items()
            }
        return results_by_geom_type

    def _get_batch(
        self, batch_size: int
    ) -> Tuple[int, int, List[Tuple[Timeline, List[FigureInfo]]]]:
        """Get batch of frames to track. Returns batch_from, batch_to, timelines_with_figures."""
        # filter out timelines with no frames to track
        filtered_timelines = []
        for timeline in self.timelines:
            timeline_batch_from, timeline_batch_to, timeline_figures = timeline.get_batch(
                batch_size
            )
            if timeline_batch_from >= timeline_batch_to:
                # no frames to track
                continue
            filtered_timelines.append(
                (timeline, timeline_batch_from, timeline_batch_to, timeline_figures)
            )
            self.logger.debug(
                "timeline batch",
                extra={
                    "timeline": timeline.log_data(),
                    "batch": (timeline_batch_from, timeline_batch_to),
                },
            )
        if len(filtered_timelines) == 0:
            return None, None, None
        # find the lowest starting frame index
        batch_from = None
        for (
            timeline,
            timeline_batch_from,
            timeline_batch_to,
            timeline_figures,
        ) in filtered_timelines:
            batch_from = (
                min(timeline_batch_from, batch_from)
                if batch_from is not None
                else timeline_batch_from
            )
        # find the second lowest starting frame index
        batch_to = batch_from + batch_size
        for (
            timeline,
            timeline_batch_from,
            timeline_batch_to,
            timeline_figures,
        ) in filtered_timelines:
            batch_to = (
                min(timeline_batch_from, batch_to)
                if timeline_batch_from != batch_from
                else batch_to
            )
        # find the batch to
        for (
            timeline,
            timeline_batch_from,
            timeline_batch_to,
            timeline_figures,
        ) in filtered_timelines:
            batch_to = min(timeline_batch_to, batch_to)
        # leave only timelines starting at batch_from
        result = []
        for (
            timeline,
            timeline_batch_from,
            timeline_batch_to,
            timeline_figures,
        ) in filtered_timelines:
            if timeline_batch_from == batch_from:
                result.append((timeline, timeline_figures))
        return batch_from, batch_to, result

    def get_timelines_by_object_id(self, object_id: int) -> Tuple[List[int], List[Timeline]]:
        """Get indexes and timelines for the specified object_id."""
        indexes = []
        timelines = []
        for i, timeline in enumerate(self.timelines):
            if timeline.object_id == object_id:
                indexes.append(i)
                timelines.append(timeline)
        return indexes, timelines

    def continue_track(self, frame_index: int, frames_count: int):
        """Continue tracking after the specified frame index for the specified number of frames.
        Updates timelines, frames count"""
        for timeline in self.timelines:
            timeline.continue_timeline(frame_index, frames_count)
        self.frames_count = frame_index + frames_count - self.frame_index

    def apply_update(self, update: Dict):
        update_type = update["type"]
        update_frame_index = update["frame_index"]
        update_frames_count = update["frames_count"]
        update_object_ids = update["object_ids"]
        if update_type == "delete":
            for object_id in update_object_ids:
                timeline_indexes, timelines = self.get_timelines_by_object_id(object_id)
                for timeline_index, timeline in zip(timeline_indexes, timelines):
                    timeline.cut(update_frame_index - 1)
            return
        elif update_type == "continue":
            self.continue_track(update_frame_index, update_frames_count)
            return
        elif update_type == "track":
            # limit update_frames_count by the current last frame
            update_frames_count = min(
                update_frames_count, self.frame_index + self.frames_count - update_frame_index
            )
            for object_id in update_object_ids:
                timeline_indexes, timelines = self.get_timelines_by_object_id(object_id)
                # if no timelines for the object then create new timelines
                if len(timelines) == 0:
                    new_timelines = self.init_timelines(
                        object_id,
                        start_frame_index=update_frame_index,
                        frames_count=update_frames_count,
                    )
                    self.timelines.extend(new_timelines)
                    continue

                # modify existing timeline
                timeline_created = False
                for timeline_index, timeline in zip(timeline_indexes, timelines):
                    if timeline.start_frame_index == update_frame_index:
                        self.timelines[timeline_index] = self.init_timelines(
                            object_id=object_id,
                            start_frame_index=update_frame_index,
                            frames_count=update_frames_count,
                            only_one_timeline=True,
                        )[0]
                        timeline_created = True
                        break
                # if there is a timeline on that frame then replace it with new one and continue
                if timeline_created:
                    continue

                # find timelines left and right from update
                left = None
                right = None
                for timeline in timelines:
                    if timeline.start_frame_index < update_frame_index:
                        if left is None or left.start_frame_index < timeline.start_frame_index:
                            left = timeline
                    if timeline.start_frame_index > update_frame_index:
                        if right is None or right.start_frame_index > timeline.start_frame_index:
                            right = timeline

                # create timeline
                new_timeline_frames_count = update_frames_count
                # if there is a timeline to the right, limit the new timeline by it
                if right is not None:
                    new_timeline_frames_count = min(
                        update_frames_count, right.start_frame_index - update_frame_index - 1
                    )
                new_timeline = self.init_timelines(
                    object_id=object_id,
                    start_frame_index=update_frame_index,
                    frames_count=new_timeline_frames_count,
                    only_one_timeline=True,
                )
                self.timelines.extend(new_timeline)
                # if there is a timeline to the left, cut it
                if left is not None:
                    left.predicted_figures_ids.pop(update_frame_index, None)
                    left.cut(update_frame_index - 1)

    def update_progress(self):
        self.total_figures = 0
        self.figure_progress = 0
        start = None
        end = None
        for timeline in self.timelines:
            timeline_frames = timeline.end_frame_index - timeline.start_frame_index
            timeline_figures_count = len(timeline.start_frame_figures)
            self.total_figures += timeline_frames * timeline_figures_count
            timeline_tracked = sum(len(figs) for figs in timeline.predicted_figures_ids.values())
            self.figure_progress += timeline_tracked
            start = (
                min(timeline.start_frame_index, start)
                if start is not None
                else timeline.start_frame_index
            )
            end = (
                max(timeline.end_frame_index, end) if end is not None else timeline.end_frame_index
            )
        self.frame_index = start
        self.frames_count = end - start

    def log_timelines(self):
        self.logger.debug(
            "Timelines",
            extra={
                "timelines": [timeline.log_data() for timeline in self.timelines],
                **self.logger_extra,
            },
        )

    def run(self):
        self.notify(task="Start tracking", pos_increment=0)
        while True:
            total_tm = sly.TinyTimer()

            # If there are updates in the queue, merge them to the current track
            updates = self.pop_updates()
            if len(updates) > 0:
                self.logger.debug("Apply updates", extra={"updates": updates})
            for update in updates:
                self.apply_update(update)
            if len(updates) > 0:
                self.update_progress()
                self.log_timelines()
            apply_updates_time = sly.TinyTimer.get_sec(total_tm)

            # get batch
            tm = sly.TinyTimer()
            frame_from, frame_to, timelines_data = self._get_batch(self.batch_size)
            if frame_from is None:  # no frames to track
                received_updates = False
                for _ in range(30):
                    if len(self.updates) > 0:
                        self.logger.debug(
                            "Received updates after tracking finished",
                            extra={"track_id": self.track_id},
                        )
                        received_updates = True
                        break
                    time.sleep(1)  # sleep for 30 seconds to receive updates
                if received_updates:
                    continue
                break

            # get batch figures
            figures_by_type = {}
            timeline_index_by_type = {}  # to get timeline knowing figure index in figures_by_type
            for timeline_index, (_, timeline_figures) in enumerate(timelines_data):
                for timeline_figure in timeline_figures:
                    geom_name = get_figure_geometry_name(timeline_figure)
                    figures_by_type.setdefault(geom_name, []).append(timeline_figure)
                    timeline_index_by_type.setdefault(geom_name, []).append(timeline_index)
            get_batch_time = sly.TinyTimer.get_sec(tm)

            if frame_from >= frame_to:
                self.logger.debug("No frames to track", extra={"frame_index": frame_from})
                continue

            self.logger.debug(
                "Start iteration",
                extra={
                    "frame_from": frame_from,
                    "frame_to": frame_to,
                    "figures count": {
                        geom_name: len(figures) for geom_name, figures in figures_by_type.items()
                    },
                },
            )
            self.notify(task="Track", pos_increment=0)

            if self.cloud_token is not None and self.cloud_action_id is not None:
                figures_count = sum(len(figures) for figures in figures_by_type.values())
                expected_predictions = figures_count * (frame_to - frame_from)
                try:
                    transaction_id = self.api.cloud.billing_reserve(
                        self.user_id,
                        items_count=expected_predictions,
                        cloud_token=self.cloud_token,
                        cloud_action_id=self.cloud_action_id,
                    )["transactionId"]
                except Exception as e:
                    self.logger.error("Unable to reserve tokens for predictions", exc_info=True)
                    raise RuntimeError("Unable to reserve tokens for predictions") from None

            # run iteration
            tm = sly.TinyTimer()
            iteration_result = self._iteration(
                figures_by_type=figures_by_type,
                frame_index=frame_from,
                frames_count=frame_to - frame_from,
            )
            iteration_time = sly.TinyTimer.get_sec(tm)

            if self.cloud_token is not None and self.cloud_action_id is not None:
                actual_predictions = sum(
                    len(frame_predicted_figures)
                    for frame_predicted_figures in iteration_result.values()
                )
                try:
                    self.api.cloud.billing_withdrawal(
                        self.user_id,
                        items_count=actual_predictions,
                        transaction_id=transaction_id,
                        cloud_token=self.cloud_token,
                        cloud_action_id=self.cloud_action_id,
                    )
                except Exception as e:
                    self.logger.error("Unable to withdraw tokens for predictions", exc_info=True)
                    raise RuntimeError("Unable to withdraw tokens for predictions") from None

            # update timelines with predictions
            tm = sly.TinyTimer()
            for geom_type in iteration_result:
                for i, frame_predicted_figures in enumerate(iteration_result[geom_type]):
                    frame_index = frame_from + i + 1  # can be obtained from figure
                    for figure_index, predicted_figure in enumerate(frame_predicted_figures):
                        timeline: Track.Timeline = timelines_data[
                            timeline_index_by_type[geom_type][figure_index]
                        ][0]
                        timeline.add_predictions(frame_index, [predicted_figure])
            update_timelines_time = sly.TinyTimer.get_sec(tm)

            total_delta = sly.TinyTimer.get_sec(total_tm)
            self.logger.debug(
                "Iteration time",
                extra={
                    "total": f"{total_delta:.6f}  sec",
                    "get and apply updates": f"{apply_updates_time:.6f} sec",
                    "get batch data": f"{get_batch_time:.6f} sec",
                    "prediction": f"{iteration_time:.6f} sec",
                    "update timelines": f"{update_timelines_time:.6f} sec",
                },
            )


@send_error_data
def track(
    api: sly.Api,
    context: Dict,
    nn_settings: Dict,
    update_type: str = "track",
    cloud_token: str = None,
    cloud_action_id: str = None,
):
    sly.logger.debug("track", extra={"context": context, "nn_settings": nn_settings})

    if update_type == "delete":
        delete_data = {}
        track_id = context["trackId"]
        for figure in context["figures"]:
            object_id = figure["objectId"]
            frame = figure["frames"][0]
            delete_data.setdefault(object_id, []).append(frame)
        cur_track = g.current_tracks.get(track_id, None)
        if cur_track is None:
            return
        api.logger.info(
            "Delete figures", extra={"track_id": cur_track.track_id, "delete_data": delete_data}
        )
        for object_id, frames in delete_data.items():
            cur_track.put_update([object_id], min(frames), 0, update_type)
        return

    if update_type == "continue":
        track_id = context["trackId"]
        video_id = context["videoId"]
        cur_track = g.current_tracks.get(track_id, None)
        if cur_track is not None:
            api.logger.info("Continue tracking.", extra={"track_id": track_id})
            cur_track.put_update(
                [],
                cur_track.frame_index + cur_track.frames_count,
                cur_track.init_frames_count,
                update_type,
            )
            return
        else:
            api.logger.info("Track not found.", extra={"track_id": track_id})

    session_id = context.get("sessionId", None)
    if session_id is None:
        api.logger.warn("Session id is not provided. Some features may not work correctly.")
    track_id = context["trackId"]
    video_id = context["videoId"]
    figure_ids = list(context["figureIds"])
    object_ids = list(context["objectIds"])
    frame_index = context["frameIndex"]
    frames_count = context["frames"]
    user_id = api.user.get_my_info().id
    # direction = context["direction"]

    cur_track: Track = g.current_tracks.get(track_id, None)
    if cur_track is None or cur_track.frame_index + cur_track.frames_count < frame_index:
        api.retry_count = 1
        cur_track = Track(
            track_id=track_id,
            session_id=session_id,
            api=api,
            video_id=video_id,
            object_ids=object_ids,
            frame_index=frame_index,
            frames_count=frames_count,
            nn_settings=nn_settings,
            user_id=user_id,
            cloud_token=cloud_token,
            cloud_action_id=cloud_action_id,
        )
        api.logger.info("Start tracking.")
        g.current_tracks[track_id] = cur_track
        try:
            cur_track.run()
        finally:
            if not cur_track.global_stop_indicatior:
                cur_track.notify(stop=True, task="tracking completed")
            g.current_tracks.pop(track_id, None)
            api.logger.debug("Tracking completed.")
    elif update_type == "track":
        api.logger.info("Figure changed. Update tracking", extra={"track_id": track_id})
        cur_track.put_update(object_ids, frame_index, frames_count, update_type)
