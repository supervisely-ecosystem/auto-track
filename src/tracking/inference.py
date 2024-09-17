from typing import Dict, List
import uuid
import requests
from concurrent.futures import ThreadPoolExecutor

import supervisely as sly
from supervisely.api.entity_annotation.figure_api import FigureInfo

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
    response = api.task.send_request(task_id, "track-api", {}, context=data, retries=1)

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


def get_detections(api: sly.Api, nn_settings: Dict, video_id: int, frame_from, frame_to, conf=0.7):
    if "task_id" in nn_settings:
        session = sly.nn.inference.Session(
            api, nn_settings["task_id"], inference_settings={"conf": conf}
        )
        detections = session.inference_video_id(video_id, frame_from, frame_to - frame_from)
        return detections

    elif "url" in nn_settings:
        return []
