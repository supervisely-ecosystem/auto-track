from typing import Dict, Generator, List, Tuple
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

import src.utils as utils


INTERPOLATION_FRAMES_LIMIT = 200
MIN_GEOMETRIES_BATCH_SIZE = 10


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
        interior = cv2.approxPolyDP(
            polygon.interior_np.reshape((-1, 1, 2)), epsilon=epsilon, closed=True
        ).reshape(-1, 2)
    return sly.Polygon(
        exterior=[sly.PointLocation(x, y) for x, y in exterior],
        interior=[sly.PointLocation(x, y) for x, y in interior],
    )


def unsupported_geometry_interpolator(this_geom: Geometry, **kwargs):
    msg = f"Interpolation for {this_geom.name()} is not supported."
    raise NotImplementedError(msg)


def interpolate_box(
    this_geom: sly.Rectangle,
    dest_geom: sly.Rectangle,
    from_frame: int,
    to_frame: int,
    video_info: VideoInfo,
) -> List[sly.Rectangle]:
    logger.debug("Interpolating box")
    n_frames = to_frame - from_frame
    rowdelta = (dest_geom.height - this_geom.height) / (n_frames)
    coldelta = (dest_geom.width - this_geom.width) / (n_frames)
    rowshift = (dest_geom.center.row - this_geom.center.row) / (n_frames)
    colshift = (dest_geom.center.col - this_geom.center.col) / (n_frames)
    created_geometries: List[sly.AnyGeometry] = []
    for frame_index in range(from_frame + 1, to_frame):
        i = frame_index - from_frame
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
        created_geometries.append(moved)
    return created_geometries


def interpolate_bitmap(
    this_geom: sly.Bitmap,
    dest_geom: sly.Bitmap,
    from_frame: int,
    to_frame: int,
    video_info: VideoInfo,
) -> Generator[List[sly.Bitmap], None, None]:
    logger.debug("Interpolating bitmap")
    n_frames = to_frame - from_frame - 1
    this_mask = this_geom.get_mask((video_info.frame_height, video_info.frame_width))
    next_mask = dest_geom.get_mask((video_info.frame_height, video_info.frame_width))
    for mask in _morph_masks_gen(this_mask, next_mask, n_frames):
        yield [sly.Bitmap(mask)]
    logger.debug("Done interpolating bitmap")


def interpolate_polygon(
    this_geom: sly.Polygon,
    dest_geom: sly.Polygon,
    from_frame: int,
    to_frame: int,
    video_info: VideoInfo,
) -> Generator[List[List[sly.Polygon]], None, None]:
    logger.debug("Interpolating polygon")
    n_frames = to_frame - from_frame - 1
    this_mask = this_geom.get_mask((video_info.frame_height, video_info.frame_width))
    next_mask = dest_geom.get_mask((video_info.frame_height, video_info.frame_width))
    for mask in _morph_masks_gen(this_mask, next_mask, n_frames):
        polys = sly.Bitmap(mask).to_contours()
        polys = [_simplify_polygon(poly) for poly in polys]
        yield [polys]
    logger.debug("Done interpolating polygon")


def interpolate_line(
    this_geom: sly.Polyline,
    dest_geom: sly.Polyline,
    from_frame: int,
    to_frame: int,
    video_info: VideoInfo,
) -> List[sly.Polyline]:
    logger.debug("Interpolating line")
    n_frames = to_frame - from_frame - 1
    created_geometries: List[sly.Polyline] = []
    if len(this_geom.exterior) != len(dest_geom.exterior):
        logger.warning("Cannot interpolate lines with different number of points")
        return []
    for i in range(n_frames):
        t = i / n_frames
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
    from_frame: int,
    to_frame: int,
    video_info: VideoInfo,
) -> List[sly.Point]:
    logger.debug("Interpolating point")
    n_frames = to_frame - from_frame - 1
    created_geometries: List[sly.Point] = []
    for i in range(n_frames):
        t = i / n_frames
        x = int(this_geom.row * (1 - t) + dest_geom.row * t)
        y = int(this_geom.col * (1 - t) + dest_geom.col * t)
        created_geometries.append(sly.Point(row=x, col=y))
    logger.debug("Done interpolating point")
    return created_geometries


INTERPOLATORS = {
    sly.Rectangle.name(): interpolate_box,
    sly.Bitmap.name(): interpolate_bitmap,
    sly.Polygon.name(): interpolate_polygon,
    sly.Polyline.name(): interpolate_line,
    sly.Point.name(): interpolate_point,
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
        self.frame_end = min(
            self.frame_start + self.INTERPOLATION_FRAMES_LIMIT, self.video_info.frames_count
        )

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
        utils.notify_warning(self.api, self.track_id, message)

    def find_destination_figures(self) -> List[FigureInfo]:
        all_figures: List[FigureInfo] = self.api.video.figure.get_list(
            dataset_id=self.dataset_id,
            filters=[
                {
                    "field": "objectId",
                    "operator": "in",
                    "value": list(set([figure.object_id for figure in self.figures])),
                },
                {"field": "startFrame", "operator": ">", "value": self.frame_start + 1},
                {"field": "endFrame", "operator": "<=", "value": self.frame_end},
            ],
        )

        next_figures = []
        for this_figure in self.figures:
            object_id = this_figure.object_id
            this_object_figures = [
                fig
                for fig in all_figures
                if fig.object_id == object_id and fig.frame_index > this_figure.frame_index + 1
            ]
            if len(this_object_figures) == 0:
                next_figures.append(None)
            next_figure = min(this_object_figures, key=lambda fig: fig.frame_index)
            next_figures.append(next_figure)
        return next_figures

    def filter_destination_figures(
        self, dest_figures: List[FigureInfo]
    ) -> Tuple[List[FigureInfo], List[FigureInfo]]:
        figures, dest_figures = filter_figures(
            self.figures, dest_figures, condition=lambda x: x[1] is not None
        )
        if len(figures) < len(self.figures):
            msg = f"Destination figures not found for {len(self.figures) - len(figures)} figures. Such figures will be skipped"
            logger.info(msg, extra=self.log_extra)
            self.send_warning(msg)

        original_figures_len = len(figures)
        figures, dest_figures = filter_figures(
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
            logger.warning(message, extra=self.log_extra)
            self.send_warning(message)

    def interpolate_frames(self):
        self.notify_progress()
        dest_figures: List[FigureInfo] = self.find_destination_figures()
        figures, dest_figures = self.filter_destination_figures(dest_figures)
        self.frame_end = max([figure.frame_index for figure in dest_figures]) - 1
        self.progress_total = sum(
            [
                dest_figure.frame_index - this_figure.frame_index - 1
                for this_figure, dest_figure in zip(figures, dest_figures)
            ]
        )
        if len(figures) == 0:
            msg = "No valid figures to interpolate"
            logger.info(msg, extra=self.log_extra)
            utils.notify_error(self.api, self.track_id, msg)
            return
        for this_figure, dest_figure in zip(figures, dest_figures):
            # interpolate between this_figure and next_figure
            this_geometry = sly.deserialize_geometry(
                this_figure.geometry_type, this_figure.geometry
            )
            dest_geometry = sly.deserialize_geometry(
                dest_figure.geometry_type, dest_figure.geometry
            )
            frames_n = dest_figure.frame_index - this_figure.frame_index - 1
            before_this_figure_progress = self.progress_current

            if this_geometry.name() != dest_geometry.name():
                msg = (
                    f"Cannot interpolate between {this_geometry.name()} and {dest_geometry.name()}"
                )
                logger.info(msg, extra=self.log_extra)
                self.send_warning(msg)
                self.progress_current += frames_n
                continue

            interpolator_func = INTERPOLATORS.get(
                this_geometry.name(), unsupported_geometry_interpolator
            )
            try:
                frame_index = this_figure.frame_index + 1
                cum_batch = []
                for batch in interpolator_func():
                    cum_batch.extend(batch)
                    if len(cum_batch) < MIN_GEOMETRIES_BATCH_SIZE:
                        continue
                    frame_indexes = list(range(frame_index, frame_index + len(cum_batch)))
                    frame_index = frame_indexes[-1] + 1
                    self.upload(this_figure.object_id, cum_batch, frame_indexes)
                    self.progress_current += len(cum_batch)
                    self.notify_progress()
                if len(cum_batch) > 0:
                    frame_indexes = list(range(frame_index, frame_index + len(cum_batch)))
                    self.upload(this_figure.object_id, cum_batch, frame_indexes)
                    self.progress_current += len(cum_batch)
                    self.notify_progress()
            except NotImplementedError:
                msg = f"Interpolation for geometry {this_geometry.name()} is not supported yet. Skipping figure"
                logger.info(msg, extra=self.log_extra)
                self.send_warning(msg)
                self.progress_current += frames_n
            except Exception as e:
                msg = f"Unexpected Error during interpolation: {str(e)}"
                logger.warning(msg, extra=self.log_extra)
                self.send_warning(msg)
                this_figure_progress = before_this_figure_progress - self.progress_current
                self.progress_current += frames_n - this_figure_progress
            finally:
                self.notify_progress()
        self.progress_current = self.progress_total
        self.notify_progress()


@utils.send_error_data
def interpolate_frames(api, context):
    Interpolator(api, context).interpolate_frames()
