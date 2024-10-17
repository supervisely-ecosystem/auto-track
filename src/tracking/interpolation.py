from functools import partial
import numpy as np
from typing import Dict, List
import uuid
from scipy.ndimage import distance_transform_edt
from skimage.transform import AffineTransform, warp

from supervisely.api.entity_annotation.figure_api import FigureInfo
import supervisely as sly
from supervisely.api.module_api import ApiField
from supervisely import logger
from supervisely.api.video.video_api import VideoInfo

import src.utils as utils
from scipy.ndimage import distance_transform_edt


INTERPOLATION_LIMIT = 200


def interpolate_box(
    this_bbox: sly.Rectangle,
    next_bbox: sly.Rectangle,
    from_frame: int,
    to_frame: int,
    video_info: VideoInfo,
) -> List[sly.Rectangle]:
    logger.debug("Interpolating box")
    n_frames = to_frame - from_frame
    rowdelta = (next_bbox.height - this_bbox.height) / (n_frames)
    coldelta = (next_bbox.width - this_bbox.width) / (n_frames)
    rowshift = (next_bbox.center.row - this_bbox.center.row) / (n_frames)
    colshift = (next_bbox.center.col - this_bbox.center.col) / (n_frames)
    created_geometries: List[sly.AnyGeometry] = []
    for frame_index in range(from_frame + 1, to_frame):
        i = frame_index - from_frame
        resized: sly.Rectangle = this_bbox.resize(
            in_size=(video_info.frame_height, video_info.frame_width),
            out_size=(
                int(video_info.frame_height * (1 + rowdelta * i / this_bbox.height)),
                int(video_info.frame_width * (1 + coldelta * i / this_bbox.width)),
            ),
        )
        target = int(this_bbox.center.row + i * rowshift), int(this_bbox.center.col + i * colshift)
        moved: sly.Rectangle = resized.translate(
            target[0] - resized.center.row, target[1] - resized.center.col
        )
        created_geometries.append(moved)
    return created_geometries


def morph_masks_gen(mask1, mask2, N):
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

    inner_masks = []
    for n in range(1, N + 1):
        t = n / (N + 1)
        # Interpolate bounding boxes
        bbox_n = (1 - t) * bbox1_cropped + t * bbox2_cropped

        # Compute transformations for mask1 and mask2 to align to bbox_n
        transform1 = get_affine_transform(bbox1_cropped, bbox_n)
        transform2 = get_affine_transform(bbox2_cropped, bbox_n)

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


def get_affine_transform(bbox_from, bbox_to):
    """
    Computes an affine transformation matrix that maps bbox_from to bbox_to.

    Parameters:
    - bbox_from: array-like, [x_min, y_min, x_max, y_max] of the source bounding box.
    - bbox_to: array-like, [x_min, y_min, x_max, y_max] of the target bounding box.

    Returns:
    - transform: skimage.transform.AffineTransform object
    """
    # Coordinates of the corners of the bounding boxes
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


def interpolate_bitmap(
    this_bitmap: sly.Bitmap,
    next_bitmap: sly.Bitmap,
    from_frame: int,
    to_frame: int,
    video_info: VideoInfo,
    notify_func=None,
) -> List[sly.Bitmap]:
    logger.debug("Interpolating bitmap")
    n_frames = to_frame - from_frame - 1
    created_geometries: List[sly.Bitmap] = []
    this_mask = this_bitmap.get_mask((video_info.frame_height, video_info.frame_width))
    next_mask = next_bitmap.get_mask((video_info.frame_height, video_info.frame_width))
    intermediate_masks = morph_masks_gen(this_mask, next_mask, n_frames)
    for mask in intermediate_masks:
        created_geometries.append(sly.Bitmap(mask))
        if notify_func is not None:
            notify_func()
    logger.debug("Done interpolating bitmap. Masks: %s", len(created_geometries))
    return created_geometries


@utils.send_error_data
def interpolate_frames(api: sly.Api, context: Dict):
    video_id = context["videoId"]
    figure_ids = context["figureIds"]
    track_id = context.get("trackId", None)
    video_info = api.video.get_info_by_id(video_id)
    dataset_id = video_info.dataset_id
    figures = api.video.figure.get_by_ids(dataset_id, figure_ids)
    from_frame = min([figure.frame_index for figure in figures])
    end_frame = min(from_frame + INTERPOLATION_LIMIT, video_info.frames_count)
    api.video.notify_progress(
        track_id, video_id, frame_start=from_frame, frame_end=from_frame + 1, current=0, total=1
    )

    all_figures: List[FigureInfo] = api.video.figure.get_list(
        dataset_id=dataset_id,
        filters=[
            {
                "field": "objectId",
                "operator": "in",
                "value": list(set([figure.object_id for figure in figures])),
            },
            {"field": "startFrame", "operator": ">=", "value": from_frame},
            {"field": "endFrame", "operator": "<=", "value": end_frame},
        ],
    )

    next_figures = []
    for i, this_figure in enumerate(figures):
        object_id = this_figure.object_id
        this_object_figures = [
            fig
            for fig in all_figures
            if fig.object_id == object_id and fig.frame_index > this_figure.frame_index
        ]
        if len(this_object_figures) == 0:
            continue
        next_figure = min(this_object_figures, key=lambda fig: fig.frame_index)
        next_figures.append(next_figure)

    end_frame = max([f.frame_index for f in next_figures])
    total = 0
    for this_figure, next_figure in zip(figures, next_figures):
        total += next_figure.frame_index - this_figure.frame_index
    api.video.notify_progress(
        track_id,
        video_id,
        frame_start=from_frame,
        frame_end=end_frame,
        current=0,
        total=total,
    )
    done = 0
    for i, this_figure, next_figure in zip(list(range(len(figures))), figures, next_figures):
        # interpolate between this_figure and next_figure
        this_geometry = sly.deserialize_geometry(this_figure.geometry_type, this_figure.geometry)
        next_geometry = sly.deserialize_geometry(next_figure.geometry_type, next_figure.geometry)

        if this_geometry.geometry_name() != next_geometry.geometry_name():
            logger.warning(
                f"Cannot interpolate between {this_geometry.geometry_name()} and {next_geometry.geometry_name()}"
            )
            continue

        if this_geometry.geometry_name() == sly.Rectangle.geometry_name():
            created_geometries = interpolate_box(
                this_geometry,
                next_geometry,
                this_figure.frame_index,
                next_figure.frame_index,
                video_info,
            )
        elif this_geometry.geometry_name() == sly.Bitmap.geometry_name():
            current = done

            def _notify_func():
                nonlocal current
                current += 1
                api.video.notify_progress(
                    track_id,
                    video_id,
                    frame_start=from_frame,
                    frame_end=end_frame,
                    current=current,
                    total=total,
                )

            created_geometries = interpolate_bitmap(
                this_geometry,
                next_geometry,
                this_figure.frame_index,
                next_figure.frame_index,
                video_info,
                notify_func=_notify_func,
            )
        else:
            logger.warning(f"Unsupported geometry type: {this_geometry.geometry_name()}")
            continue

        figures_json = [
            {
                ApiField.OBJECT_ID: object_id,
                ApiField.GEOMETRY_TYPE: geom.geometry_name(),
                ApiField.GEOMETRY: geom.to_json(),
                ApiField.META: {ApiField.FRAME: from_frame + i + 1},
                ApiField.TRACK_ID: track_id,
            }
            for i, geom in enumerate(created_geometries)
        ]
        figures_keys = [uuid.uuid4() for _ in figures_json]
        key_id_map = sly.KeyIdMap()
        # pylint: disable=protected-access
        api.video.figure._append_bulk(
            entity_id=video_id,
            figures_json=figures_json,
            figures_keys=figures_keys,
            key_id_map=key_id_map,
        )
        done += next_figure.frame_index - this_figure.frame_index
        api.video.notify_progress(
            track_id,
            video_id,
            frame_start=from_frame,
            frame_end=end_frame,
            current=done,
            total=total,
        )
    api.video.notify_progress(
        track_id,
        video_id,
        frame_start=from_frame,
        frame_end=end_frame,
        current=total,
        total=total,
    )
