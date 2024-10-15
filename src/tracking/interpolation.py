import numpy as np
from scipy import ndimage
from typing import Dict, List
import uuid

from supervisely.api.entity_annotation.figure_api import FigureInfo
import supervisely as sly
from supervisely.api.module_api import ApiField
from supervisely import logger
from supervisely.api.video.video_api import VideoInfo


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


def morph_masks(initial_mask, final_mask, num_iterations):
    # Ensure masks are binary and of the same shape
    assert initial_mask.shape == final_mask.shape, "Masks must be the same shape"
    initial_mask = initial_mask.astype(bool)
    final_mask = final_mask.astype(bool)

    # Compute pixels to add and remove
    to_add = np.logical_and(final_mask, np.logical_not(initial_mask))
    to_remove = np.logical_and(initial_mask, np.logical_not(final_mask))

    # Compute distance transforms
    # For adding pixels: distance from initial mask
    distance_to_initial = ndimage.distance_transform_edt(np.logical_not(initial_mask))
    add_distances = distance_to_initial[to_add]

    # For removing pixels: distance from final mask
    distance_to_final = ndimage.distance_transform_edt(np.logical_not(final_mask))
    remove_distances = distance_to_final[to_remove]

    # Normalize distances and assign time steps
    if add_distances.size > 0:
        add_time_steps = (add_distances / add_distances.max() * (num_iterations - 1)).astype(int)
    else:
        add_time_steps = np.array([], dtype=int)

    if remove_distances.size > 0:
        remove_time_steps = (
            remove_distances / remove_distances.max() * (num_iterations - 1)
        ).astype(int)
    else:
        remove_time_steps = np.array([], dtype=int)

    # Get indices of pixels to add and remove
    add_indices = np.argwhere(to_add)
    remove_indices = np.argwhere(to_remove)

    # Prepare a list to hold all masks
    masks = []

    for t in range(num_iterations):
        mask_t = initial_mask.copy()

        # Add pixels scheduled to be added at or before time t
        if add_indices.size > 0:
            indices_to_add = add_indices[add_time_steps <= t]
            mask_t[tuple(indices_to_add.T)] = True

        # Remove pixels scheduled to be removed at or before time t
        if remove_indices.size > 0:
            indices_to_remove = remove_indices[remove_time_steps <= t]
            mask_t[tuple(indices_to_remove.T)] = False

        masks.append(mask_t.astype(np.uint8))

    return masks


def interpolate_bitmap(
    this_bitmap: sly.Bitmap,
    next_bitmap: sly.Bitmap,
    from_frame: int,
    to_frame: int,
    video_info: VideoInfo,
) -> List[sly.Bitmap]:
    logger.debug("Interpolating bitmap")
    n_frames = to_frame - from_frame
    created_geometries: List[sly.Bitmap] = []
    this_mask = this_bitmap.get_mask(video_info.frame_height, video_info.frame_width)
    next_mask = next_bitmap.get_mask(video_info.frame_height, video_info.frame_width)
    intermediate_masks = morph_masks(this_mask, next_mask, n_frames)
    for mask in enumerate(intermediate_masks):
        created_geometries.append(sly.Bitmap(mask))
    return created_geometries


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
        track_id, video_id, frame_start=from_frame, frame_end=end_frame, current=0, total=1
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

    for this_figure in figures:
        api.video.notify_progress(
            track_id, video_id, frame_start=from_frame, frame_end=end_frame, current=0, total=1
        )
        object_id = this_figure.object_id
        this_object_figures = [
            fig
            for fig in all_figures
            if fig.object_id == object_id and fig.frame_index > this_figure.frame_index
        ]
        if len(this_object_figures) == 0:
            continue
        next_figure = min(this_object_figures, key=lambda fig: fig.frame_index)
        # interpolate between this_figure and next_figure
        this_geometry = sly.deserialize_geometry(this_figure.geometry_type, this_figure.geometry)
        next_geometry = sly.deserialize_geometry(next_figure.geometry_type, next_figure.geometry)

        if this_geometry.geometry_name() != next_geometry.geometry_name():
            logger.warn(
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
        # elif this_geometry.geometry_name() == sly.Bitmap.geometry_name():
        #     created_geometries = interpolate_bitmap(
        #         this_geometry,
        #         next_geometry,
        #         this_figure.frame_index,
        #         next_figure.frame_index,
        #         video_info,
        #     )
        else:
            raise ValueError(f"Unsupported geometry type: {this_geometry.geometry_name()}")

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
        api.video.notify_progress(
            track_id,
            video_id,
            frame_start=from_frame,
            frame_end=next_figure.frame_index,
            current=1,
            total=1,
        )
