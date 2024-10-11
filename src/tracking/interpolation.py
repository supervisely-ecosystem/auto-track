from typing import Dict, List
import uuid

from supervisely.api.entity_annotation.figure_api import FigureInfo
import supervisely as sly
from supervisely.api.module_api import ApiField
from supervisely import logger


INTERPOLATION_LIMIT = 200


def interpolate_frames(api: sly.Api, context: Dict):
    video_id = context["videoId"]
    figure_ids = context["figureIds"]
    track_id = context.get("trackId", None)
    video_info = api.video.get_info_by_id(video_id)
    dataset_id = video_info.dataset_id
    figures = api.video.figure.get_by_ids(dataset_id, figure_ids)
    from_frame = min([figure.frame_index for figure in figures])
    end_frame = min(from_frame + INTERPOLATION_LIMIT, video_info.frames_count)

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
        n_frames = next_figure.frame_index - this_figure.frame_index

        this_bbox: sly.Rectangle = this_geometry.to_bbox()
        next_bbox: sly.Rectangle = next_geometry.to_bbox()
        rowdelta = (next_bbox.height - this_bbox.height) / (n_frames - 1)
        coldelta = (next_bbox.width - this_bbox.width) / (n_frames - 1)
        rowshift = (next_bbox.center.row - this_bbox.center.row) / (n_frames - 1)
        colshift = (next_bbox.center.col - this_bbox.center.col) / (n_frames - 1)

        logger.debug(
            "Interpolating between frames %d and %d",
            this_figure.frame_index,
            next_figure.frame_index,
        )
        logger.debug("Rowdelta: %f", rowdelta)
        logger.debug("Coldelta: %f", coldelta)
        logger.debug("Rowshift: %f", rowshift)
        logger.debug("Colshift: %f", colshift)
        logger.debug("Center: %s", (this_bbox.center.row, this_bbox.center.col))
        logger.debug(
            "height this: %f, height next: %f, diff: %f",
            this_bbox.height,
            next_bbox.height,
            next_bbox.height - this_bbox.height,
        )
        logger.debug(
            "width this: %f, width next: %f, diff: %f",
            this_bbox.width,
            next_bbox.width,
            next_bbox.width - this_bbox.width,
        )

        created_geometries: List[sly.AnyGeometry] = []
        for frame_index in range(this_figure.frame_index + 1, next_figure.frame_index):
            logger.debug("Interpolating frame %d", frame_index)
            i = frame_index - this_figure.frame_index
            logger.debug(
                "i: %d, shift: (%d, %d), delta: (%d, %d)",
                i,
                int(rowshift * i),
                int(colshift * i),
                int(i * rowdelta),
                int(i * coldelta),
            )
            moved: sly.AnyGeometry = this_geometry.translate(int(rowshift * i), int(colshift * i))
            logger.debug("Moved: %s", (moved.to_bbox().center.row, moved.to_bbox().center.col))
            resized: sly.AnyGeometry = moved.resize(
                in_size=(video_info.frame_height, video_info.frame_width),
                out_size=(
                    int(video_info.frame_height + (i * rowdelta)),
                    int(video_info.frame_width + (i * coldelta)),
                ),
            )
            logger.debug(
                "Resized: %s", (resized.to_bbox().center.row, resized.to_bbox().center.col)
            )
            created_geometries.append(resized)

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
