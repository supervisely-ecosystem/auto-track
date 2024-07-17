from collections import namedtuple
from typing import TYPE_CHECKING, List, NamedTuple, Tuple

from supervisely.api.entity_annotation.figure_api import FigureInfo

if TYPE_CHECKING:
    from src.tracking.track import Track


class Timeline:

    def __init__(
        self,
        track: "Track",
        object_id: int,
        object_info: NamedTuple,
        start_frame: int,
        end_frame: int,
        all_object_figures: List[FigureInfo],
        no_object_tag_meta: NamedTuple = None,
    ) -> None:
        self.track = track
        self.object_id = object_id
        self.object_info = object_info
        self.start_frame = start_frame
        self.start_figures = []
        self.end_frame = end_frame

        self._init(
            object_info=object_info,
            all_object_figures=all_object_figures,
            no_object_tag_meta=no_object_tag_meta,
        )

        self.last_tracked_figures = (start_frame, self.start_figures)  # frame_index, FigureInfo

    def _init(
        self,
        object_info: NamedTuple,
        all_object_figures: List[FigureInfo],
        no_object_tag_meta: None,
    ) -> None:

        for figure in all_object_figures:
            self.end_frame = max(figure.frame_index)

    def delete_figures(self):
        pass

    def continue_tracking(self, frame_index: int, frames_count: int):
        """Continue Timeline from frame_index for frames_count"""
        end_frame = min(frame_index + frames_count, self.track.video_info.frames_count)
        self.end_frame = max(self.end_frame, end_frame)
        self.delete_figures()

    def cut(self, frame_index: int):
        """Cut Timeline from frame_index"""
        self.end_frame = frame_index
        self.delete_figures()  # TODO: maybe not needed

    def get_batch(self, batch_size: int) -> Tuple[int, int, List[FigureInfo]]:
        """Get batch of frames to track"""
        last_tracked_frame, figures = self.last_tracked_figures
        batch_end_frame = min(last_tracked_frame + batch_size, self.end_frame)
        return last_tracked_frame, batch_end_frame, figures

    def update(self, frame_index: int, figures: List[FigureInfo]):
        self.last_tracked_figures = (frame_index, figures)

    def log_data(self):
        return {
            "object_id": self.object_id,
            "start_frame_index": self.start_frame,
            "end_frame_index": self.end_frame,
            "start_frame_figures": [figure.id for figure in self.start_figures],
            # "progress": {
            #     "frame_index": self.progress.get("frame_index", None),
            #     "figures": [f.id for f in self.progress.get("figures", [])],
            # },
            # "can_continue": self.can_continue,
        }
