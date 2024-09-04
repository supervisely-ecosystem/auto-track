import ast
from typing import List, Dict, Literal, NamedTuple, Set, Tuple
from concurrent.futures import ThreadPoolExecutor
from logging import Logger
import queue
import threading
import time
import uuid

import requests
import supervisely as sly
from supervisely.api.entity_annotation.figure_api import FigureInfo
from supervisely.api.module_api import ApiField
from supervisely import TinyTimer

import src.globals as g
import src.utils as utils
import src.tracking.inference as inference


def validate_nn_settings_for_geometry(
    nn_settings: Dict, geometry_name: str, raise_error: bool = True, logger: Logger = None
) -> bool:
    geoms_to_validate = [geometry_name]
    if geometry_name == g.GEOMETRY_NAME.SMARTTOOL:
        geoms_to_validate.extend([g.GEOMETRY_NAME.RECTANGLE, g.GEOMETRY_NAME.POINT])
    invalid = []
    for geom in geoms_to_validate:
        if geom not in nn_settings:
            invalid.append(geom)
        elif "task_id" in nn_settings[geom] and nn_settings[geom].get("task_id", None) is None:
            invalid.append(geom)
    if len(invalid) > 0:
        if raise_error:
            raise ValueError(f"NN settings for {', '.join(invalid)} are not specified")
        if logger is not None:
            logger.warning(f"NN settings for {', '.join(invalid)} are not specified")
        return False
    return True


def find_key_figures(figures: List[FigureInfo]) -> Dict[int, List[FigureInfo]]:
    # find key figures and figures with the same track_id to delete it
    key_figures = {}
    for figure in figures:
        track_id = figure.track_id
        if track_id is None:
            key_figures.setdefault(figure.frame_index, []).append(figure)
    return key_figures


class Tracklet:
    def __init__(
        self, timeline: "Timeline", start_frame: int, end_frame: int, figures: List[FigureInfo]
    ):
        self.timeline = timeline
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.last_tracked = (start_frame, figures)
        self.area_hist = {}

    def continue_tracklet(self, frame_to: int):
        if frame_to <= self.end_frame:
            return
        end_frame = self.end_frame
        for i in range(
            self.end_frame + 1,
            frame_to + 1,
        ):
            if i in self.timeline.key_figures:
                break
            if i in self.timeline.no_object_frames:
                break
            end_frame = i
        self.end_frame = end_frame

    def update(self, frame_index: int, figures: List[FigureInfo], stop=False):
        if stop:
            self.end_frame = frame_index - 1
            return
        if len(figures) == 0:
            return

        self.area_hist[frame_index] = sum(
            [utils.maybe_literal_eval(figure.area) for figure in figures]
        )
        self.last_tracked = (frame_index, figures)

    def median_area(self):
        return sorted(self.area_hist.values())[len(self.area_hist) // 2]

    def cut(self, frame_index: int, remove_added_figures: bool = False):
        if frame_index < self.start_frame:
            return
        if frame_index > self.end_frame:
            return
        if remove_added_figures:
            self.clear(from_frame=frame_index, to_frame=self.end_frame)
        self.end_frame = frame_index - 1

    def clear(self, from_frame: int = None, to_frame: int = None):
        if from_frame is None:
            from_frame = self.start_frame
        if to_frame is None:
            to_frame = self.end_frame
        threading.Thread(
            target=self.timeline.track.remove_predicted_figures_by_frame_range,
            args=([self.timeline.object_id], (from_frame, to_frame)),
        ).start()

    def get_batch(self, batch_size: int = None) -> Tuple[int, int, List[FigureInfo]]:
        if batch_size is None:
            batch_size = self.timeline.track.batch_size
        frame_from, figures = self.last_tracked
        frame_to = min(frame_from + batch_size, self.end_frame)
        return frame_from, frame_to, figures

    def is_finished(self):
        return self.last_tracked[0] >= self.end_frame

    def get_progress_total(self):
        return (self.end_frame - self.start_frame) * len(self.last_tracked[1])

    def get_progress_current(self):
        return (min(self.last_tracked[0], self.end_frame) - self.start_frame) * len(
            self.last_tracked[1]
        )

    def log_data(self):
        return {
            "frame_range": [self.start_frame, self.end_frame],
            "last_tracked_frame": self.last_tracked[0],
            "progress": f"{self.get_progress_current()} / {self.get_progress_total()}",
        }


class Timeline:
    def __init__(
        self,
        track: "Track",
        object_id: int,
        start_frame: int,
        end_frame: int,
        object_info: NamedTuple = None,
        figures: List[FigureInfo] = None,
    ) -> None:
        self.track = track
        self.object_id = object_id

        self.object_info: NamedTuple = object_info
        self.update_object_info(self.object_info)
        if figures is None:
            figures = self.track.api.video.figure.get_list(
                dataset_id=self.track.dataset_id,
                filters=[
                    {"field": "objectId", "operator": "=", "value": self.object_id},
                    {"field": "startFrame", "operator": ">=", "value": start_frame},
                    {"field": "endFrame", "operator": "<=", "value": end_frame},
                ],
            )
        self.key_figures: Dict[int, List[FigureInfo]] = find_key_figures(figures)
        self.no_object_frames: Set[int] = set()
        self.update_no_object_frames()
        start_figures = [figure for figure in figures if figure.frame_index == start_frame]

        self.tracklets: List[Tracklet] = []
        self.tracklets.append(
            Tracklet(
                self,
                start_frame,
                self._find_end_frame(start_frame, end_frame - start_frame),
                start_figures,
            )
        )

    def _find_end_frame(self, start_frame: int, frames_count) -> int:
        end_frame = start_frame
        for i in range(start_frame + 1, start_frame + frames_count + 1):
            if i in self.key_figures:
                break
            if i in self.no_object_frames:
                break
            end_frame = i
        return end_frame

    def update_object_info(self, object_info=None):
        if object_info is None:
            object_info = self.track.api.video.object.get_info_by_id(self.object_id)
        self.object_info = object_info

    def update_no_object_frames(self):
        if self.object_info is None:
            self.update_object_info()
        no_object_frames = set()
        for tag in self.object_info.tags:
            if tag["tagId"] in self.track.no_object_tag_ids:
                range_from, range_to = tag["frameRange"]
                for i in range(range_from, range_to + 1):
                    no_object_frames.add(i)
        self.no_object_frames = no_object_frames

    def get_batch(self, batch_size: int) -> Tuple[int, int, List[FigureInfo]]:
        """
        Get batch of frames to track
        Returns (frame_from, frame_to, figures)
        """
        for tracklet in self.tracklets:
            if tracklet.is_finished():
                continue
            return tracklet.get_batch(batch_size)
        return None, None, None  # all tracklets are finished

    def object_changed(self, frame_index: int, frames_count: int):
        """One of the figures was changed or new figure was added"""
        frame_figures = self.track.api.video.figure.get_list(
            dataset_id=self.track.dataset_id,
            filters=[
                {"field": "objectId", "operator": "=", "value": self.object_id},
                {"field": "startFrame", "operator": ">=", "value": frame_index},
                {"field": "endFrame", "operator": "<=", "value": frame_index},
            ],
        )
        key_frame_figures = find_key_figures(frame_figures)
        self.key_figures.update(key_frame_figures)

        for i, tracklet in enumerate(self.tracklets):
            if frame_index == tracklet.start_frame:
                # reset tracklet
                tracklet.update(frame_index, frame_figures)
                return
            if tracklet.start_frame < frame_index <= tracklet.end_frame:
                # split tracklet
                new_tracklet_end_frame = self._find_end_frame(frame_index, frames_count)
                new_tracklet = Tracklet(self, frame_index, new_tracklet_end_frame, frame_figures)
                self.tracklets.insert(i + 1, new_tracklet)
                tracklet.cut(frame_index, remove_added_figures=True)
                return
        # no intersections - insert new tracklet
        new_tracklet_end_frame = self._find_end_frame(frame_index, frames_count)
        new_tracklet = Tracklet(self, frame_index, new_tracklet_end_frame, frame_figures)
        self.tracklets.append(new_tracklet)
        self.tracklets.sort(key=lambda x: x.start_frame)
        self.track.logger.debug(
            "Timeline tracklets after update:",
            extra={**self.track.logger_extra, **self.log_data()},
        )

    def continue_timeline(self, frame_index: int, frames_count: int):
        figures = self.track.api.video.figure.get_list(
            dataset_id=self.track.dataset_id,
            filters=[
                {"field": "objectId", "operator": "=", "value": self.object_id},
                {"field": "startFrame", "operator": ">=", "value": frame_index},
                {"field": "endFrame", "operator": "<=", "value": frame_index + frames_count},
            ],
        )
        key_frame_figures = find_key_figures(figures)
        if len(key_frame_figures.get(frame_index, [])) > 0:
            self.key_figures.update(key_frame_figures)

        for tracklet in self.tracklets:
            if tracklet.start_frame <= frame_index <= tracklet.end_frame:
                tracklet.continue_tracklet(frame_index + frames_count)

    def object_removed(self, frame_index: int, frames_count: int):
        for frame_index in range(frame_index, frame_index + frames_count + 1):
            self.no_object_frames.add(frame_index)
            if frame_index in self.key_figures:
                del self.key_figures[frame_index]
            tracklets_to_remove = []
            for tracklet in self.tracklets:
                if tracklet.start_frame == frame_index:
                    tracklets_to_remove.append(frame_index)
                if tracklet.start_frame < frame_index <= tracklet.end_frame:
                    tracklet.cut(frame_index, remove_added_figures=False)
            for frame_index in tracklets_to_remove:
                self.remove_tracklet(frame_index, clear=False)
        # remove all figures on selected frames
        threading.Thread(
            target=self.track.remove_predicted_figures_by_frame_range,
            args=([self.object_id], (frame_index, frame_index + frames_count)),
        ).start()

    def no_object_tag_removed(self, frame_index: int, frame_range: Tuple[int, int]):
        self.update_object_info()
        self.update_no_object_frames()

        frame_figures = None

        for tracklet in self.tracklets:
            if frame_index == tracklet.end_frame + 1:
                tracklet.continue_tracklet(frame_range[1])
                if tracklet.last_tracked[0] >= frame_index:
                    if frame_figures is None:
                        frame_figures = self.track.api.video.figure.get_list(
                            dataset_id=self.track.dataset_id,
                            filters=[
                                {"field": "objectId", "operator": "=", "value": self.object_id},
                                {"field": "startFrame", "operator": ">=", "value": frame_index},
                                {"field": "endFrame", "operator": "<=", "value": frame_index},
                            ],
                        )
                    tracklet.last_tracked = (frame_index, frame_figures)

    def manual_figure_removed(self, frame_index: int, end_frame_for_current_range: int):
        frame_figures = self.track.api.video.figure.get_list(
            dataset_id=self.track.dataset_id,
            filters=[
                {"field": "objectId", "operator": "=", "value": self.object_id},
                {"field": "startFrame", "operator": ">=", "value": frame_index},
                {"field": "endFrame", "operator": "<=", "value": frame_index},
            ],
        )
        key_frame_figures = find_key_figures(frame_figures)
        self.key_figures.update(key_frame_figures)
        if len(key_frame_figures.get(frame_index, None)) == 0:
            del self.key_figures[frame_index]

        tracklets_to_remove = []
        for tracklet in self.tracklets:
            if tracklet.start_frame == frame_index:
                if len(self.key_figures.get(frame_index, [])) == 0:
                    tracklets_to_remove.append(frame_index)
                else:
                    self.track.prevent_object_upload(
                        self.object_id, (frame_index, (frame_index, tracklet.end_frame))
                    )
                    tracklet.update(frame_index, self.key_figures.get(frame_index, []))
                    tracklet.clear(from_frame=frame_index + 1)
            elif tracklet.end_frame + 1 == frame_index:
                tracklet.continue_tracklet(end_frame_for_current_range)

        for i in tracklets_to_remove:
            self.track.prevent_object_upload(
                self.object_id, (frame_index, self.tracklets[i].end_frame)
            )
            self.remove_tracklet(i, clear=True)

    def remove_tracklet(self, frame_index: int, clear=False):
        if clear:
            for tracklet in self.tracklets:
                if tracklet.start_frame == frame_index:
                    tracklet.clear()
        self.tracklets = [
            tracklet for tracklet in self.tracklets if tracklet.start_frame != frame_index
        ]

    def update(self, frame_from: int, frame_to: int, predictions: List[List[FigureInfo]]):
        self.track.logger.debug("Update timeline", extra={"timeline": self.log_data(), "frame_from": frame_from, "frame_to": frame_to})
        for tracklet in self.tracklets:
            if tracklet.start_frame <= frame_from <= tracklet.end_frame:
                for frame_index in range(frame_from + 1, frame_to + 1):
                    figures = predictions.pop(0)
                    if len(figures) == 0:  # objects dissapear
                        tracklet.update(frame_index, figures, stop=True)
                    else:
                        tracklet.update(frame_index, figures)
                return

    def get_progress_total(self):
        total = 0
        for tracklet in self.tracklets:
            total += tracklet.get_progress_total()
        return total

    def get_progress_current(self):
        current = 0
        for tracklet in self.tracklets:
            current += tracklet.get_progress_current()
        return current

    def log_data(self):
        return {
            "object_id": self.object_id,
            "progress": f"{self.get_progress_current()} / {self.get_progress_total()}",
            "tracklets": [tracklet.log_data() for tracklet in self.tracklets],
        }


class Progress:
    def __init__(self, track: "Track"):
        self.track = track
        self.total = 0  # figures
        self.current = 0  # figures
        self.frame_range = (0, 0)

    def update(self, count: int, notify: bool = True):
        self.current += count
        if notify:
            self.notify()

    def notify(self, stop=False):
        pos = self.current
        if stop:
            pos = self.total

        self.track.logger.info(
            "Progress: %s/%s. Frame range: %s - %s",
            pos,
            self.total,
            self.frame_range[0] + 1,
            self.frame_range[1],
            extra=self.track.logger_extra,
        )

        global_stop_indicator = self.track.api.video.notify_progress(
            self.track.track_id,
            self.track.video_id,
            self.frame_range[0] + 1,
            self.frame_range[1],
            pos,
            self.total,
        )
        if not stop and global_stop_indicator and self.current < self.total:
            self.track.global_stop_indicator = True


class Update:
    class Type:
        # if updated, update typehint in __init__
        TRACK = "track"
        CONTINUE = "continue"
        DELETE = "delete"
        REMOVE_TAG = "remove_tag"
        MANUAL_OBJECTS_REMOVED = "manual_objects_removed"

    def __init__(
        self,
        object_ids: List[int],
        frame_index: int,
        frames_count: int,
        type: Literal["track", "continue", "delete"] = "track",
        tag_id: int = None,
    ):
        self.object_ids = object_ids
        self.frame_index = frame_index
        self.frames_count = frames_count
        self.type = type
        self.tag_id = tag_id

    def __repr__(self):
        return f'Update "{self.type}": Object IDs: {self.object_ids}, Frame index: {self.frame_index}, Frames count: {self.frames_count}'


class Track:
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
        self.api = api
        self.session_id = session_id
        if session_id is not None:
            self.api.headers.update({"x-toolbox-session-id": session_id})
        self.video_id = video_id
        self.video_info = api.video.get_info_by_id(video_id)
        self.dataset_id = self.video_info.dataset_id
        self.project_id = self.video_info.project_id
        self.user_id = user_id

        self.cloud_token = cloud_token
        self.cloud_action_id = cloud_action_id

        self.object_ids = list(set(object_ids))
        self.nn_settings = nn_settings
        self.frames_count = frames_count
        self.frame_ranges = [(frame_index, frame_index + frames_count)]

        self.logger = self.api.logger
        self.logger_extra = {
            "video_id": self.video_id,
            "track_id": self.track_id,
            "object_ids": self.object_ids,
        }

        self.batch_size = 16
        self.updates: List[Update] = []
        self.updates_pending = False
        self._lock = threading.Lock()
        self._lock = threading.Lock()
        self._upload_queue = queue.Queue()
        self._upload_thread = None
        self.prevent_upload_objects = []

        self.no_object_tag_ids = [
            t.id
            for t in self.api.video.object.tag.get_list(
                project_id=self.project_id,
                filters=[
                    {"field": "name", "operator": "=", "value": "no-objects"},
                ],
            )
        ]

        self.timelines: List[Timeline] = []
        init_timelines_time, _ = utils.time_it(self._init_timelines)

        self.logger.debug(
            "Init timelines time:",
            extra={**self.logger_extra, "time": f"{init_timelines_time:.6f} sec"},
        )

        self.global_stop_indicator = False
        self.progress = Progress(self)
        self.progress.frame_range = self.frame_ranges[0]
        self.refresh_progress()

    def _init_timelines(self):
        # Get common data for all timelines to avoid multiple requests
        # Figures
        all_figures_dict: Dict[int, FigureInfo] = {}
        for figure in self.api.video.figure.get_list(
            dataset_id=self.dataset_id,
            filters=[
                {"field": "objectId", "operator": "in", "value": self.object_ids},
                {"field": "startFrame", "operator": ">=", "value": self.frame_ranges[0][0]},
                {"field": "endFrame", "operator": "<=", "value": self.frame_ranges[0][1]},
            ],
        ):
            all_figures_dict.setdefault(figure.object_id, []).append(figure)

        # Object infos
        object_infos_dict = {
            object_info.id: object_info
            for object_info in self.api.video.object.get_list(
                dataset_id=self.dataset_id,
                filters=[{"field": "id", "operator": "in", "value": self.object_ids}],
            )
        }

        for object_id in self.object_ids:
            timeline = Timeline(
                self,
                object_id,
                self.frame_ranges[0][0],
                self.frame_ranges[0][1],
                object_info=object_infos_dict.get(object_id, None),
                figures=all_figures_dict.get(object_id, None),
            )
            self.timelines.append(timeline)

        self.logger.debug(
            "inited timelines",
            extra={**self.logger_extra, "timelines": [tl.log_data() for tl in self.timelines]},
        )

    def validate_timelines(self):
        for timeline in self.timelines:
            for tracklet in timeline.tracklets:
                for figure in tracklet.last_tracked[1]:
                    if validate_nn_settings_for_geometry(
                        self.nn_settings, figure.geometry_type, raise_error=False
                    ):
                        return True
        return False

    def nullify_progress(self):
        self.progress.current = 0
        self.progress.total = 0
        self.start_frame = 0
        self.end_frame = 0

    def update_timelines(
        self,
        frame_from: int,
        frame_to: int,
        timelines_indexes: List[int],
        predictions: List[List[List[FigureInfo]]],
    ):
        for timeline_index, timeline_predictions in zip(timelines_indexes, predictions):
            timeline = self.timelines[timeline_index]
            timeline.update(frame_from, frame_to, timeline_predictions)

    def refresh_progress(self):
        total = 0
        current = 0
        for timeline in self.timelines:
            total += timeline.get_progress_total()
            current += timeline.get_progress_current()
        self.progress.total = total
        self.progress.current = current
        self.log_timelines()

    def get_batch(self, batch_size: int = None):
        if batch_size is None:
            batch_size = self.batch_size
        tl_batches = [
            [*tl.get_batch(batch_size), tl_index] for tl_index, tl in enumerate(self.timelines)
        ]  # frame_from, frame_to, figures, tl_index
        tl_batches = [
            tl_batch for tl_batch in tl_batches if tl_batch[0] is not None and tl_batch[2]
        ]
        # filter out figures with no settings
        for tl_batch in tl_batches:
            tl_batch[2] = [
                figure
                for figure in tl_batch[2]
                if validate_nn_settings_for_geometry(
                    self.nn_settings, figure.geometry_type, raise_error=False, logger=self.logger
                )
            ]
        tl_batches = [tl_batch for tl_batch in tl_batches if len(tl_batch[2]) > 0]
        # Filter out finished timelines
        tl_batches = [tl_batch for tl_batch in tl_batches if tl_batch[0] < tl_batch[1]]
        if len(tl_batches) == 0:
            return None, None, None, None  # all timelines are finished
        # Find lowest starting frame
        frame_from = min([batch[0] for batch in tl_batches])
        # Leave only batches starting with the lowest starting frame
        tl_batches = [tl_batch for tl_batch in tl_batches if tl_batch[0] == frame_from]
        # Find lowest ending frame
        frame_to = min([batch[1] for batch in tl_batches])
        return (
            frame_from,
            frame_to,
            [tl_batch[2] for tl_batch in tl_batches],  # timelines figures
            [tl_batch[3] for tl_batch in tl_batches],  # timelines indexes
        )

    def log_timelines(self):
        """log timelines data"""
        self.logger.debug(
            "Timelines",
            extra={
                "timelines": [timeline.log_data() for timeline in self.timelines],
                **self.logger_extra,
            },
        )

    def is_object_disapeared(self, timeline: Timeline, frame_index: int, figures: List[FigureInfo]):
        dissapear_threshold = 0.5
        dissapear_frames = 10
        for tracklet in timeline.tracklets:
            if tracklet.start_frame < frame_index <= tracklet.end_frame:
                this_area = sum([utils.maybe_literal_eval(figure.area) for figure in figures])
                last_areas = [
                    tracklet.area_hist[fr_idx]
                    for fr_idx in sorted(tracklet.area_hist.keys())[-dissapear_frames:]
                ]
                if len(last_areas) < dissapear_frames - 1:
                    return False
                last_areas.append(this_area)
                med = tracklet.median_area()
                if all([area < med * dissapear_threshold for area in last_areas]):
                    sly.logger.debug(
                        "Object disapeared",
                        extra={
                            "timeline": timeline.log_data(),
                            "median": med,
                            "last_areas": last_areas,
                        },
                    )
                    return True
                return False
        return False

    # Updates
    def append_update(self, update: Update):
        """append update"""
        self.updates_pending = True
        with self._lock:
            self.updates.append(update)

    def pop_update(self) -> Update:
        """pop update"""
        with self._lock:
            if len(self.updates) == 0:
                return None
            return self.updates.pop(0)

    def pop_updates(self) -> List[Update]:
        """pop all updates"""
        updates = []
        while len(self.updates) > 0:
            updates.append(self.pop_update())
        return updates

    def apply_update(self, update: Update):
        """apply update"""
        if update.type == Update.Type.TRACK:
            self.object_changed(update.object_ids, update.frame_index, update.frames_count)
        elif update.type == Update.Type.CONTINUE:
            self.continue_track(update.frame_index, update.frames_count)
        elif update.type == Update.Type.DELETE:
            self.object_removed(update.object_ids[0], update.frame_index, update.frames_count)
        elif update.type == Update.Type.REMOVE_TAG:
            self.no_object_tag_removed(update.object_ids[0], update.frame_index)
        elif update.type == Update.Type.MANUAL_OBJECTS_REMOVED:
            self.manual_figure_removed(update.object_ids[0], update.frame_index)
        else:
            self.logger.warning(
                "Unknown update type", extra={"update": update, **self.logger_extra}
            )

    def apply_updates(self):
        pending_updates = self.pop_updates()
        if len(pending_updates) == 0:
            return
        self.logger.info("Apply updates", extra={"updates": pending_updates})
        for update in pending_updates:
            self.apply_update(update)
            self.refresh_progress()
            self.progress.notify()
            self.log_timelines()

    def wait_for_updates(self, timeout: int = 30):
        """wait for updates"""
        self.logger.debug("Waiting %s seconds for updates", timeout, extra={**self.logger_extra})
        for i in range(timeout):
            if self.updates_pending:
                self.logger.debug(
                    "Received updates after tracking finished", extra={**self.logger_extra}
                )
                return True
            time.sleep(1)
            if (i + 1) % 10 == 0:
                self.logger.debug(
                    "Waiting for updates %s/%s seconds...",
                    i + 1,
                    timeout,
                    extra={**self.logger_extra},
                )
            if self.global_stop_indicator:
                self.logger.info("Tracking stopped by user.")
                self.progress.notify(stop=True)
                return False
        return False

    # Billing
    def reserve_billing(self, items_count: int):
        """Reserve credits for predictions for Cloud Tracking."""
        if self.cloud_token is None or self.cloud_action_id is None:
            return
        try:
            transaction_id = self.api.cloud.billing_reserve(
                self.user_id,
                items_count=items_count,
                cloud_token=self.cloud_token,
                cloud_action_id=self.cloud_action_id,
            )["transactionId"]
            return transaction_id
        except Exception:
            self.logger.error("Unable to reserve tokens for predictions", exc_info=True)
            raise RuntimeError("Unable to reserve tokens for predictions") from None

    def withdraw_billing(self, transaction_id: str, items_count: int):
        if self.cloud_token is not None and self.cloud_action_id is not None:
            try:
                self.api.cloud.billing_withdrawal(
                    self.user_id,
                    items_count=items_count,
                    transaction_id=transaction_id,
                    cloud_token=self.cloud_token,
                    cloud_action_id=self.cloud_action_id,
                )
            except Exception:
                self.logger.error("Unable to withdraw tokens for predictions", exc_info=True)
                raise RuntimeError("Unable to withdraw tokens for predictions") from None

    # Predictions
    def run_geometry(
        self,
        geometry_type: str,
        figures: List[FigureInfo],
        frame_from: int,
        frame_to: int,
    ) -> List[List[FigureInfo]]:
        """
        Run tracking for the specified geometry type.
        Returns list of lists of FigureInfo objects.
        First list is for frames, second list is for predicted figures on the frame.
        run_geometry(*args)[i][j] is the prediction of j-th figure on the i-th frame.
        """
        self.logger.debug("Tracking geometry type %s", geometry_type)
        validate_nn_settings_for_geometry(self.nn_settings, geometry_type)

        frames_count = frame_to - frame_from
        try:
            if geometry_type == "smarttool":
                predictions = inference.predict_smarttool(
                    api=self.api,
                    video_id=self.video_id,
                    track_id=self.track_id,
                    nn_settings=self.nn_settings,
                    figure_metas=[figure.meta for figure in figures],
                    frame_index=frame_from,
                    frames_count=frames_count,
                )
            elif "url" in self.nn_settings[geometry_type]:
                url = self.nn_settings[geometry_type]["url"]
                predictions = inference.predict_by_url(
                    api=self.api,
                    video_id=self.video_id,
                    frame_index=frame_from,
                    frames_count=frames_count,
                    nn_url=url,
                    geometries_data=[figure.geometry for figure in figures],
                    geometry_type=geometry_type,
                )
            else:
                task_id = self.nn_settings[geometry_type]["task_id"]
                predictions = inference.predict_with_app(
                    api=self.api,
                    video_id=self.video_id,
                    task_id=task_id,
                    geometries_data=[figure.geometry for figure in figures],
                    geometry_type=geometry_type,
                    frame_index=frame_from,
                    frames_count=frames_count,
                )
        except Exception as e:
            cls, exc_str = utils.parse_exception(
                e, {"geometry": geometry_type, "frames": [frame_from, frame_to]}
            )
            raise cls(exc_str) from None
        result = []
        for i, frame_predictions in enumerate(predictions):
            result.append([])
            for prediction, src_figure in zip(frame_predictions, figures):
                result[-1].append(
                    utils.figure_from_prediction(
                        prediction=prediction,
                        figure_id=None,  # figure is not uploaded yet
                        object_id=src_figure.object_id,
                        frame_index=frame_from + 1 + i,
                        track_id=self.track_id,
                    )
                )
        return result

    def predict_batch(
        self, frame_from: int, frame_to: int, timelines_figures: List[List[FigureInfo]]
    ) -> List[List[List[FigureInfo]]]:
        """
        Get predictions for each figure for each timeline for a frame range.
        Returns list of lists of lists of FigureInfo objects.
        First list is for timelines, second list is for frames, third list is for predicted figures on the frame.
        predict_batch(*args)[i][j][k] is the prediction of k-th figure on the j-th frame of the i-th timeline.
        """
        # Merge and sort all figures by geometry type
        figures_by_type: Dict[str, List[FigureInfo]] = {}
        figures_by_type_index_to_timeline_index: Dict[str, List[int]] = {}
        for timeline_index, timeline_figures in enumerate(timelines_figures):
            timeline_figures_by_type = utils.split_figures_by_type(timeline_figures)
            for geometry_type, figures in timeline_figures_by_type.items():
                figures_by_type.setdefault(geometry_type, []).extend(figures)
                figures_by_type_index_to_timeline_index.setdefault(geometry_type, []).extend(
                    [timeline_index] * len(figures)
                )

        geom_types = list(figures_by_type.keys())
        with ThreadPoolExecutor(len(geom_types)) as executor:
            tasks_by_geom_type = {}
            for geom_type in geom_types:
                if geom_type == g.GEOMETRY_NAME.SMARTTOOL:
                    continue
                try:
                    validate_nn_settings_for_geometry(self.nn_settings, geom_type)
                except ValueError:
                    self.logger.warning("No settings for geometry type %s", geom_type)
                    continue
                task = executor.submit(
                    self.run_geometry,
                    geometry_type=geom_type,
                    figures=figures_by_type[geom_type],
                    frame_from=frame_from,
                    frame_to=frame_to,
                )
                tasks_by_geom_type[geom_type] = task
            results_by_geom_type = {
                geom_type: task.result() for geom_type, task in tasks_by_geom_type.items()
            }
            results_by_geom_type: Dict[str, List[List[FigureInfo]]]
        if g.GEOMETRY_NAME.SMARTTOOL in geom_types:
            try:
                validate_nn_settings_for_geometry(self.nn_settings, g.GEOMETRY_NAME.SMARTTOOL)
            except ValueError:
                self.logger.warning("No settings for geometry type smarttool")
            results_by_geom_type[g.GEOMETRY_NAME.SMARTTOOL] = self.run_geometry(
                geometry_type=g.GEOMETRY_NAME.SMARTTOOL,
                figures=figures_by_type[g.GEOMETRY_NAME.SMARTTOOL],
                frame_from=frame_from,
                frame_to=frame_to,
            )

        results = [
            [[] for _ in range(frame_to - frame_from)] for _ in range(len(timelines_figures))
        ]
        for geom_type, geom_predictions in results_by_geom_type.items():
            for frame_index, frame_predictions in enumerate(geom_predictions):
                for figure_index, predicted_figure in enumerate(frame_predictions):
                    timeline_index = figures_by_type_index_to_timeline_index[geom_type][
                        figure_index
                    ]
                    results[timeline_index][frame_index].append(predicted_figure)
        return results

    # Upload
    def _get_figures_from_predictions(
        self, predictions: List[List[List[FigureInfo]]]
    ) -> List[FigureInfo]:
        return [
            figure
            for timeline_predictions in predictions
            for frame_predictions in timeline_predictions
            for figure in frame_predictions
        ]

    def remove_predicted_figures_by_frame_range(
        self, object_ids: List[int], frame_range: Tuple[int, int]
    ):
        figures_to_delete: List[FigureInfo] = self.api.video.figure.get_list(
            dataset_id=self.dataset_id,
            filters=[
                {"field": "objectId", "operator": "in", "value": object_ids},
                {"field": "startFrame", "operator": ">=", "value": frame_range[0]},
                {"field": "endFrame", "operator": "<=", "value": frame_range[1]},
            ],
        )
        figures_to_delete = [figure for figure in figures_to_delete if figure.track_id is not None]
        if len(figures_to_delete) == 0:
            return
        self._remove_figures(figures_to_delete)

    def _remove_figures(self, figures: List[FigureInfo]):
        existing = self.api.video.figure.get_by_ids(self.dataset_id, [fig.id for fig in figures])
        try:
            self.api.video.figure.remove_batch([fig.id for fig in existing])
            return
        except Exception as e:
            self.logger.warning("Unable to remove figures in batch", extra={"exception": str(e)})
        for fig in existing:
            try:
                self.api.video.figure.remove(fig.id)
            except Exception as e:
                self.logger.warning("Unable to remove figure", extra={"exception": str(e)})

    def _safe_upload_figures(self, figures: List[FigureInfo]):
        try:
            return self._upload_figures(figures), []
        except Exception:
            self.logger.warning("Unable to upload figures", exc_info=True)
        figures_by_object = {}
        for figure in figures:
            figures_by_object.setdefault(figure.object_id, []).append(figure)
        uploaded_figures = []
        bad_objects = []
        for object_id, object_figures in figures_by_object.items():
            try:
                uploaded_figures.extend(self._upload_figures(object_figures))
            except Exception:
                self.logger.warning(
                    "Unable to upload figures for object %s", object_id, exc_info=True
                )
                bad_objects.append(object_id)
        return uploaded_figures, bad_objects

    def _upload_figures(self, figures: List[FigureInfo]):
        figures_json = [
            {
                ApiField.OBJECT_ID: figure.object_id,
                ApiField.GEOMETRY_TYPE: figure.geometry_type,
                ApiField.GEOMETRY: figure.geometry,
                ApiField.META: {**figure.meta, ApiField.FRAME: figure.frame_index},
                ApiField.TRACK_ID: self.track_id,
            }
            for figure in figures
        ]
        sly.logger.debug(
            "Uploading predictions batch",
            extra={
                "video_id": self.video_id,
                "track_id": self.track_id,
                "figures": [
                    {
                        "object_id": figure.object_id,
                        "frame_index": figure.frame_index,
                        "geometry_type": figure.geometry_type,
                        "meta": figure.meta,
                    }
                    for figure in figures
                ],
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
            "Uploaded predictions batch",
            extra={
                "video_id": self.video_id,
                "track_id": self.track_id,
                "figures": [
                    {
                        "id": key_id_map.get_figure_id(figure_key),
                        "object_id": figure.object_id,
                        "frame_index": figure.frame_index,
                        "geometry_type": figure.geometry_type,
                        "meta": figure.meta,
                    }
                    for figure, figure_key in zip(figures, figures_keys)
                ],
                "predictions_count": len(figures_json),
            },
        )

        return [key_id_map.get_figure_id(figure_key) for figure_key in figures_keys]

    def _filter_figures_to_upload(
        self, figures: List[FigureInfo], timestamp: float
    ) -> List[FigureInfo]:
        filtered = []

        def __prevent_upload_check(figure):
            for object_id, frame_range, obj_deleted_at in self.prevent_upload_objects:
                if (
                    figure.object_id == object_id
                    and obj_deleted_at > timestamp
                    and frame_range[0] <= figure.frame_index <= frame_range[1]
                ):
                    return False
            return True

        filtered = [figure for figure in figures if __prevent_upload_check(figure)]

        return filtered

    def _upload_iteration(
        self,
        predictions: List[List[List[FigureInfo]]],
        frame_range: Tuple[int, int],
        transaction_id: str,
    ):
        timestamp = time.time()
        figures = self._get_figures_from_predictions(predictions)
        figures = self._filter_figures_to_upload(figures, timestamp)
        object_ids = list(set([fig.object_id for fig in figures]))
        if len(object_ids) == 0:
            return

        figures_to_delete: List[FigureInfo] = self.api.video.figure.get_list(
            dataset_id=self.dataset_id,
            filters=[
                {"field": "objectId", "operator": "in", "value": object_ids},
                {"field": "startFrame", "operator": ">=", "value": frame_range[0]},
                {"field": "endFrame", "operator": "<=", "value": frame_range[1]},
            ],
        )
        figures_to_delete = [figure for figure in figures_to_delete if figure.track_id is not None]
        if figures_to_delete:
            threading.Thread(target=self._remove_figures, args=(figures_to_delete,)).start()

        uploaded_figures, bad_object_ids = self._safe_upload_figures(figures)

        self.withdraw_billing(transaction_id, items_count=len(uploaded_figures))

    def _upload_loop(self):
        while not self._upload_queue.empty():
            predictions, frame_range, transaction_id = self._upload_queue.get()
            self._upload_iteration(predictions, frame_range, transaction_id)

    def upload_predictions(
        self,
        predictions: List[List[List[FigureInfo]]],
        frame_range: Tuple[int, int],
        transaction_id: str = None,
    ):
        """Put predictions to upload queue and start upload thread if needed"""
        self._upload_queue.put((predictions, frame_range, transaction_id))
        if self._upload_thread is None or not self._upload_thread.is_alive():
            self._upload_thread = threading.Thread(target=self._upload_loop)
            self._upload_thread.start()

    # RUN main loop
    def run(self):
        # Notify the Annotation tool that the tracking is in progress
        self.progress.notify()
        while True:  # Main loop
            if self.global_stop_indicator:
                self.logger.info("Tracking stopped by user.")
                self.progress.notify(stop=True)
                return

            total_tm = TinyTimer()

            self.apply_updates()

            wait_update_time = TinyTimer()
            with self._lock:
                # Get batch data
                self.updates_pending = False
                get_batch_time, (frame_from, frame_to, timelines_figures, timelines_indexes) = (
                    utils.time_it(self.get_batch)
                )
            wait_update_time = wait_update_time.get_sec() - get_batch_time

            if frame_from is None:
                if self._upload_thread is not None and self._upload_thread.is_alive():
                    self._upload_thread.join()
                if self.wait_for_updates():
                    continue
                return

            self.logger.debug(
                "Batch data",
                extra={
                    "frame_from": frame_from,
                    "frame_to": frame_to,
                    "timelines": [self.timelines[i].log_data() for i in timelines_indexes],
                },
            )

            # billing reserve
            frames_count = frame_to - frame_from
            expected_predictions_count = sum(
                len(figures) * frames_count for figures in timelines_figures
            )
            transaction_id = self.reserve_billing(expected_predictions_count)

            # run iteration
            batch_prediction_time, batch_predictions = utils.time_it(
                self.predict_batch, frame_from, frame_to, timelines_figures
            )
            batch_predictions: List[List[List[FigureInfo]]]

            self.logger.debug(
                "Predictions before filtering",
                extra={
                    "timelines count": len(batch_predictions),
                    "frame_predictions_counts": [len(tl) for tl in batch_predictions],
                    "per_frame_predictions_counts": [
                        len(frame) for tl in batch_predictions for frame in tl
                    ],
                },
            )

            # filter disappearing figures
            for tl_index, timeline_predictions in enumerate(batch_predictions):
                timeline = self.timelines[timelines_indexes[tl_index]]
                removed_object_frame_idxs = set()
                for fr_index, frame_predictions in enumerate(timeline_predictions):
                    if self.is_object_disapeared(
                        timeline, frame_from + fr_index, frame_predictions
                    ):
                        removed_object_frame_idxs.add(fr_index)
                    if fr_index in removed_object_frame_idxs:
                        frame_predictions.clear()

            self.logger.debug(
                "Predictions after filtering",
                extra={
                    "timelines count": len(batch_predictions),
                    "frame_predictions_counts": [len(tl) for tl in batch_predictions],
                    "per_frame_predictions_counts": [
                        len(frame) for tl in batch_predictions for frame in tl
                    ],
                },
            )

            # upload and withdraw billing in parallel
            upload_time, _ = utils.time_it(
                self.upload_predictions,
                batch_predictions,
                frame_range=(frame_from + 1, frame_to),
                transaction_id=transaction_id,
            )

            # Update timelines
            update_timelines_time, _ = utils.time_it(
                self.update_timelines, frame_from, frame_to, timelines_indexes, batch_predictions
            )

            frame_range = self.frame_ranges[0]
            for fr in self.frame_ranges[1:]:
                if fr[0] <= frame_to <= fr[1]:
                    frame_range = fr
                    break
            self.progress.frame_range = frame_range
            self.refresh_progress()
            self.progress.notify()

            self.logger.debug(
                "Iteration time",
                extra={
                    "total": f"{total_tm.get_sec():.6f}  sec",
                    "wait update": f"{wait_update_time:.6f} sec",
                    "get batch data": f"{get_batch_time:.6f} sec",
                    "prediction": f"{batch_prediction_time:.6f} sec",
                    "upload predictions": f"{upload_time:.6f} sec",
                    "update timelines": f"{update_timelines_time:.6f} sec",
                },
            )

    def merge_frame_ranges(self):
        """Sorts and merges frame ranges"""
        merged_ranges = []
        for frame_range in sorted(self.frame_ranges, key=lambda x: x[0]):
            if len(merged_ranges) == 0 or merged_ranges[-1][1] < frame_range[0]:
                merged_ranges.append(frame_range)
            else:
                merged_ranges[-1] = (
                    merged_ranges[-1][0],
                    max(merged_ranges[-1][1], frame_range[1]),
                )
        self.frame_ranges = merged_ranges

    def get_frame_range(self, frame_index):
        for frame_range in self.frame_ranges:
            if frame_range[0] <= frame_index <= frame_range[1]:
                return frame_range
        return None

    def object_changed(self, object_ids: List[int], frame_index: int, frames_count: int):
        for obj_id in object_ids:
            self.prevent_object_upload(obj_id, (frame_index, self.video_info.frames_count))
        self.object_ids = list(set(self.object_ids + object_ids))
        frames_count = min(self.video_info.frames_count - frame_index, frames_count)
        self.frame_ranges.append((frame_index, frame_index + frames_count))
        self.merge_frame_ranges()
        frame_range = self.get_frame_range(frame_index)

        obj_id_to_timeline = {tl.object_id: tl for tl in self.timelines}
        for object_id in object_ids:
            if object_id in obj_id_to_timeline:
                timeline = obj_id_to_timeline[object_id]
                timeline.object_changed(frame_index, frame_range[1] - frame_index)
            else:
                self.timelines.append(
                    Timeline(
                        self,
                        object_id,
                        frame_index,
                        frame_range[1],
                    )
                )

    def continue_track(self, frame_index: int, frames_count: int):
        """
        Update track if it was extended
        Should extends timelines and download new figures
        """

        frames_count = min(self.video_info.frames_count - frame_index, frames_count)
        for frame_range in self.frame_ranges:
            if frame_range[0] <= frame_index <= frame_range[1]:
                if frame_index + frames_count <= frame_range[1]:
                    return
        self.frame_ranges.append((frame_index, frame_index + frames_count))
        self.merge_frame_ranges()
        for timeline in self.timelines:
            timeline.continue_timeline(frame_index, frames_count)

    def object_removed(self, object_id: int, frame_index: int, frames_count: int):
        """
        Update track if objects were removed
        It means that objects should not be tracked on the frames where they were removed
        Should update timelines and remove added figures
        """
        self.prevent_object_upload(object_id, (frame_index, self.video_info.frames_count))
        for timeline in self.timelines:
            if timeline.object_id == object_id:
                timeline.object_removed(frame_index, frames_count)

    def no_object_tag_removed(self, object_id: int, frame_index: int):
        """
        Update track if objects removal was cancelled
        It means that objects should be tracked on the frames where they were removed
        Should update timelines and set last tracked figures to one frame before the removal
        """
        frame_range = self.get_frame_range(frame_index)
        if frame_range is None:
            frame_range = self.frame_ranges[0]
        for timeline in self.timelines:
            if timeline.object_id == object_id:
                timeline.no_object_tag_removed(frame_index=frame_index, frame_range=frame_range)

    def manual_figure_removed(self, object_id: int, frame_index: int):
        """Update track if figures annotated by user were removed"""
        frame_range = None
        for fr in self.frame_ranges:
            if fr[0] <= frame_index <= fr[1]:
                frame_range = fr
                break
        if frame_range is None:
            return
        for timeline in self.timelines:
            if timeline.object_id == object_id:
                timeline.manual_figure_removed(frame_index, frame_range[1])

    def prevent_object_upload(self, object_id: int, frame_range: Tuple[int, int]):
        self.prevent_upload_objects.append((object_id, frame_range, time.time()))

    def stop(self):
        self.global_stop_indicator = True


@utils.send_error_data
def track(
    api: sly.Api,
    context: Dict,
    nn_settings: Dict,
    update_type: str = "track",
    cloud_token: str = None,
    cloud_action_id: str = None,
):
    sly.logger.debug("track", extra={"context": context, "nn_settings": nn_settings})

    if update_type == Update.Type.DELETE:
        delete_data = []
        for figure in context["figures"]:
            object_id = figure["objectId"]
            frame_range = figure["frames"]
            delete_data.append((object_id, frame_range))
        api.logger.info("Delete figures", extra={"delete_data": delete_data})
        for object_id, frame_range in delete_data:
            frame_index = frame_range[0]
            frames_count = frame_range[1] - frame_range[0] + 1
            tracks_to_update = set()
            for track_id, cur_track in g.current_tracks.items():
                if object_id in cur_track.object_ids:
                    cur_track.append_update(
                        Update([object_id], frame_range[0], frames_count, update_type)
                    )
                    tracks_to_update.add(track_id)
            for track_id in tracks_to_update:
                cur_track = g.current_tracks[track_id]
                threading.Thread(target=cur_track.apply_updates).start()
        return

    if update_type == Update.Type.CONTINUE:
        track_id = context["trackId"]
        video_id = context["videoId"]
        frame_index = context["frameIndex"]
        frames_count = context["frames"]
        cur_track = g.current_tracks.get(track_id, None)
        if cur_track is not None:
            api.logger.info("Continue tracking.", extra={"track_id": track_id})
            cur_track.append_update(
                Update(
                    [],
                    frame_index,
                    frames_count,
                    update_type,
                )
            )
            cur_track.apply_updates()
            return
        else:
            api.logger.info("Track not found. Starting new one", extra={"track_id": track_id})

    if update_type == Update.Type.REMOVE_TAG:
        tag = context["tag"]
        video_id = tag.get("imageId", tag.get("videoId", None))
        object_id = tag["objectId"]
        frame_range = tag["frameRange"]
        tag_id = tag["tagId"]  # NOT USED
        tracks_to_update = set()
        for cur_track in g.current_tracks.values():
            if cur_track.video_id == video_id:
                cur_track.append_update(
                    Update(
                        object_ids=[object_id],
                        frame_index=frame_range[0],
                        frames_count=frame_range[1] - frame_range[0] + 1,
                        type=update_type,
                        tag_id=tag_id,  # NOT USED
                    )
                )
                tracks_to_update.add(cur_track.track_id)
        for track_id in tracks_to_update:
            cur_track = g.current_tracks[track_id]
            threading.Thread(target=cur_track.apply_updates).start()
        return

    if update_type == Update.Type.MANUAL_OBJECTS_REMOVED:
        figures = context["figures"]
        video_id = context["videoId"]
        tracks_to_update = set()
        for cur_track in g.current_tracks.values():
            if cur_track.video_id == video_id:
                for figure in figures:
                    cur_track.append_update(
                        Update(
                            [figure["objectId"]],
                            figure["frame"],
                            1,
                            update_type,
                        )
                    )
                tracks_to_update.add(cur_track.track_id)
        for track_id in tracks_to_update:
            cur_track = g.current_tracks[track_id]
            threading.Thread(target=cur_track.apply_updates).start()
        return

    # track
    session_id = context.get("sessionId", context.get("session_id", None))
    if session_id is None:
        api.logger.warn("Session id is not provided. Some features may not work correctly.")
    track_id = context["trackId"]
    video_id = context["videoId"]
    object_ids = list(context["objectIds"])
    frame_index = context["frameIndex"]
    frames_count = context["frames"]
    user_id = api.user.get_my_info().id
    # direction = context["direction"]
    with g.tracks_lock:
        cur_track: Track = g.current_tracks.get(track_id, None)
        if cur_track is not None:
            api.logger.info("Figure changed. Update tracking", extra={"track_id": track_id})
            cur_track.append_update(Update(object_ids, frame_index, frames_count, update_type))
            return
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
        if not cur_track.validate_timelines():
            cur_track.nullify_progress()
            raise ValueError("No settings for selected geometries. Tracking stopped.")
    try:
        cur_track.run()
    finally:
        if not cur_track.global_stop_indicator:
            cur_track.progress.notify(stop=True)
        g.current_tracks.pop(track_id, None)
        api.logger.debug("Tracking completed.")
