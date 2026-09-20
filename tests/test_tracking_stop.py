"""Exercise cancellation with the real SDK API boundary and deterministic model responses."""
import importlib
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

import requests
import supervisely as sly


globals_stub = types.ModuleType("src.globals")
with patch.dict(sys.modules, {"src.globals": globals_stub}):
    tracking = importlib.import_module("src.tracking.track")
    operation = importlib.import_module("src.tracking.operation")
    control = importlib.import_module("src.tracking.request_control")
    inference = importlib.import_module("src.tracking.inference")


class TrackingStopTests(unittest.TestCase):
    def setUp(self):
        self.shared_api = sly.Api("https://example.test", "test-token")
        self.shared_api.headers["x-toolbox-session-id"] = "original"
        self.cancel = control.register_operation("track-1")
        self.api = control.TrackingApi(self.shared_api, self.cancel)
        globals_stub.current_tracks = {}
        globals_stub.tracks_lock = threading.Lock()
        self.addCleanup(control.finish_operation, "track-1", self.cancel)

    def test_stop_before_initialization_blocks_api_requests(self):
        control.stop_operation("track-1")
        control.stop_operation("track-1")
        with patch.object(control.requests, "post") as post:
            with self.assertRaises(control.TrackingCancelled):
                self.api.video.get_info_by_id(1)
        post.assert_not_called()
        self.assertEqual(self.shared_api.headers["x-toolbox-session-id"], "original")

    def test_sdk_task_request_has_transport_timeout_and_no_retry(self):
        with patch.object(control.requests, "post", side_effect=requests.ReadTimeout) as post:
            with self.assertRaises(requests.ReadTimeout):
                self.api.task.send_request(22, "track-api", {}, retries=10)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.kwargs["timeout"], (5, 30))
        self.assertEqual(post.call_args.args[0], "https://example.test/public/api/v3/tasks.request.direct")

    def test_stop_discards_model_result_returned_after_cancellation(self):
        def reply(url, json, timeout):
            self.cancel.set()
            return Mock()
        with patch.object(control.requests, "post", side_effect=reply):
            with self.assertRaises(control.TrackingCancelled):
                control.post_model(self.api, "https://model.test/track-api", {})

    def test_stop_prevents_later_annotation_writes(self):
        self.cancel.set()
        with patch.object(control.requests, "post") as post:
            with self.assertRaises(control.TrackingCancelled):
                self.api.post("videos.figures.bulk.add", {"figures": []})
        post.assert_not_called()

    def test_detector_session_uses_bounded_transport(self):
        session = control.BoundedSession.__new__(control.BoundedSession)
        session.api = self.api
        with patch.object(control.requests, "post", side_effect=requests.ReadTimeout) as post:
            with self.assertRaises(requests.ReadTimeout):
                session._post("https://model.test/inference_video_id", json={})
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.kwargs["timeout"], (5, 30))

    def test_operation_cleans_registry_when_stopped_during_initialization(self):
        context = {"trackId": "track-1", "videoId": 1, "frameIndex": 0,
                   "frames": 5, "objectIds": [8]}
        api = Mock(cancellation=self.cancel)
        def load_user():
            control.stop_operation("track-1")
            raise control.TrackingCancelled()
        api.user.get_my_info.side_effect = load_user
        with patch.dict(sys.modules, {"src.tracking.track": tracking}), \
             patch.object(operation, "TrackingApi", return_value=api), \
             patch.object(operation.utils, "notify_error") as notify_error:
            operation.run_operation(self.shared_api, context, {}, {}, None, None, "track")
        notify_error.assert_not_called()
        self.assertEqual(globals_stub.current_tracks, {})
        self.assertNotIn("track-1", control._operations)
        api.video.notify_progress.assert_called_once()

    def test_cancelled_inference_cannot_reach_upload(self):
        track = tracking.Track.__new__(tracking.Track)
        track.api = self.api
        track.logger = Mock()
        track._check_and_notify_missing_geoms = Mock()
        track.progress = Mock()
        track.apply_updates = Mock()
        track.is_detection_enabled = Mock(return_value=False)
        track._lock = threading.Lock()
        track.get_batch = Mock(return_value=(0, 2, [[]], []))
        track.timelines = []
        track.video_id = 1
        track.track_id = "track-1"
        track.reserve_billing = Mock()
        track._upload_iteration_and_update_timelines = Mock()
        def infer(frame_from, frame_to, figures):
            track.stop()
            return [[]]
        track.predict_batch = infer
        with self.assertRaises(control.TrackingCancelled):
            track.run()
        track._upload_iteration_and_update_timelines.assert_not_called()

    def test_progress_honors_toolbox_stop_response(self):
        track = Mock(global_stop_indicator=False, timelines=[], logger_extra={})
        track.api.video.notify_progress.return_value = True
        tracking.Progress(track).notify()
        track.stop.assert_called_once()

    def test_final_notification_remains_possible_after_stop(self):
        self.cancel.set()
        with patch.object(control.requests, "post", return_value=Mock()) as post:
            self.api.post("videos.notify-annotation-tool", {"type": "videos:fetch-figures-in-range"})
        post.assert_called_once()

    def test_billing_for_an_accepted_write_is_bounded_and_not_cancelled(self):
        self.cancel.set()
        with patch.object(sly.env, "sly_cloud_server_address", return_value="https://cloud.test", create=True), \
             patch.object(control.requests, "post", return_value=Mock()) as post:
            control.post_billing(self.api, "withdrawal", 7, 3, "token", "action", "transaction")
        self.assertEqual(post.call_args.args[0], "https://cloud.test/billing/withdrawal")
        self.assertEqual(post.call_args.kwargs["timeout"], (5, 30))
        self.assertEqual(post.call_args.kwargs["json"], {
            "userId": 7, "objects": 3, "transactionId": "transaction",
        })

    def test_completed_operation_does_not_poison_next_request_with_same_id(self):
        self.cancel.set()
        control.finish_operation("track-1", self.cancel)
        new_cancellation = control.register_operation("track-1")
        self.addCleanup(control.finish_operation, "track-1", new_cancellation)
        self.assertFalse(new_cancellation.is_set())


if __name__ == "__main__":
    unittest.main()
