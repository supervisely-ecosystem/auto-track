"""Per-operation cancellation and bounded HTTP calls; no worker threads or subprocesses."""
from threading import Event
import os

import requests
import supervisely as sly
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from supervisely.nn.inference.session import Session


CONNECT_TIMEOUT_SECONDS = 5
READ_TIMEOUT_SECONDS = int(os.environ.get("AUTO_TRACK_READ_TIMEOUT_SECONDS", "30"))
if READ_TIMEOUT_SECONDS < 1:
    raise ValueError("AUTO_TRACK_READ_TIMEOUT_SECONDS must be a positive integer.")
NOTIFICATION_TIMEOUT_SECONDS = 5
_operations = {}


class TrackingCancelled(Exception):
    pass


def register_operation(track_id):
    return _operations.setdefault(track_id, Event())


def stop_operation(track_id):
    cancellation = _operations.get(track_id)
    if cancellation is not None:
        cancellation.set()


def finish_operation(track_id, cancellation):
    if _operations.get(track_id) is cancellation:
        _operations.pop(track_id)


class TrackingApi(sly.Api):
    """Own request limits and cancellation without mutating the shared app API."""

    def __init__(self, api, cancellation):
        self.cancellation = cancellation
        super().__init__(api.server_address, api.token, retry_count=1,
                         external_logger=api.logger, api_server_address=api.api_server_address)
        self.headers = dict(api.headers)
        self.additional_fields = dict(api.additional_fields)

    def check_cancelled(self):
        if self.cancellation.is_set():
            raise TrackingCancelled("Tracking stopped by user.")

    def post(self, method, data, retries=None, stream=False, raise_error=False):
        # Final progress and accounting for already accepted writes must remain possible.
        if method != "videos.notify-annotation-tool":
            self.check_cancelled()
        headers = dict(self.headers)
        json_body, body = None, None
        if isinstance(data, dict):
            json_body = {**data, **self.additional_fields}
        else:
            body = data
            if isinstance(data, (MultipartEncoder, MultipartEncoderMonitor)):
                headers["Content-Type"] = data.content_type
        response = requests.post(
            f"{self.api_server_address}/v3/{method}", json=json_body, data=body,
            headers=headers, stream=stream,
            timeout=(CONNECT_TIMEOUT_SECONDS, NOTIFICATION_TIMEOUT_SECONDS
                     if method == "videos.notify-annotation-tool" else READ_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        return response


def post_model(api, url, json_body):
    if isinstance(api, TrackingApi):
        api.check_cancelled()
    response = requests.post(url, json=json_body,
                             timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS))
    if isinstance(api, TrackingApi):
        api.check_cancelled()
    response.raise_for_status()
    return response


class BoundedSession(Session):
    """Keep SDK annotation conversion while bounding its otherwise unbounded model HTTP calls."""

    def _post(self, url, json=None):
        return post_model(self.api, url, json)


def post_billing(api, action, user_id, count, cloud_token, cloud_action_id, transaction_id=None):
    if action == "reserve":
        api.check_cancelled()
    payload = {"userId": user_id, "objects": count}
    if transaction_id is not None:
        payload["transactionId"] = transaction_id
    response = requests.post(
        f"{sly.env.sly_cloud_server_address()}/billing/{action}", json=payload,
        headers={"x-sly-cloud-token": cloud_token, "x-sly-cloud-action-id": cloud_action_id},
        timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
    )
    response.raise_for_status()
    return response.json()
