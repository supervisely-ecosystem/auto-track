"""Own registration, initialization, execution, and terminal cleanup of a tracking operation."""
import requests

import src.globals as g
import src.utils as utils
from .request_control import TrackingApi, TrackingCancelled, register_operation, finish_operation


def run_operation(api, context, nn_settings, disappear_params, cloud_token, cloud_action_id, update_type):
    from .track import Track, Update

    track_id = context["trackId"]
    cancellation = register_operation(track_id)
    api = TrackingApi(api, cancellation)
    cur_track = None
    owns_operation = False
    try:
        # Register cancellation before any initialization or model requests.
        with g.tracks_lock:
            cur_track = g.current_tracks.get(track_id)
            if cur_track is not None:
                cur_track.append_update(Update(context["objectIds"], context["frameIndex"],
                                               context["frames"], update_type))
                cur_track.disappear_params = disappear_params
                return
            owns_operation = True
            api.check_cancelled()
            cur_track = Track(
                track_id=track_id, session_id=context.get("sessionId", context.get("session_id")),
                api=api, video_id=context["videoId"], object_ids=list(context["objectIds"]),
                frame_index=context["frameIndex"], frames_count=context["frames"],
                nn_settings=nn_settings, user_id=api.user.get_my_info().id,
                cloud_token=cloud_token, cloud_action_id=cloud_action_id,
                disappear_params=disappear_params,
                detection_enabled=context.get("trackByDetection", True),
                disappear_enabled=context.get("detectOffScreen", True),
            )
            g.current_tracks[track_id] = cur_track
            api.check_cancelled()
            if not cur_track.validate_timelines():
                raise ValueError("No settings for selected geometries. Tracking stopped.")
        cur_track.run()
    except TrackingCancelled:
        api.logger.info("Tracking cancelled", extra={"track_id": track_id})
    except Exception as error:
        if not cancellation.is_set():
            api.logger.error("Tracking failed", exc_info=True, extra={"track_id": track_id})
            message = ("The model or server did not respond in time. Check the session and retry."
                       if isinstance(error, requests.Timeout) else str(error))
            utils.notify_error(api, track_id, context["videoId"], message)
    finally:
        if owns_operation:
            cancellation.set()
            # Keep the legacy terminal progress signal until the separate progress refactor.
            try:
                if cur_track is not None:
                    cur_track.progress.notify(stop=True)
                else:
                    api.video.notify_progress(track_id, context["videoId"], context["frameIndex"] + 1,
                                              context["frameIndex"] + context["frames"], 1, 1)
            except requests.RequestException:
                api.logger.warning("Could not send final tracking status", exc_info=True)
            finally:
                g.current_tracks.pop(track_id, None)
                finish_operation(track_id, cancellation)
