import requests
import supervisely as sly
from fastapi import BackgroundTasks, Request, Response

import src.globals as g
from src.ui import layout, get_nn_settings, update_all_nn, get_disappear_parameters
from src.tracking import track, cache_video, interpolate_frames
from src.tracking.track import Update, Track

app = sly.Application(layout=layout)

server = app.get_server()


@server.post("/track")
def start_track(request: Request, task: BackgroundTasks):
    """Start a new track or add new objects to the existing track."""
    sly.logger.debug("recieved call to /track")
    nn_settings = get_nn_settings()
    disappear_params = get_disappear_parameters()
    api = request.state.api
    if api is None:
        api = g.api
    context = request.state.context
    cloud_token = request.headers.get("x-sly-cloud-token", None)
    cloud_action_id = request.headers.get("x-sly-cloud-action-id", None)
    task.add_task(
        track,
        api,
        context,
        nn_settings,
        cloud_token=cloud_token,
        cloud_action_id=cloud_action_id,
        disappear_params=disappear_params,
    )
    return {"message": "Track task started."}


@server.post("/tracking_by_detection")
def start_tracking_by_detection(request: Request, task: BackgroundTasks):
    sly.logger.debug("recieved call to /tracking_by_detection")
    nn_settings = get_nn_settings()
    api = request.state.api
    if api is None:
        api = g.api
    context = request.state.context
    task_id = nn_settings.get(g.GEOMETRY_NAME.DETECTOR, {}).get("task_id", None)
    if task_id is None:
        raise ValueError("Detection model is not selected")
    return api.task.send_request(task_id, "POST", "tracking_by_detection", data=context)
    return {"message": "Track task started."}


@server.post("/cache_video")
def start_cache_video(request: Request, task: BackgroundTasks):
    """Starts a task to cache the video by its id.
    Request should contain the server_address and api_token"""
    sly.logger.debug("recieved call to /cache_video")
    nn_settings = get_nn_settings()
    api = request.state.api
    if api is None:
        api = g.api
    state = request.state.state
    task.add_task(cache_video, api, state, nn_settings)
    return {"message": "Cache video task started."}


@server.post("/smart_segmentation")
def smart_segmentation(request: Request):
    sly.logger.debug("recieved call to /smart_segmentation")
    nn_settings = get_nn_settings()
    if "url" in nn_settings[g.GEOMETRY_NAME.SMARTTOOL]:
        url = nn_settings[g.GEOMETRY_NAME.SMARTTOOL]["url"]
        if not url:
            return {
                "origin": None,
                "bitmap": None,
                "success": False,
                "error": {"message": "Smart tool model is not selected"},
            }
        state = request.state.state
        context = request.state.context
        data = {
            "state": state,
            "context": context,
            "server_address": request.state.server_address,
            "api_token": request.state.api_token,
        }
        r = requests.post(f"{url}/smart_segmentation", json=data, timeout=60)
        return Response(r.content, status_code=r.status_code, media_type=r.headers["Content-Type"])
    else:
        task_id = nn_settings[g.GEOMETRY_NAME.SMARTTOOL]["task_id"]
        if task_id is None:
            return {
                "origin": None,
                "bitmap": None,
                "success": False,
                "error": {"message": "Smart tool model is not selected"},
            }
        state = request.state.state
        context = request.state.context
        return g.api.app.send_request(
            task_id, "smart_segmentation", data=state, context=context, retries=1
        )


@server.post("/available_geometries")
def available_geometries(request: Request):
    sly.logger.debug("recieved call to /available_geometries")
    nn_settings = get_nn_settings()
    available = []
    for geometry_name, settings in nn_settings.items():
        if "url" in settings:
            if settings["url"]:
                available.append(geometry_name)
        else:
            if settings["task_id"]:
                available.append(geometry_name)
    if all(
        geom in available
        for geom in [g.GEOMETRY_NAME.POINT, g.GEOMETRY_NAME.RECTANGLE, g.GEOMETRY_NAME.SMARTTOOL]
    ):
        available.append("smarttool-track")
    return available


@server.post("/project_meta_changed")
def project_meta_changed(request: Request):
    """Project meta changed"""
    sly.logger.debug(
        "recieved call to /project_meta_changed", extra={"context": request.state.context}
    )
    api = request.state.api
    if api is None:
        api = g.api
    context = request.state.context
    project_id = context.get("projectId", None)
    for cur_track in g.current_tracks.values():
        cur_track: Track
        if cur_track.project_id == project_id:
            cur_track.update_project_meta()


@server.post("/no_objects_tag_changed")
def no_objects_tag_changed(request: Request, task: BackgroundTasks):
    """No objects tag changed"""
    sly.logger.debug(
        "recieved call to /no_objects_tag_changed", extra={"context": request.state.context}
    )
    nn_settings = get_nn_settings()
    disappear_params = get_disappear_parameters()
    api = request.state.api
    if api is None:
        api = g.api
    context = request.state.context
    cloud_token = request.headers.get("x-sly-cloud-token", None)
    cloud_action_id = request.headers.get("x-sly-cloud-action-id", None)
    task.add_task(
        track,
        api,
        context,
        nn_settings,
        Update.Type.NO_OBJECTS_TAG_CHANGED,
        cloud_token=cloud_token,
        cloud_action_id=cloud_action_id,
        disappear_params=disappear_params,
    )


@server.post("/continue_track")
def continue_track(request: Request, task: BackgroundTasks):
    """
    Continue existing track
    """
    sly.logger.debug("recieved call to /continue_track")
    nn_settings = get_nn_settings()
    disappear_params = get_disappear_parameters()
    api = request.state.api
    if api is None:
        api = g.api
    context = request.state.context
    cloud_token = request.headers.get("x-sly-cloud-token", None)
    cloud_action_id = request.headers.get("x-sly-cloud-action-id", None)
    task.add_task(
        track,
        api,
        context,
        nn_settings,
        Update.Type.CONTINUE,
        cloud_token=cloud_token,
        cloud_action_id=cloud_action_id,
        disappear_params=disappear_params,
    )
    return {"message": "Track task started."}


@server.post("/objects_removed")
def objects_removed(request: Request, task: BackgroundTasks):
    """Objects removed from tracking on specific frames"""
    sly.logger.debug("recieved call to /objects_removed")
    nn_settings = get_nn_settings()
    disappear_params = get_disappear_parameters()
    api = request.state.api
    if api is None:
        api = g.api
    context = request.state.context
    cloud_token = request.headers.get("x-sly-cloud-token", None)
    cloud_action_id = request.headers.get("x-sly-cloud-action-id", None)
    task.add_task(
        track,
        api,
        context,
        nn_settings,
        Update.Type.DELETE,
        cloud_token=cloud_token,
        cloud_action_id=cloud_action_id,
        disappear_params=disappear_params,
    )
    return {"message": "Objects removed."}


@server.post("/tag_removed")
def tag_removed(request: Request, task: BackgroundTasks):
    """Remove no-objects tag"""
    sly.logger.debug("recieved call to /tag_removed", extra={"context": request.state.context})
    nn_settings = get_nn_settings()
    disappear_params = get_disappear_parameters()
    api = request.state.api
    if api is None:
        api = g.api
    context = request.state.context
    cloud_token = request.headers.get("x-sly-cloud-token", None)
    cloud_action_id = request.headers.get("x-sly-cloud-action-id", None)
    task.add_task(
        track,
        api,
        context,
        nn_settings,
        Update.Type.REMOVE_TAG,
        cloud_token=cloud_token,
        cloud_action_id=cloud_action_id,
        disappear_params=disappear_params,
    )


@server.post("/manual_objects_removed")
def manual_objects_removed(request: Request, task: BackgroundTasks):
    """Figure annotated by user was deleted"""
    sly.logger.debug(
        "recieved call to /manual_objects_removed", extra={"context": request.state.context}
    )
    nn_settings = get_nn_settings()
    disappear_params = get_disappear_parameters()
    api = request.state.api
    if api is None:
        api = g.api
    context = request.state.context
    cloud_token = request.headers.get("x-sly-cloud-token", None)
    cloud_action_id = request.headers.get("x-sly-cloud-action-id", None)
    task.add_task(
        track,
        api,
        context,
        nn_settings,
        Update.Type.MANUAL_OBJECTS_REMOVED,
        cloud_token=cloud_token,
        cloud_action_id=cloud_action_id,
        disappear_params=disappear_params,
    )


@server.post("/stop_tracking")
def stop_tracking(request: Request, task: BackgroundTasks):
    """Stop tracking"""
    sly.logger.debug("recieved call to /stop_tracking", extra={"context": request.state.context})
    context = request.state.context
    track_id = context["trackId"]
    cur_track = g.current_tracks.get(track_id, None)
    if cur_track is None:
        return
    cur_track.stop()


@server.post("/interpolate")
def interpolate(request: Request, task: BackgroundTasks):
    """Interpolate missing frames"""
    sly.logger.debug(
        "recieved call to /interpolate",
        extra={"context": request.state.context, "state": request.state.state},
    )
    api = request.state.api
    if api is None:
        api = g.api
    context = request.state.context
    task.add_task(interpolate_frames, api, context)


update_all_nn()
