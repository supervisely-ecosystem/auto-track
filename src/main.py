import requests
import supervisely as sly
from fastapi import BackgroundTasks, Request, Response

import src.globals as g
from src.ui import layout, get_nn_settings, update_all_nn
from src.tracking import track, cache_video

app = sly.Application(layout=layout)

server = app.get_server()


@server.post("/track")
def start_track(request: Request, task: BackgroundTasks):
    sly.logger.debug("recieved call to /track")
    nn_settings = get_nn_settings()
    api = request.state.api
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
    )
    return {"message": "Track task started."}


@server.post("/cache_video")
def start_cache_video(request: Request, task: BackgroundTasks):
    """Starts a task to cache the video by its id.
    Request should contain the server_address and api_token"""
    sly.logger.debug("recieved call to /cache_video")
    nn_settings = get_nn_settings()
    api = request.state.api
    state = request.state.state
    task.add_task(cache_video, api, state, nn_settings)
    return {"message": "Cache video task started."}


@server.post("/smart_segmentation")
def smart_segmentation(request: Request):
    sly.logger.debug("recieved call to /smart_segmentation")
    nn_settings = get_nn_settings()
    if "url" in nn_settings["smarttool"]:
        url = nn_settings["smarttool"]["url"]
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
        task_id = nn_settings["smarttool"]["task_id"]
        state = request.state.state
        context = request.state.context
        return g.api.app.send_request(task_id, "smart_segmentation", data=state, context=context)


@server.post("/continue_track")
def continue_track(request: Request, task: BackgroundTasks):
    sly.logger.debug("recieved call to /continue_track")
    nn_settings = get_nn_settings()
    api = request.state.api
    context = request.state.context
    cloud_token = request.headers.get("x-sly-cloud-token", None)
    cloud_action_id = request.headers.get("x-sly-cloud-action-id", None)
    task.add_task(
        track,
        api,
        context,
        nn_settings,
        "continue",
        cloud_token=cloud_token,
        cloud_action_id=cloud_action_id,
    )
    return {"message": "Track task started."}


@server.post("/objects_removed")
def objects_removed(request: Request, task: BackgroundTasks):
    sly.logger.debug("recieved call to /objects_removed")
    nn_settings = get_nn_settings()
    api = request.state.api
    context = request.state.context
    cloud_token = request.headers.get("x-sly-cloud-token", None)
    cloud_action_id = request.headers.get("x-sly-cloud-action-id", None)
    task.add_task(
        track,
        api,
        context,
        nn_settings,
        "delete",
        cloud_token=cloud_token,
        cloud_action_id=cloud_action_id,
    )
    return {"message": "Objects removed."}


update_all_nn()
