from concurrent.futures import ThreadPoolExecutor
from typing import Dict
import requests
import supervisely as sly


import src.globals as g


def cache_geometry(api: sly.Api, nn_settings: Dict, geometry: str, state: Dict):
    if "url" in nn_settings[geometry]:
        url = nn_settings[geometry]["url"]
        if url is None or url == "":
            sly.logger.debug(
                "Cache video is skipped because url is empty", extra={"geometry": geometry}
            )
            return
        sly.logger.debug("Cache video request", extra={"url": url, "geometry": geometry})
        r = requests.post(
            f"{url}/smart_cache",
            json={"state": state, "server_address": api.server_address, "api_token": api.token},
            timeout=60,
        )
        r.raise_for_status()
        sly.logger.debug("Cache video response", extra={"response": r.json(), "geometry": geometry})

    elif "task_id" in nn_settings[geometry]:
        task_id = nn_settings[geometry]["task_id"]
        if task_id is None or task_id == "":
            sly.logger.debug(
                "Cache video is skipped because task_id is empty", extra={"geometry": geometry}
            )
            return
        sly.logger.debug(
            "Cache video request", extra={"app_task_id": task_id, "geometry": geometry}
        )
        r = api.app.send_request(task_id, "smart_cache", state, retries=1)
        sly.logger.debug("Cache video response", extra={"response": r, "geometry": geometry})


def cache_video(api: sly.Api, state: Dict, nn_settings: dict):
    geometries = state.pop("geometries", None)
    if geometries is None or sly.AnyGeometry.geometry_name() in geometries:
        geometries = set(g.geometry_nn.keys())
    geometries = set(geometries)
    if sly.Bitmap.geometry_name() in geometries:
        geometries.add(g.GEOMETRY_NAME.SMARTTOOL)
    if g.GEOMETRY_NAME.SMARTTOOL in geometries:
        geometries.update([g.GEOMETRY_NAME.POINT, g.GEOMETRY_NAME.RECTANGLE])

    sly.logger.debug(
        "Start caching video for this geometries",
        extra={"geometries": geometries, "nn_settings": nn_settings},
    )

    geometries = list(geometries)
    with ThreadPoolExecutor(max_workers=len(geometries)) as executor:
        tasks = [
            executor.submit(
                cache_geometry,
                api,
                nn_settings,
                geometry,
                state,
            )
            for geometry in geometries
        ]
    for geometry, task in zip(geometries, tasks):
        try:
            task.result()
        except Exception as e:
            sly.logger.warning("Error caching video", exc_info=e, extra={"geometry": geometry})
            raise
