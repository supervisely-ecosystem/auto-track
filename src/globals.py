import os
import threading
from typing import Dict, List, Literal, Tuple

from dotenv import load_dotenv

import supervisely as sly

# Enabling advanced debug mode.
if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()
api.retry_count = 2

team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()

tracks_lock = threading.Lock()

NO_OBJECT_TAGS = ["no-objects"]


class ENV:
    @classmethod
    def mixformer_url(cls) -> str:
        return os.getenv("MIXFORMER_URL", "http://mixformer")

    @classmethod
    def xmem_url(cls) -> str:
        return os.getenv("XMEM_URL", "http://xmem")

    @classmethod
    def cotracker_url(cls) -> str:
        return os.getenv("COTRACKER_URL", "http://cotracker")

    @classmethod
    def clickseg_url(cls) -> str:
        return os.getenv("CLICKSEG_URL", "http://clickseg")

    @classmethod
    def yolov8_url(cls) -> str:
        return os.getenv("YOLOV8_URL", "http://yolov8")

    @classmethod
    def is_cloud(cls) -> bool:
        if hasattr(sly.env, "sly_cloud_server_address"):
            sly.env.sly_cloud_server_address(raise_not_found=False) is not None
        return False


class GEOMETRY_NAME:
    RECTANGLE = sly.Rectangle.geometry_name()
    POINT = sly.Point.geometry_name()
    POLYLINE = sly.Polyline.geometry_name()
    POLYGON = sly.Polygon.geometry_name()
    BITMAP = sly.Bitmap.geometry_name()
    GRAPH_NODES = sly.GraphNodes.geometry_name()
    SMARTTOOL = "smarttool"
    DETECTOR = "detector"


APP_STATUS = {
    "ready": [api.app.Status.STARTED, api.app.Status.DEPLOYED],
    "not_ready": [
        api.app.Status.CONSUMED,
        api.app.Status.QUEUED,
    ],
    "stopped": [
        api.app.Status.STOPPED,
        api.app.Status.ERROR,
        api.app.Status.TERMINATING,
        api.app.Status.FINISHED,
    ],
}


class AppParameterDescription:
    def __init__(
        self,
        title: str,
        description: str,
        type_: Literal["str", "int", "float", "bool"],
        options: List[str] = None,
        range_: Tuple = None,
        default=None,
    ):
        self.title = title
        self.description = description
        self.type = type_
        if options is None:
            options = []
        self.options = options
        if range_ is None:
            range_ = (None, None)
        self.range = range_
        self.default = default

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            title=data["title"],
            description=data["description"],
            type_=data["type"],
            options=data.get("options", []),
            range_=data.get("range", (None, None)),
            default=data.get("default", None),
        )


class NeuralNetwork:
    def __init__(
        self,
        name: str,
        module_id: int,
        title: str = None,
        description: str = "",
        cloud_url: str = None,
        params: Dict[str, AppParameterDescription] = None,
    ):
        self.name = name
        self.title = title if title is not None else name
        self.description = description
        self.cloud_url = cloud_url
        self.module_id = module_id
        self.params = params if params is not None else {}

        self.module_info = sly.api.app_api.ModuleInfo.from_json(api.app.get_info(module_id))


class NN:
    MIX_FORMER = NeuralNetwork(
        name="mix_former",
        module_id=api.app.get_ecosystem_module_id("supervisely-ecosystem/mixformer/serve/serve"),
        title="MixFormer",
        description="",
        cloud_url=ENV.mixformer_url(),
        params={
            "modelName": AppParameterDescription(
                title="Model",
                description="Select Model",
                type_="str",
                options=[
                    ("mixformer_vit_online", "MixViT-Large"),
                    ("mixformer_convmae_online", "MixViT - L(ConvMAE)"),
                ],
            )
        },
    )
    XMEM = NeuralNetwork(
        name="xmem",
        module_id=api.app.get_ecosystem_module_id(
            "supervisely-ecosystem/xmem/supervisely_integration/serve"
        ),
        title="XMem",
        description="",
        cloud_url=ENV.xmem_url(),
        params={},
    )
    CO_TRACKER = NeuralNetwork(
        name="co_tracker",
        module_id=api.app.get_ecosystem_module_id(
            "supervisely-ecosystem/co-tracker/supervisely_integration/serve"
        ),
        title="CoTracker",
        description="",
        cloud_url=ENV.cotracker_url(),
        params={
            "modelName": AppParameterDescription(
                title="Model",
                description="Select Model",
                type_="str",
                options=[
                    ("cotracker_stride_4_wind_8.pth", "cotracker_stride_4_wind_8.pth"),
                    ("cotracker_stride_4_wind_12.pth", "cotracker_stride_4_wind_12.pth"),
                    ("cotracker_stride_8_wind_16.pth", "cotracker_stride_8_wind_16.pth"),
                ],
            )
        },
    )
    CLICKSEG = NeuralNetwork(
        name="clickseg",
        module_id=api.app.get_ecosystem_module_id("supervisely-ecosystem/serve-clickseg"),
        title="ClickSeg",
        description="",
        cloud_url=ENV.clickseg_url(),
        params={},
    )
    YOLOV8 = NeuralNetwork(
        name="yolov8",
        module_id=api.app.get_ecosystem_module_id("supervisely-ecosystem/yolov8/serve"),
        title="YOLOv8",
        description="",
        cloud_url=ENV.yolov8_url(),
        params={},
    )
    SAM2 = NeuralNetwork(
        name="sam2",
        module_id=api.app.get_ecosystem_module_id("supervisely-ecosystem/serve-segment-anything-2"),
        title="Segment Anything 2",
        description="",
        cloud_url="",
        params={},
    )


nns = [NN.MIX_FORMER, NN.XMEM, NN.CO_TRACKER, NN.CLICKSEG, NN.YOLOV8, NN.SAM2]
geometry_nn = {
    GEOMETRY_NAME.RECTANGLE: [NN.MIX_FORMER],
    GEOMETRY_NAME.POINT: [NN.CO_TRACKER],
    GEOMETRY_NAME.POLYLINE: [NN.CO_TRACKER],
    GEOMETRY_NAME.POLYGON: [NN.CO_TRACKER],
    GEOMETRY_NAME.GRAPH_NODES: [NN.CO_TRACKER],
    GEOMETRY_NAME.BITMAP: [NN.XMEM],
    GEOMETRY_NAME.SMARTTOOL: [NN.CLICKSEG, NN.SAM2],
    GEOMETRY_NAME.DETECTOR: [NN.YOLOV8],
}


current_tracks = {}


def get_url_for_geometry(geometry_name: str) -> str:
    nn = geometry_nn.get(geometry_name, None)
    if nn is None:
        raise KeyError(f"Unknown geometry name: {geometry_name}")
    return nn[0].cloud_url
