import os
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


nns = [NN.MIX_FORMER, NN.XMEM, NN.CO_TRACKER, NN.CLICKSEG]
geometry_nn = {
    sly.Rectangle.geometry_name(): [NN.MIX_FORMER],
    sly.Point.geometry_name(): [NN.CO_TRACKER],
    sly.Polyline.geometry_name(): [NN.CO_TRACKER],
    sly.Polygon.geometry_name(): [NN.CO_TRACKER],
    sly.GraphNodes.geometry_name(): [NN.CO_TRACKER],
    sly.Bitmap.geometry_name(): [NN.XMEM],
    "smarttool": [NN.CLICKSEG],
}


current_tracks = {}


def get_url_for_geometry(geometry_name: str) -> str:
    nn = geometry_nn.get(geometry_name, None)
    if nn is None:
        raise KeyError(f"Unknown geometry name: {geometry_name}")
    return nn[0].cloud_url
