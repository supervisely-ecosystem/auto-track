import supervisely as sly
from supervisely.app.widgets import Empty

import src.globals as g
from .classes import DeployAppParameters, DeployAppByGeometry
from .model_row import ModelRow


GEOMETRIES = (
    (
        g.GEOMETRY_NAME.RECTANGLE,
        {
            "title": "Bounding Box",
            "description": "Track rectangular object boxes.",
            "geometries": [g.GEOMETRY_NAME.RECTANGLE],
            "interpolation_supported": True,
        },
    ),
    (
        g.GEOMETRY_NAME.POINT,
        {
            "title": "Point-based",
            "description": "Track points, polylines, polygons and skeletons.",
            "geometries": [
                g.GEOMETRY_NAME.POINT,
                g.GEOMETRY_NAME.POLYLINE,
                g.GEOMETRY_NAME.POLYGON,
                g.GEOMETRY_NAME.GRAPH_NODES,
            ],
            "interpolation_supported": True,
        },
    ),
    (
        sly.Bitmap.geometry_name(),
        {
            "title": "Mask",
            "description": "Track object masks across frames.",
            "geometries": [sly.Bitmap.geometry_name()],
            "interpolation_supported": False,
        },
    ),
    (
        g.GEOMETRY_NAME.SMARTTOOL,
        {
            "title": "Smart Tool",
            "description": (
                "Create masks interactively. Tracking also requires Bounding Box and Points models."
            ),
            "geometries": [g.GEOMETRY_NAME.SMARTTOOL],
            "extra_params": {
                "notification": {
                    "type": "notification",
                    "notification_type": "info",
                    "title": "Additional models required to enable tracking for SmartTool objects",
                    "description": "This model is alows to annotate objects using SmartTool. To enable SmartTool objects tracking, Models for Bounding Box and Point based geometries should be selected as well.",
                }
            },
            "interpolation_supported": False,
        },
    ),
    (
        g.GEOMETRY_NAME.DETECTOR,
        {
            "title": "Detector",
            "description": "Find new objects automatically. Also configure a tracker for their geometry.",
            "geometries": [g.GEOMETRY_NAME.DETECTOR],
            "extra_params": {
                "enabled": {
                    "type": "bool",
                    "title": "Enabled",
                    "description": "If enabled, detector NN will be used to detect objects and track them",
                    "default": True,
                },
                "threshold": {
                    "type": "float",
                    "title": "Matching Threshold",
                    "description": "Minimum IoU value for matching detected and tracked objects",
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                },
            },
            "interpolation_supported": False,
        },
    ),
    (
        g.GEOMETRY_NAME.ORIENTED_BBOX,
        {
            "title": "Oriented Box",
            "description": "Track rotated object boxes.",
            "geometries": [g.GEOMETRY_NAME.ORIENTED_BBOX],
            "interpolation_supported": True,
        },
    ),
)
GEOMETRIES = tuple(sorted(GEOMETRIES, key=lambda item: [
    g.GEOMETRY_NAME.RECTANGLE, g.GEOMETRY_NAME.POINT, g.GEOMETRY_NAME.SMARTTOOL,
    g.GEOMETRY_NAME.DETECTOR, sly.Bitmap.geometry_name(), g.GEOMETRY_NAME.ORIENTED_BBOX,
].index(item[0])))
GEOMETRY_ICONS = {
    g.GEOMETRY_NAME.RECTANGLE: "zmdi zmdi-crop-din",
    g.GEOMETRY_NAME.POINT: "zmdi zmdi-dot-circle-alt",
    g.GEOMETRY_NAME.SMARTTOOL: "zmdi zmdi-colorize",
    g.GEOMETRY_NAME.DETECTOR: "zmdi zmdi-center-focus-strong",
    sly.Bitmap.geometry_name(): "zmdi zmdi-brush",
    g.GEOMETRY_NAME.ORIENTED_BBOX: "zmdi zmdi-rotate-cw",
}
EMPTY = Empty()
DEPLOY_APPS_PARAMETERS = {nn.name: DeployAppParameters(nn) for nn in g.nns}
DEPLOY_APP_BY_GEOMETRY = {
    geometry_name: DeployAppByGeometry(geometry_name, details["title"], DEPLOY_APPS_PARAMETERS)
    for geometry_name, details in GEOMETRIES
}
GEOMETRY_CARDS = {
    geometry_name: ModelRow(
        geometries=details["geometries"],
        title=details["title"],
        deploy_app=DEPLOY_APP_BY_GEOMETRY[geometry_name],
        description=details["description"],
        extra_params=details.get("extra_params", {}),
        interpolation_supported=details.get("interpolation_supported", False),
        icon=GEOMETRY_ICONS[geometry_name]
    )
    for geometry_name, details in GEOMETRIES
}
