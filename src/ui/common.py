import supervisely as sly
from supervisely.app.widgets import Empty

import src.globals as g
from .classes import DeployAppParameters, DeployAppByGeometry, GeometryCard


GEOMETRIES = (
    (
        g.GEOMETRY_NAME.RECTANGLE,
        {
            "title": "Bounding Box",
            "description": "Select NN model for Bounding box figures",
            "geometries": [g.GEOMETRY_NAME.RECTANGLE],
        },
    ),
    (
        g.GEOMETRY_NAME.POINT,
        {
            "title": "Point based geometries",
            "description": "Select NN model for Point, Polyline, Polygon and Keypoints figures",
            "geometries": [
                g.GEOMETRY_NAME.POINT,
                g.GEOMETRY_NAME.POLYLINE,
                g.GEOMETRY_NAME.POLYGON,
                g.GEOMETRY_NAME.GRAPH_NODES,
            ],
        },
    ),
    (
        sly.Bitmap.geometry_name(),
        {
            "title": "Mask",
            "description": "Select NN model for Mask figures",
            "geometries": [sly.Bitmap.geometry_name()],
        },
    ),
    (
        g.GEOMETRY_NAME.SMARTTOOL,
        {
            "title": "Smart Tool",
            "description": (
                "Select NN model for SmartTool. When tracking Box and points for smarttool "
                "annotation will be predicted using NN models selected in the previous steps."
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
        },
    ),
    (
        g.GEOMETRY_NAME.DETECTOR,
        {
            "title": "Detector",
            "description": "Select NN model for detection",
            "geometries": [g.GEOMETRY_NAME.DETECTOR],
            "extra_params": {
                "enabled": {
                    "type": "bool",
                    "title": "Enabled",
                    "description": "If enabled, detector NN will be used to detect objects and track them",
                    "default": True,
                },
                "mode": {
                    "type": "select",
                    "title": "Tracking by detection mode",
                    "description": "Select the mode for tracking by detection. If 'tracker Model' is selected, the model selected above will be used to track objects. If 'BoT-SORT' is selected, BoT-SORT algorithm will be used for tracking.",
                    "items": [
                        {"value": "model", "label": "tracker Model"},
                        {"value": "botsort", "label": "BoT-SORT"},
                    ],
                    "default": "botsort",
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
        },
    ),
)
EMPTY = Empty()
DEPLOY_APPS_PARAMETERS = {nn.name: DeployAppParameters(nn) for nn in g.nns}
DEPLOY_APP_BY_GEOMETRY = {
    geometry_name: DeployAppByGeometry(geometry_name, details["title"], DEPLOY_APPS_PARAMETERS)
    for geometry_name, details in GEOMETRIES
}
GEOMETRY_CARDS = {
    geometry_name: GeometryCard(
        geometries=details["geometries"],
        title=details["title"],
        deploy_app=DEPLOY_APP_BY_GEOMETRY[geometry_name],
        description=details["description"],
        extra_params=details.get("extra_params", {}),
    )
    for geometry_name, details in GEOMETRIES
}
