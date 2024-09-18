import supervisely as sly
from supervisely.app.widgets import Empty, Card

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
            "description": "Select NN model for Point, Polyline, Ploygon and Keypoints figures",
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
                    "default": False,
                },
                "confidence": {
                    "type": "float",
                    "title": "Confidence threshold",
                    "description": "Confidence threshold for detector",
                    "default": 0.7,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
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
