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
            "interpolation_supported": True,
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
            "interpolation_supported": True,
        },
    ),
    (
        sly.Bitmap.geometry_name(),
        {
            "title": "Mask",
            "description": "Select NN model for Mask figures",
            "geometries": [sly.Bitmap.geometry_name()],
            "interpolation_supported": False,
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
            "interpolation_supported": False,
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
            "title": "Oriented Bounding Box",
            "description": "Select NN model for Oriented Bounding Box figures",
            "geometries": [g.GEOMETRY_NAME.ORIENTED_BBOX],
            "interpolation_supported": True,
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
        intrepolation_supported=details.get("interpolation_supported", False)
    )
    for geometry_name, details in GEOMETRIES
}
