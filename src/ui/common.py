import supervisely as sly
from supervisely.app.widgets import Empty

import src.globals as g
from .classes import DeployAppParameters, DeployAppByGeometry, GeometryCard


GEOMETRIES = (
    (
        sly.Rectangle.geometry_name(),
        {
            "title": "Rectangle",
            "description": "Select NN model for Rectangle figures",
            "geometries": [sly.Rectangle.geometry_name()],
        },
    ),
    (
        sly.Point.geometry_name(),
        {
            "title": "Point",
            "description": "Select NN model for Point, Polyline, Ploygon and Keypoints figures",
            "geometries": [
                sly.Point.geometry_name(),
                sly.Polyline.geometry_name(),
                sly.Polygon.geometry_name(),
                sly.GraphNodes.geometry_name(),
            ],
        },
    ),
    (
        sly.Bitmap.geometry_name(),
        {
            "title": "Bitmap",
            "description": "Select NN model for Bitmap figures",
            "geometries": [sly.Bitmap.geometry_name()],
        },
    ),
    (
        "smarttool",
        {
            "title": "SmartTool",
            "description": (
                "Select NN model for SmartTool. Box and points for smarttool annotation"
                " will be predicted using NN models selected in the previous steps."
            ),
            "geometries": ["smarttool"],
        },
    ),
)
EMPTY = Empty()
DEPLOY_APPS_PARAMETERS = {nn.name: DeployAppParameters(nn) for nn in g.nns}
DEPLOY_APP_BY_GEOMETRY = {
    geometry_name: DeployAppByGeometry(geometry_name, DEPLOY_APPS_PARAMETERS)
    for geometry_name, _ in GEOMETRIES
}
GEOMETRY_CARDS = {
    geometry_name: GeometryCard(
        geometries=details["geometries"],
        title=details["title"],
        deploy_app=DEPLOY_APP_BY_GEOMETRY[geometry_name],
        description=details["description"],
    )
    for geometry_name, details in GEOMETRIES
}
