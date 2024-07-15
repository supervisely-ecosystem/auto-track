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
    )
    for geometry_name, details in GEOMETRIES
}
