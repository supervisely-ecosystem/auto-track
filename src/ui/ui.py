from supervisely.app.widgets import Container, Text, InputNumber, Card, Field

import src.globals as g
from .common import GEOMETRY_CARDS


select_nn_settings_text = Text(
    "<b>Select neural network settings:</b>",
    font_size=21,
)
select_nn_settings_description_text = Text(
    (
        "<b>Select NN model settings for each geometry type. "
        "This model will be used to track objects in Video Labeling Tool.</b>"
    ),
    font_size=15,
)
disapear_parameters = Text()
disapear_threshold = InputNumber(min=0.05, max=0.95, step=0.05, value=0.5)
disapear_threshold_field = Field(
    disapear_threshold,
    "Disapear threshold",
    "Threshold for object disapearance. If object's area is less than median area * threshold for N frames, object will be considered as disapeared.",
)
disapear_frames = InputNumber(min=1, max=100, step=1, value=5)
disapear_frames_field = Field(
    disapear_frames,
    "Disapear frames",
    "Number of frames to wait before considering object as disapeared.",
)
multiplier = InputNumber(min=0.5, max=100, step=0.1, value=5)
multiplier_field = Field(
    multiplier,
    "Distance deviation multiplier",
    "Multiplier for distance deviation threshold. If object's distance deviation from expected position is greater than max distance * multiplier, object will be considered as moved.",
)
disapear_parameters_card = Card(
    title="Disapear parameters",
    content=Container(widgets=[disapear_threshold_field, disapear_frames_field, multiplier_field]),
)

layout = Container(
    widgets=[
        Container(widgets=[select_nn_settings_text, select_nn_settings_description_text], gap=5),
        disapear_parameters_card,
        *[card.card for card in GEOMETRY_CARDS.values()],
    ]
)


def update_all_nn():
    for geometry_card in GEOMETRY_CARDS.values():
        geometry_card.update_nn()


def get_nn_settings():
    settings = {}
    for geometry_card in GEOMETRY_CARDS.values():
        selector, app_selector = geometry_card.get_selectors()
        selector_value = selector.get_value()
        for geometry_name in geometry_card.geometries:
            if selector_value == "url":
                url = geometry_card.nn_url_input.get_value()
                settings[geometry_name] = {"url": url}
            else:
                session = app_selector.get_value()
                settings[geometry_name] = {"task_id": session}

    return settings


def get_disapear_parameters():
    return (disapear_threshold.get_value(), disapear_frames.get_value(), multiplier.get_value())
