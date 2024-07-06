from supervisely.app.widgets import (
    Container,
    Text,
)

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

layout = Container(
    widgets=[
        Container(widgets=[select_nn_settings_text, select_nn_settings_description_text], gap=5),
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
