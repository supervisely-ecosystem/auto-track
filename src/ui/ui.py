from typing import List, Union
from supervisely.app.widgets import (
    Container,
    Text,
    InputNumber,
    Card,
    Field,
    Switch,
    OneOf,
    Empty,
    Checkbox,
)
from supervisely import logger
import yaml

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
disappear_threshold = InputNumber(min=0.05, max=0.95, step=0.05, value=0.2)
disappear_threshold_field = Field(
    disappear_threshold,
    "Disappear threshold",
    "Threshold for object disappearance. If object's area is less than median area * threshold for N frames, object will be considered as disappeared.",
)
disappear_frames = InputNumber(min=1, max=100, step=1, value=5)
disappear_frames_field = Field(
    disappear_frames,
    "Disappear frames",
    "Number of frames to wait before considering object as disappeared.",
)
disappear_by_area_switch = Switch(
    switched=True,
    on_content=Container(widgets=[disappear_threshold_field, disappear_frames_field]),
    off_content=Empty(),
)
disappear_by_area_one_of = OneOf(disappear_by_area_switch)
disappear_by_area_field = Field(
    title="Disappear by area",
    description="If enabled, object will be considered as disappeared if its area is less than median area * threshold for N consecutive frames.",
    content=Container(widgets=[disappear_by_area_switch, disappear_by_area_one_of]),
)

disappear_by_distance_multiplier = InputNumber(min=0.5, max=100, step=0.1, value=5)
disappear_by_distance_multiplier_field = Field(
    disappear_by_distance_multiplier,
    "Distance deviation multiplier",
    "Multiplier for distance deviation threshold.",
)
disappear_by_distance_position_deviation = InputNumber(min=1.0, max=10, step=0.05, value=1)
disappear_by_distance_position_deviation_field = Field(
    disappear_by_distance_position_deviation,
    "Position Standard Deviation",
    "Represents how uncertain you are about the object's motion. Higher values mean you expect the object's position to vary unpredictably.",
)
disappear_by_distance_velocity_deviation = InputNumber(min=0.5, max=10, step=0.05, value=0.5)
disappear_by_distance_velocity_deviation_field = Field(
    disappear_by_distance_velocity_deviation,
    "Velocity Standard Deviation",
    "Represents how uncertain you are about the object's speed. Higher values mean you expect the object's speed to vary unpredictably.",
)
disappear_by_distance_measure = InputNumber(min=0.1, max=20, step=0.1, value=3)
disappear_by_distance_measure_field = Field(
    disappear_by_distance_measure,
    "Measurement Standard Deviation",
    "Represents how uncertain you are about your detections accuracy. Higher values mean you expect your measurements to be noisy and unreliable",
)

disappear_by_distance_switch = Switch(
    switched=True,
    on_content=Container(
        widgets=[
            disappear_by_distance_multiplier_field,
            disappear_by_distance_position_deviation_field,
            disappear_by_distance_velocity_deviation_field,
            disappear_by_distance_measure_field,
        ]
    ),
    off_content=Empty(),
)
disappear_by_distance_one_of = OneOf(disappear_by_distance_switch)
disappear_by_distance_field = Field(
    title="Disappear by distance",
    description="If enabled, object will be considered as disappeared if there is sudden distance deviation.",
    content=Container(widgets=[disappear_by_distance_switch, disappear_by_distance_one_of]),
)

disappear_parameters_switch = Switch(
    switched=True,
    on_content=Container(widgets=[disappear_by_area_field, disappear_by_distance_field]),
    off_content=Empty(),
)
disappear_parameters_switch_field = Field(
    content=disappear_parameters_switch, title="Switch to turn on object disappear detection"
)
disappear_parameters_one_of = OneOf(disappear_parameters_switch)


disappear_parameters_inputs: List[Union[Switch, InputNumber]] = [
    disappear_parameters_switch,
    disappear_by_area_switch,
    disappear_by_distance_switch,
    disappear_threshold,
    disappear_frames,
    disappear_by_distance_multiplier,
    disappear_by_distance_position_deviation,
    disappear_by_distance_velocity_deviation,
    disappear_by_distance_measure,
]

for input_ in disappear_parameters_inputs:
    input_.value_changed(
        lambda *args: logger.debug(
            "Disappear parameters changed",
            extra={"disappear_parameters": get_disappear_parameters()},
        )
    )


disappear_parameters_card = Card(
    title="Disappear parameters",
    description="Parameters for object disappearance detection.",
    content=Container(
        widgets=[
            disappear_parameters_switch_field,
            disappear_parameters_one_of,
        ]
    ),
    collapsable=True,
)
disappear_parameters_card.collapse()

layout = Container(
    widgets=[
        Container(widgets=[select_nn_settings_text, select_nn_settings_description_text], gap=5),
        *[card.card for card in GEOMETRY_CARDS.values()],
        disappear_parameters_card,
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
        extra_params = geometry_card.get_extra_params()
        for geometry_name in geometry_card.geometries:
            if selector_value == "url":
                url = geometry_card.nn_url_input.get_value()
                settings[geometry_name] = {"url": url}
            else:
                session = app_selector.get_value()
                settings[geometry_name] = {"task_id": session}
            inf_settings = geometry_card.get_inference_settings().get_text()
            if inf_settings == "":
                inf_settings = {}
            else:
                inf_settings = yaml.safe_load(inf_settings)
            settings[geometry_name]["inference_settings"] = inf_settings
            settings[geometry_name]["extra_params"] = {
                name: widget.is_checked() if isinstance(widget, Checkbox) else widget.get_value()
                for name, widget in extra_params.items()
            }

    return settings


def get_disappear_parameters():
    return {
        "enabled": disappear_parameters_switch.is_switched(),
        "disappear_by_area": {
            "enabled": disappear_by_area_switch.is_switched(),
            "threshold": disappear_threshold.get_value(),
            "frames": disappear_frames.get_value(),
        },
        "disappear_by_distance": {
            "enabled": disappear_by_distance_switch.is_switched(),
            "multiplier": disappear_by_distance_multiplier.get_value(),
            "position_deviation": disappear_by_distance_position_deviation.get_value(),
            "velocity_deviation": disappear_by_distance_velocity_deviation.get_value(),
            "measure": disappear_by_distance_measure.get_value(),
        },
    }
