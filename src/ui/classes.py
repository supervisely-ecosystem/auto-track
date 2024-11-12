import time
from typing import Callable, Dict, List, Tuple, Union

from supervisely.app.widgets import (
    Container,
    Select,
    Field,
    OneOf,
    Empty,
    Button,
    AgentSelector,
    Input,
    InputNumber,
    Checkbox,
    Dialog,
    Card,
    Flexbox,
    Editor,
    NotificationBox,
)
from supervisely import logger
from supervisely.nn.inference.session import Session
import requests
import yaml

import src.globals as g


EMPTY = Empty()


class AppParameterUI:
    def __init__(self, parameter_name: str, app_parameter_description: g.AppParameterDescription):
        self.parameter_name = parameter_name
        self.app_parameter_description = app_parameter_description
        self.widget = None
        self._get_value = None
        self._craete_widgets()

    def _craete_widgets(self):
        if len(self.app_parameter_description.options) > 0:
            items = [Select.Item(*option) for option in self.app_parameter_description.options]
            self.widget = Select(items=items)
            self._get_value = self.widget.get_value
            if self.app_parameter_description.default is not None:
                self.widget.set_value(self.app_parameter_description.default)
        elif self.app_parameter_description.type == "str":
            self.widget = Input()
            self._get_value = self.widget.get_value
            if self.app_parameter_description.default is not None:
                self.widget.set_value(self.app_parameter_description.default)
        elif self.app_parameter_description.type == "int":
            min_, max_ = self.app_parameter_description.range
            self.widget = InputNumber(min=min_, max=max_, step=1)
            self._get_value = self.widget.get_value
            if self.app_parameter_description.default is not None:
                self.widget.value = self.app_parameter_description.default
        elif self.app_parameter_description.type == "float":
            min_, max_ = self.app_parameter_description.range
            self.widget = InputNumber(min=min_, max=max_, step=0.0001)
            self._get_value = self.widget.get_value
            if self.app_parameter_description.default is not None:
                self.widget.value = self.app_parameter_description.default
        elif self.app_parameter_description.type == "bool":
            self.widget = Checkbox(content=self.app_parameter_description.title)
            self._get_value = self.widget.is_checked
            if self.app_parameter_description.default is not None:
                if self.app_parameter_description.default:
                    self.widget.check()
                else:
                    self.widget.uncheck()
        self.widget_field = Field(
            title=self.app_parameter_description.title,
            description=self.app_parameter_description.description,
            content=self.widget,
        )

    def get_widget(self):
        return self.widget_field

    def get_value(self):
        return self._get_value()

    def to_json(self):
        return {self.parameter_name: self._get_value()}


class DeployAppParameters:
    def __init__(self, nn: g.NeuralNetwork):
        self.nn = nn

        self._params_ui = []
        for param_name, param_description in self.nn.params.items():
            self._params_ui.append(AppParameterUI(param_name, param_description))

        if len(self._params_ui) > 0:
            _params_field = Field(
                title="App Parameters",
                content=Container(widgets=[param.get_widget() for param in self._params_ui]),
            )
            self._widget = _params_field
        else:
            self._widget = EMPTY

    def get_widget(self):
        return self._widget

    def get_params(self):
        params = {}
        for param_ui in self._params_ui:
            params.update(param_ui.to_json())
        return params


class DeployAppByGeometry:
    def __init__(
        self,
        geometry_name: str,
        title: str,
        deploy_apps_parameters: Dict[str, DeployAppParameters],
    ):
        self._geometry_name = geometry_name
        self._title = title
        self._nns = g.geometry_nn[geometry_name]
        self._deploy_apps_parameters = deploy_apps_parameters
        self._nn_selector = None
        self._nn_parameters = None
        self._dialog = None
        self.create_widgets()
        self._on_deploy_callback = lambda: None

    def create_widgets(self):
        self._nn_selector = Select(
            items=[
                Select.Item(nn.name, nn.title, self._deploy_apps_parameters[nn.name].get_widget())
                for nn in self._nns
            ]
        )
        self._nn_parameters = OneOf(self._nn_selector)

        self._agent_selector = AgentSelector(
            g.team_id, show_only_gpu=True, show_only_running=True, compact=False
        )
        self._deploy_button = Button(
            '<i style="margin-right: 5px" class="zmdi zmdi-fire"></i>Deploy model',
            button_type="success",
            button_size="small",
        )
        self._deploy_button.click(self.deploy)
        self._deploy_button.disable()

        self._dialog = Dialog(
            title=f"Deploy New App for {self._title}",
            content=Container(
                widgets=[
                    Field(title="Select NN model App", content=self._nn_selector),
                    Field(title="Select Agent", content=self._agent_selector),
                    self._nn_parameters,
                    self._deploy_button,
                ]
            ),
        )

        @self._agent_selector.value_changed
        def _on_agent_changed(value):
            if value is None:
                self._deploy_button.disable()
            else:
                self._deploy_button.enable()

    def deploy(self):
        # get and params
        selected_nn_name = self._nn_selector.get_value()
        selected_nn = self._deploy_apps_parameters[selected_nn_name].nn
        selected_nn_parameters = self._deploy_apps_parameters[selected_nn_name].get_params()

        # start app
        session = g.api.app.start(
            agent_id=self._agent_selector.get_value(),
            module_id=selected_nn.module_id,
            workspace_id=g.workspace_id,
            is_branch=False,
            params=selected_nn_parameters,
            task_name="run-from-auto-track",
        )
        # wait for app to start
        g.api.app.wait(session.task_id, target_status=g.api.app.Status.STARTED)

        # update nn for all geometries related to this nn
        self._on_deploy_callback()

        self._dialog.hide()

        return session.task_id

    def set_on_deploy_callback(self, callback: Callable):
        self._on_deploy_callback = callback

    def get_dialog(self):
        return self._dialog

    def get_neural_networks(self):
        return self._nns


class DeployedAppInfo:
    def __init__(self, task_id: int):
        self.task_id = task_id
        self.info = g.api.app.send_request(task_id, "get_session_info", {})
        self.inf_settings = g.api.app.send_request(task_id, "get_custom_inference_settings", {})

    def get_data(self):
        return {**self.info, **self.inf_settings}


class GeometryCard:
    def __init__(
        self,
        geometries: List[str],
        title: str,
        deploy_app: DeployAppByGeometry,
        description="",
        extra_params={},
    ):
        self.geometries = geometries
        self.title = title
        self.description = description
        self.deploy_app = deploy_app
        self.extra_params = extra_params
        self.inference_settings = None
        self.card = None
        self._nn_url_changed = time.monotonic()
        self.default_inference_settings = ""
        self.create_widgets()

    def create_widgets(self):
        self.select_nn_app = Select(items=[])
        select_nn_app_oneof = OneOf(self.select_nn_app)
        self.deploy_app_button = Button(
            '<i style="margin-right: 5px" class="zmdi zmdi-fire"></i>Deploy new model',
            button_type="success",
            button_size="small",
        )
        self.refresh_nn_app_button = Button(
            "",
            icon="zmdi zmdi-refresh",
            button_type="success",
            button_size="small",
            style="margin-left: 5px",
        )
        self.deploy_app_dialog = self.deploy_app.get_dialog()
        select_nn_app_container = Container(
            widgets=[
                self.select_nn_app,
                select_nn_app_oneof,
            ]
        )
        self.nn_url_input = Input(
            value=g.get_url_for_geometry(self.geometries[0]), placeholder="Enter url to model"
        )
        select_nn_items = [
            Select.Item("url", "Access to model by url", content=self.nn_url_input),
            Select.Item(
                "app",
                "Deployed model session",
                content=select_nn_app_container,
                disabled=True,
            ),
        ]
        if not g.ENV.is_cloud():
            select_nn_items = select_nn_items[1:]
        self.select_nn = Select(items=select_nn_items)
        select_nn_field = Field(self.select_nn, title="Select model for predictions")
        select_nn_one_of = OneOf(self.select_nn)
        self.app_status_ok = EMPTY
        self.app_status_not_ready = NotificationBox(
            "Model is not ready",
            "Model is not ready yet. Wait for the model to start before using the application or select another session",
            "warning",
        )
        self.app_status_not_ready.hide()

        @self.deploy_app_button.click
        def on_deploy_app_button_click():
            self.deploy_app_dialog.show()

        @self.select_nn.value_changed
        def on_select_nn_changed(value):
            self.update_nn()

        @self.refresh_nn_app_button.click
        def on_refresh_nn_app_button_click():
            self.update_nn()

        # to update selectors when app is deployed
        self.deploy_app.set_on_deploy_callback(self.update_nn)

        extra_params_widgets = {}
        for name, details in self.extra_params.items():
            if details["type"] == "bool":
                widget = Checkbox(content=details["title"], checked=details["default"])
                widget_field = Field(
                    widget, title=details["title"], description=details["description"]
                )
                extra_params_widgets[name] = (widget, widget_field)
            elif details["type"] == "float":
                widget = InputNumber(
                    min=details.get("min", 0),
                    max=details.get("max", 1),
                    step=details.get("step", 0.01),
                    value=details["default"],
                )
                widget_field = Field(
                    widget, title=details["title"], description=details["description"]
                )
                extra_params_widgets[name] = (widget, widget_field)
            elif details["type"] == "notification":
                widget = Empty()
                widget_field = NotificationBox(
                    title=details.get("title", ""),
                    description=details.get("description", ""),
                    box_type=details.get("notification_type", "info"),
                )
                extra_params_widgets[name] = (widget, widget_field)

        self.extra_params: Dict[str, Union[Checkbox, InputNumber]] = {
            name: widget for name, (widget, _) in extra_params_widgets.items()
        }
        for name, widget in self.extra_params.items():
            if hasattr(widget, "value_changed"):
                widget.value_changed(lambda *args: logger.debug("Extra parameters changed"))

        self.inference_settings = Editor(
            language_mode="yaml",
            readonly=self.geometries[0] != "detector",
            restore_default_button=False,
        )
        restore_inferene_settings_button = Button("Restore default", button_type="text")
        save_inference_settings_button = Button("Save", button_type="text")
        inference_settings_container = Container(
            widgets=[
                Flexbox(widgets=[restore_inferene_settings_button, save_inference_settings_button]),
                self.inference_settings,
            ],
            gap=2,
        )
        self.inference_settings_field = Field(
            inference_settings_container,
            title="Inference settings",
            description="Some Models allow user to configure the following parameters. If it is not editable, it means that the model does not support custom inference settings.",
        )

        @restore_inferene_settings_button.click
        def on_restore_inference_settings_button_click():
            self.inference_settings.set_text(self.default_inference_settings)

        @save_inference_settings_button.click
        def on_save_inference_settings_button_click():
            pass

        @self.nn_url_input.value_changed
        def on_nn_url_input_changed(value):
            this_nn_url_changed = time.monotonic()
            self._nn_url_changed = this_nn_url_changed
            time.sleep(1)
            if self._nn_url_changed == this_nn_url_changed:
                return
            try:
                r = requests.post(
                    f"{value}/smart_segmentation_batch",
                    json={
                        "state": {},
                        "context": {},
                        "server_address": g.api.server_address,
                        "api_token": g.api.token,
                    },
                    timeout=5,
                )
                r.raise_for_status()
                settings = r.json()["settings"]
                if not isinstance(settings, str):
                    settings = yaml.safe_dump(settings)
                self.inference_settings.set_text(settings)
                self.default_inference_settings = settings
            except Exception:
                logger.warning(f"Failed to get inference settings from {value}", exc_info=True)
                self.inference_settings.set_text("")
                self.default_inference_settings = ""

        self.card = Card(
            title=self.title,
            description=self.description,
            content=Container(
                widgets=[
                    Container(
                        widgets=[
                            select_nn_field,
                            select_nn_one_of,
                            Flexbox(
                                widgets=[self.deploy_app_button, self.refresh_nn_app_button], gap=0
                            ),
                            self.deploy_app_dialog,
                            *[x[1] for x in extra_params_widgets.values()],
                            self.app_status_not_ready,
                        ],
                    ),
                    self.inference_settings_field,
                ],
                direction="horizontal",
                fractions=[35, 65],
            ),
        )

    def get_widget(self) -> Card:
        return self.card

    def get_selectors(self) -> Tuple[Select, Select]:
        return self.select_nn, self.select_nn_app

    def get_extra_params(self) -> Dict[str, Union[Checkbox, InputNumber]]:
        return self.extra_params

    def get_inference_settings(self) -> Editor:
        return self.inference_settings

    def update_nn(self):
        self.refresh_nn_app_button.loading = True
        nns = self.deploy_app.get_neural_networks()
        nn_selector, session_selector = self.get_selectors()
        current_session = session_selector.get_value()
        items = []
        for nn in nns:
            module_id = nn.module_id
            sessions = g.api.app.get_sessions(
                g.team_id,
                module_id=module_id,
                statuses=g.APP_STATUS["ready"] + g.APP_STATUS["not_ready"],
            )
            items.extend(
                [
                    Select.Item(
                        session.task_id,
                        f"{nn.title}: {session.task_id}",
                        content=EMPTY,
                    )
                    for session in sessions
                ]
            )
        session_selector.set(items=items)
        if current_session in [item.value for item in items]:
            session_selector.set_value(current_session)
        if len(items) == 0 and g.ENV.is_cloud():
            items = nn_selector.get_items()
            changed = False
            for item in items:
                if item.value == "app" and not item.disabled:
                    item.disabled = True
                    changed = True
            if changed:
                nn_selector.set(items=items)
                nn_selector.set_value("url")
        else:
            items = nn_selector.get_items()
            changed = False
            for item in items:
                if item.value == "app" and item.disabled:
                    item.disabled = False
                    changed = True
            if changed:
                nn_selector.set(items=items)
                nn_selector.set_value("app")

        if not self.deploy_app_dialog.is_hidden():
            self.deploy_app_dialog.hide()

        if nn_selector.get_value() == "app":
            selected_session_id = session_selector.get_value()
            is_deployed = False
            settings = ""
            if selected_session_id is not None:
                try:
                    selected_session = Session(g.api, selected_session_id)
                    is_deployed = selected_session.is_model_deployed()
                    settings = selected_session.get_default_inference_settings()
                    settings = yaml.safe_dump(settings)
                except Exception:
                    logger.warning("Failed to get inference settings", exc_info=True)
                    settings = ""
                if is_deployed:
                    self.app_status_not_ready.hide()
                else:
                    self.app_status_not_ready.show()
            else:
                self.app_status_not_ready.hide()
            self.inference_settings.set_text(settings)
            self.default_inference_settings = settings
        self.refresh_nn_app_button.loading = False
