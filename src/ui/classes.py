from typing import Callable, Dict, List, Tuple

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
    Text,
)

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
        self, geometries: List[str], title: str, deploy_app: DeployAppByGeometry, description=""
    ):
        self.geometries = geometries
        self.title = title
        self.description = description
        self.deploy_app = deploy_app
        self.card = None
        self.create_widgets()

    def create_widgets(self):
        self.select_nn_app = Select(items=[])
        select_nn_app_oneof = OneOf(self.select_nn_app)
        self.deploy_app_button = Button(
            '<i style="margin-right: 5px" class="zmdi zmdi-fire"></i>Deploy new model',
            button_type="success",
            button_size="small",
        )
        refresh_nn_app_button = Button(
            "",
            icon="zmdi zmdi-refresh",
            button_type="success",
            button_size="small",
            style="margin-left: 5px",
        )
        deploy_app_dialog = self.deploy_app.get_dialog()
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

        @self.deploy_app_button.click
        def on_deploy_app_button_click():
            deploy_app_dialog.show()

        @self.select_nn.value_changed
        def on_select_nn_changed(value):
            self.update_nn()

        @refresh_nn_app_button.click
        def on_refresh_nn_app_button_click():
            self.update_nn()

        # to update selectors when app is deployed
        self.deploy_app.set_on_deploy_callback(self.update_nn)

        self.card = Card(
            title=self.title,
            description=self.description,
            content=Container(
                widgets=[
                    Container(
                        widgets=[
                            select_nn_field,
                            select_nn_one_of,
                            Flexbox(widgets=[self.deploy_app_button, refresh_nn_app_button], gap=0),
                            deploy_app_dialog,
                        ],
                    ),
                    Empty(),
                ],
                direction="horizontal",
            ),
        )

    def get_widget(self) -> Card:
        return self.card

    def get_selectors(self) -> Tuple[Select, Select]:
        return self.select_nn, self.select_nn_app

    def update_nn(self):
        nns = self.deploy_app.get_neural_networks()
        nn_selector, session_selector = self.get_selectors()
        items = []
        for nn in nns:
            module_id = nn.module_id
            sessions = g.api.app.get_sessions(
                g.team_id, module_id=module_id, statuses=[g.api.app.Status.STARTED]
            )
            items.extend(
                [
                    Select.Item(session.task_id, f"{nn.title}: {session.task_id}", content=EMPTY)
                    for session in sessions
                ]
            )
        session_selector.set(items=items)
        if len(items) == 0:
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
