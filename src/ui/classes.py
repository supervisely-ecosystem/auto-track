from typing import Callable, Dict

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
    Text,
    Checkbox,
)
from supervisely import logger

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
        self.create_widgets()
        self._on_deploy_callback = lambda task_id: None

    def create_widgets(self):
        self._nn_selector = Select(
            items=[
                Select.Item(nn.name, nn.title, self._deploy_apps_parameters[nn.name].get_widget())
                for nn in self._nns
            ]
        )
        self._nn_parameters = OneOf(self._nn_selector)

        self._agent_selector = AgentSelector(
            g.team_id, show_only_gpu=True, show_only_running=True, compact=True
        )
        self._deploy_button = Button(
            '<i style="margin-right: 5px" class="zmdi zmdi-fire"></i>Deploy model',
            button_type="success",
            button_size="small",
        )
        self._deploy_button.click(self._deploy)
        self._deploy_button.disable()

        self._deployment_status = Text("")
        self._deployment_status.hide()
        self._content = Container(widgets=[
            Field(title="Model", content=self._nn_selector),
            Field(title="GPU agent", content=self._agent_selector),
            self._nn_parameters, self._deploy_button, self._deployment_status,
        ])

        @self._agent_selector.value_changed
        def _on_agent_changed(value):
            if value is None:
                self._deploy_button.disable()
            else:
                self._deploy_button.enable()

    def _deploy(self):
        self._deploy_button.disable()
        self._deploy_button.loading = True
        self._deployment_status.hide()
        try:
            selected_nn = self._deploy_apps_parameters[self._nn_selector.get_value()]
            session = g.api.app.start(
                agent_id=self._agent_selector.get_value(), module_id=selected_nn.nn.module_id,
                workspace_id=g.workspace_id, is_branch=False,
                params=selected_nn.get_params(), task_name="run-from-auto-track",
            )
        except Exception:
            logger.warning("Unable to start model session", exc_info=True)
            self._deployment_status.set("Could not start the model. Check the agent and try again.", "error")
            self._deployment_status.show()
            return
        finally:
            self._deploy_button.loading = False
            self._deploy_button.enable()
        self._on_deploy_callback(session.task_id)

    def set_on_deploy_callback(self, callback: Callable):
        self._on_deploy_callback = callback

    def get_content(self):
        return self._content

    def get_neural_networks(self):
        return self._nns
