import html
import threading
import time

import requests
from supervisely import logger
from supervisely.app.widgets import Button, Input, Select, Text
from supervisely.nn.inference.session import Session

import src.globals as g
from .model_settings import ModelSettings
from .model_configuration import ModelConfiguration


class ModelRow:
    def __init__(self, geometries, title, deploy_app, description="", extra_params=None,
                 interpolation_supported=False, icon="zmdi zmdi-shape"):
        self.title = title
        self.description = description
        self.icon = icon
        self.geometries = geometries
        self.deploy_app = deploy_app
        self._interpolation_supported = interpolation_supported
        self._lock = threading.RLock()
        self._sessions = {}
        self._session_items = []
        self._pending_session_id = None
        self._selected_source = None
        self._selected_value = None
        self._selected_url = g.get_url_for_geometry(geometries[0])
        self.settings = ModelSettings(title, geometries[0] == "detector", extra_params or {})
        self.selector = Select(items=[], placeholder="Choose a model…", filterable=True, width_px=260)
        self.status = Text("Choose an existing model or deploy a new one.", status="info")
        self.session_link = Text("")
        self.session_link.hide()
        self.nn_url_input = Input(value=g.get_url_for_geometry(geometries[0]), placeholder="Model URL")
        self.model_name = Text("—", status="text")
        self.configure_button = Button("Configure", button_type="text", button_size="small")
        self.configuration = ModelConfiguration(self)
        self.configure_button.click(self.configuration.open)
        self.deploy_app.set_on_deploy_callback(self._select_deployed_session)
        threading.Thread(target=self._auto_refresh, daemon=True).start()

    def get_settings(self):
        selected = self._selected_value
        if selected == "interpolation":
            settings = {"interpolation": True}
        elif selected == "url":
            settings = {"url": self._selected_url}
        else:
            settings = {"task_id": selected}
        return {**settings, "inference_settings": dict(self.settings.values),
                "extra_params": dict(self.settings.extra_values)}

    def update_nn(self):
        with self._lock:
            selected = self._selected_value
            draft = self.selector.get_value()
            sessions = {}
            labels = []
            for nn in self.deploy_app.get_neural_networks():
                for session in g.api.app.get_sessions(
                    g.team_id, module_id=nn.module_id,
                    statuses=g.APP_STATUS["ready"] + g.APP_STATUS["not_ready"], with_shared=True,
                ):
                    sessions[session.task_id] = session
                    labels.append((session.task_id, f"{nn.title} · #{session.task_id}"))
            if selected is not None and selected not in sessions and isinstance(selected, int):
                labels.append((selected, f"Session #{selected} · unavailable"))
            if self._interpolation_supported:
                labels.append(("interpolation", "Interpolation"))
            if g.ENV.is_cloud():
                labels.append(("url", "Model URL"))
            self._sessions = sessions
            if labels != self._session_items:
                self.selector.set(items=[Select.Item(value, label) for value, label in labels])
                self.selector.set_value(draft)
                self._session_items = labels
            self._selection_changed()
            self.configuration._update_controls()

    def _select_deployed_session(self, task_id):
        with self._lock:
            self._pending_session_id = task_id
            self._selected_value = task_id
            self.configuration.dialog.hide()
            self.selector.set_value(task_id)
            self.update_nn()

    def _selection_changed(self):
        with self._lock:
            selected = self._selected_value
            source = (selected, self._selected_url if selected == "url" else None)
            if source != self._selected_source:
                self.settings.load(("selected", source), {})
                self._selected_source = source
            self.settings.button.disable()
            self.session_link.hide()
            self.configure_button.text = "Configure" if selected is None else "Change"
            labels = dict(self._session_items)
            name = labels.get(selected, "Interpolation" if selected == "interpolation" else "Model URL" if selected == "url" else "—")
            self.model_name.set(html.escape(name), "text")
            self.settings.button.text = html.escape(name)
            if selected == "interpolation":
                self.settings.load(selected, {})
                self._set_status("Ready", "success")
            elif selected == "url":
                self._load_url_settings()
            elif selected is None:
                self.settings.load(None, {})
                self._set_status("Not configured", "text")
            else:
                self._load_session_settings(selected)

    def _load_session_settings(self, task_id):
        session = self._sessions.get(task_id)
        url = html.escape(f"{g.api.server_address}{g.api.app.get_url(task_id)}", quote=True)
        self.session_link.set(f'<a href="{url}" target="_blank" rel="noopener noreferrer">Open session ↗</a>', "text")
        self.session_link.show()
        if session is None:
            message = "Unavailable · open the session or choose another model"
            if task_id == self._pending_session_id:
                task = g.api.task.get_info_by_id(task_id)
                if task is not None and task.status not in g.APP_STATUS["stopped"]:
                    message = "Starting · open the session to follow deployment"
                else:
                    self._pending_session_id = None
                    message = "Deployment stopped · open the session to check its logs"
            self._set_status("Starting" if message.startswith("Starting") else "Unavailable",
                             "info" if message.startswith("Starting") else "error")
            return
        if session.status in g.APP_STATUS["not_ready"]:
            self._set_status("Starting", "info")
            return
        try:
            serving = Session(g.api, task_id)
            if not serving.is_model_deployed():
                self._set_status("Needs setup", "warning")
                self.session_link.set(f'<a href="{url}" target="_blank" rel="noopener noreferrer">Finish setup ↗</a>', "text")
                return
            defaults = serving.get_default_inference_settings()
            self.settings.load((task_id, "ready"), defaults)
            self.settings.button.enable()
            self._set_status("Ready", "success")
            self._pending_session_id = None
        except Exception:
            logger.warning("Unable to check model session", exc_info=True)
            self._set_status("Unavailable", "error")

    def _load_url_settings(self):
        url = self._selected_url.strip().rstrip("/")
        if not url:
            self.settings.load(("url", url), {})
            self._set_status("Needs setup", "info")
            return
        try:
            response = requests.post(
                f"{url}/smart_segmentation_batch",
                json={"state": {}, "context": {}, "server_address": g.api.server_address,
                      "api_token": g.api.token}, timeout=5,
            )
            response.raise_for_status()
            defaults = response.json()["settings"]
            if isinstance(defaults, str):
                import yaml
                defaults = yaml.safe_load(defaults)
            self.settings.load(("url", url), defaults)
            self.settings.button.enable()
            self._set_status("Ready", "success")
        except Exception:
            logger.warning("Unable to load model URL settings", exc_info=True)
            self._set_status("Unavailable", "error")

    def _set_status(self, label, status):
        if status == "text":
            label = f'<i class="zmdi zmdi-circle-o mr5" aria-hidden="true"></i>{label}'
        self.status.set(label, status)

    def _auto_refresh(self):
        while True:
            time.sleep(60)
            try:
                self.update_nn()
            except Exception:
                logger.warning("Unable to refresh model sessions", exc_info=True)
                self._set_status("Check connection", "warning")
