import html

from supervisely.app.widgets import Button, Container, Dialog, Field, Text


class ModelConfiguration:
    def __init__(self, row):
        self.row = row
        self.empty = Text("No compatible sessions yet. Deploy a model to get started.")
        self.deployment = Container([row.deploy_app.get_content()])
        self.deployment.hide()
        self.deploy = Button("Deploy new", button_type="text", button_size="small")
        self.url = Field(row.nn_url_input, title="Model URL")
        self.interpolation = Text("Fill between manual keyframes without a serving session.")
        self.apply = Button("Use model", button_size="small")
        cancel = Button("Cancel", button_type="text", button_size="small")
        refresh = Button("Refresh sessions", button_type="text", button_size="small")
        self.error = Text("")
        self.error.hide()
        self.dialog = Dialog(title=f"Configure {row.title}", content=Container([
            Text(html.escape(row.description)), self.empty, row.selector,
            self.url, self.interpolation, row.session_link,
            Container([self.deploy, refresh], direction="horizontal", overflow="wrap"),
            self.deployment, self.error,
            Container([cancel, self.apply], direction="horizontal", overflow="wrap"),
        ]))
        row.selector.value_changed(lambda value: self._update_controls())
        self.deploy.click(self._toggle_deployment)
        self.apply.click(self._apply)
        cancel.click(self.dialog.hide)
        refresh.click(row.update_nn)
        self._update_controls()

    def open(self):
        self.row.selector.set_value(self.row._selected_value)
        self.row.nn_url_input.set_value(self.row._selected_url)
        self.deployment.hide()
        self.error.hide()
        self._update_controls()
        self.dialog.show()

    def _toggle_deployment(self):
        self.deployment.show() if self.deployment.is_hidden() else self.deployment.hide()
        self.error.hide()
        self._update_controls()

    def _update_controls(self):
        selected = self.row.selector.get_value()
        deploying = not self.deployment.is_hidden()
        self.empty.hide() if self.row._sessions else self.empty.show()
        self.row.selector.show() if self.row._session_items else self.row.selector.hide()
        self.url.show() if selected == "url" else self.url.hide()
        self.interpolation.show() if selected == "interpolation" else self.interpolation.hide()
        self.deploy.text = "Cancel deployment" if deploying else "Deploy new"
        self.apply.hide() if deploying or selected is None else self.apply.show()
        self.apply.text = "Use interpolation" if selected == "interpolation" else "Use model"

    def _apply(self):
        value = self.row.selector.get_value()
        if value not in dict(self.row._session_items) or (
            isinstance(value, int) and value not in self.row._sessions
        ):
            self.error.set("Select an available session or deploy a new model.", "warning")
            self.error.show()
            return
        if value == "url" and not self.row.nn_url_input.get_value().strip():
            self.error.set("Enter the model URL.", "warning")
            self.error.show()
            return
        self.row._selected_value = value
        self.row._selected_url = self.row.nn_url_input.get_value().strip()
        self.row._selection_changed()
        self.dialog.hide()
