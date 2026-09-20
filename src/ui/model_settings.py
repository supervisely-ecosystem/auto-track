import html
import yaml
from supervisely.app.widgets import Button, Checkbox, Container, Dialog, Editor, Field, InputNumber, Text


class ModelSettings:
    def __init__(self, title, editable, parameters):
        self.values = {}
        self.defaults = {}
        self.extra_values = {}
        self._source = None
        self._editable = editable
        self._parameters = {}
        fields = self._build_parameters(parameters)
        self.editor = Editor(language_mode="yaml", readonly=not editable, restore_default_button=False)
        self.error = Text("")
        self.error.hide()
        self.button = Button("—", button_type="text", button_size="small")
        self.button.disable()
        restore = Button("Restore defaults", button_type="text")
        apply = Button("Apply", button_type="primary")
        self.dialog = Dialog(
            title=f"{title} — inference settings",
            content=Container([
                Text("Adjust inference settings." if editable else "Model settings are read-only."),
                self.editor, *fields, self.error,
                Container([restore, apply], direction="horizontal"),
            ]),
        )
        self.button.click(self._open)
        restore.click(lambda: self.editor.set_text(yaml.safe_dump(self.defaults)))
        apply.click(self._apply)

    def _build_parameters(self, parameters):
        fields = []
        for name, details in parameters.items():
            if details["type"] == "notification":
                continue
            if details["type"] == "bool":
                widget = Checkbox(content=details["title"], checked=details["default"])
            else:
                widget = InputNumber(
                    min=details.get("min", 0), max=details.get("max", 1),
                    step=details.get("step", 0.01), value=details["default"],
                )
            self._parameters[name] = widget
            self.extra_values[name] = details["default"]
            fields.append(Field(widget, title=details["title"], description=details["description"]))
        return fields

    def load(self, source, defaults):
        """Replace settings only when selecting a different model, preserving open drafts on refresh."""
        if source == self._source:
            return
        self._source = source
        self.defaults = defaults or {}
        self.values = dict(self.defaults)
        self.editor.set_text(yaml.safe_dump(self.values))
        self.error.hide()
        self.dialog.hide()

    def _open(self):
        self.editor.set_text(yaml.safe_dump(self.values))
        for name, widget in self._parameters.items():
            if isinstance(widget, Checkbox):
                widget.check() if self.extra_values[name] else widget.uncheck()
            else:
                widget.value = self.extra_values[name]
        self.error.hide()
        self.dialog.show()

    def _apply(self):
        try:
            values = yaml.safe_load(self.editor.get_text())
            if values is None:
                values = {}
            if not isinstance(values, dict):
                raise ValueError("Settings must be a YAML mapping of names to values.")
        except (yaml.YAMLError, ValueError) as error:
            self.error.set(f"Invalid settings: {html.escape(str(error))}", "error")
            self.error.show()
            return
        self.values = values
        self.extra_values = {
            name: widget.is_checked() if isinstance(widget, Checkbox) else widget.get_value()
            for name, widget in self._parameters.items()
        }
        self.dialog.hide()
