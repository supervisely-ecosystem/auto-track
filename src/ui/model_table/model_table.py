from supervisely.app.widgets import Widget


class ModelTable(Widget):
    """Render model controls with the platform table theme, SDK widgets, and geometry icons."""

    def __init__(self, rows):
        self.rows = rows
        super().__init__(file_path=__file__)

    def get_json_data(self):
        return {"rows": [{"index": index, "title": row.title, "icon": row.icon}
                         for index, row in enumerate(self.rows)]}

    def get_json_state(self):
        return {}
