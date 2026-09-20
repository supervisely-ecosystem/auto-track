"""Exercise the real SDK widgets with a deterministic serving catalog."""
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from supervisely.app.widgets import Button, Dialog, Select, Text


UI_PATH = Path(__file__).resolve().parents[1] / "src" / "ui"
package = types.ModuleType("model_setup_checks")
package.__path__ = [str(UI_PATH)]
sys.modules[package.__name__] = package
globals_stub = types.ModuleType("src.globals")
globals_stub.AppParameterDescription = object
globals_stub.NeuralNetwork = object
with patch.dict(sys.modules, {"src.globals": globals_stub}):
    geometry = importlib.import_module("model_setup_checks.model_row")
    deployment = importlib.import_module("model_setup_checks.classes")


class ModelSetupTests(unittest.TestCase):
    def setUp(self):
        self.api = Mock(server_address="https://example.test")
        self.api.app.get_sessions.return_value = [types.SimpleNamespace(task_id=11, status="started")]
        self.api.app.get_url.side_effect = lambda task_id: f"/apps/sessions/{task_id}"
        globals_stub.api = self.api
        globals_stub.team_id = 9
        globals_stub.workspace_id = 8
        globals_stub.APP_STATUS = {"ready": ["started"], "not_ready": ["queued"], "stopped": ["error"]}
        globals_stub.ENV = types.SimpleNamespace(is_cloud=lambda: False)
        globals_stub.get_url_for_geometry = lambda name: ""
        self.serving = Mock()
        self.serving.is_model_deployed.return_value = True
        self.serving.get_default_inference_settings.return_value = {"confidence": 0.5}
        self.session_patch = patch.object(geometry, "Session", return_value=self.serving)
        self.session_patch.start()
        self.addCleanup(self.session_patch.stop)
        self.deploy = Mock()
        self.deploy.get_content.return_value = Text("Deploy")
        self.deploy.get_neural_networks.return_value = [types.SimpleNamespace(module_id=1, title="Model")]
        with patch.object(geometry.ModelRow, "_auto_refresh"):
            self.card = geometry.ModelRow(["detector"], "Detector", self.deploy)
        self.card.update_nn()

    def _select_ready_model(self):
        self.card._selected_value = 11
        self.card._selection_changed()

    def test_refresh_preserves_open_deploy_dialog_and_empty_selection(self):
        self.card.configuration.dialog.show()
        self.card.update_nn()
        self.assertFalse(self.card.configuration.dialog.is_hidden())
        self.assertIsNone(self.card.get_settings()["task_id"])

    def test_refresh_preserves_inference_draft_until_apply(self):
        self._select_ready_model()
        self.card.settings._open()
        self.card.settings.editor.set_text("confidence: 0.8")
        self.card.update_nn()
        self.assertFalse(self.card.settings.dialog.is_hidden())
        self.assertEqual(self.card.settings.editor.get_text(), "confidence: 0.8")
        self.assertEqual(self.card.get_settings()["inference_settings"], {"confidence": 0.5})
        self.card.settings._apply()
        self.assertEqual(self.card.get_settings()["inference_settings"], {"confidence": 0.8})

    def test_invalid_yaml_does_not_replace_applied_settings(self):
        self._select_ready_model()
        self.card.settings._open()
        for invalid in ("[invalid", "- not-a-mapping"):
            self.card.settings.editor.set_text(invalid)
            self.card.settings._apply()
            self.assertFalse(self.card.settings.dialog.is_hidden())
            self.assertFalse(self.card.settings.error.is_hidden())
            self.assertEqual(self.card.get_settings()["inference_settings"], {"confidence": 0.5})

    def test_deployment_selects_the_exact_created_session(self):
        callback = self.deploy.set_on_deploy_callback.call_args.args[0]
        callback(22)
        self.assertEqual(self.card.get_settings()["task_id"], 22)
        self.assertIn("Starting", self.card.status.get_json_data()["text"])
        self.assertIn("/apps/sessions/22", self.card.session_link.get_json_data()["text"])

    def test_switch_to_unreachable_session_clears_previous_model_settings(self):
        self._select_ready_model()
        self.api.app.get_sessions.return_value.append(types.SimpleNamespace(task_id=22, status="started"))
        self.card._selected_value = 22
        self.serving.is_model_deployed.side_effect = RuntimeError("Offline")
        self.card.update_nn()
        self.assertEqual(self.card.get_settings()["inference_settings"], {})
        self.assertEqual(self.card.get_settings()["task_id"], 22)

    def test_failed_deployment_does_not_keep_showing_starting(self):
        self.api.task.get_info_by_id.return_value = types.SimpleNamespace(status="error")
        self.card._select_deployed_session(22)
        self.assertIn("Unavailable", self.card.status.get_json_data()["text"])

    def test_unready_model_links_to_setup_then_becomes_ready(self):
        self.serving.is_model_deployed.return_value = False
        self._select_ready_model()
        self.assertIn("Finish setup", self.card.session_link.get_json_data()["text"])
        self.serving.is_model_deployed.return_value = True
        self.card.update_nn()
        self.assertEqual(self.card.get_settings()["inference_settings"], {"confidence": 0.5})
        self.assertIn("Ready", self.card.status.get_json_data()["text"])

    def test_interpolation_is_offered_only_for_supported_geometry(self):
        self.assertNotIn("interpolation", dict(self.card._session_items))
        with patch.object(geometry.ModelRow, "_auto_refresh"):
            row = geometry.ModelRow(["rectangle"], "Bounding box", self.deploy,
                                    interpolation_supported=True)
        row.update_nn()
        self.assertIn("interpolation", dict(row._session_items))
        row.selector.set_value("interpolation")
        row.configuration._apply()
        self.assertEqual(row.get_settings(), {
            "interpolation": True, "inference_settings": {}, "extra_params": {},
        })

    def test_model_selection_is_committed_only_on_apply(self):
        self.card.configuration.open()
        self.card.selector.set_value(11)
        self.card.update_nn()
        self.assertIsNone(self.card.get_settings()["task_id"])
        self.card.configuration.dialog.hide()
        self.assertIsNone(self.card.get_settings()["task_id"])
        self.card.configuration.open()
        self.card.selector.set_value(11)
        self.card.configuration._apply()
        self.assertEqual(self.card.get_settings()["task_id"], 11)

    def test_empty_catalog_opens_deploy_in_same_dialog(self):
        self.api.app.get_sessions.return_value = []
        self.card.update_nn()
        self.card.configuration.open()
        self.assertTrue(self.card.selector.is_hidden())
        self.assertTrue(self.card.configuration.deployment.is_hidden())
        self.card.configuration._toggle_deployment()
        self.assertFalse(self.card.configuration.deployment.is_hidden())
        self.assertTrue(self.card.configuration.apply.is_hidden())

    def test_inline_deployment_preserves_session_draft_across_refresh_and_cancel(self):
        self.card.configuration.open()
        self.card.selector.set_value(11)
        self.card.configuration._toggle_deployment()
        self.card.update_nn()
        self.assertFalse(self.card.selector.is_hidden())
        self.assertFalse(self.card.configuration.deployment.is_hidden())
        self.assertTrue(self.card.configuration.apply.is_hidden())
        self.assertEqual(self.card.selector.get_value(), 11)
        self.card.configuration._toggle_deployment()
        self.assertFalse(self.card.configuration.apply.is_hidden())
        self.assertIsNone(self.card.get_settings()["task_id"])
        self.card.configuration._apply()
        self.assertEqual(self.card.get_settings()["task_id"], 11)

    def test_queued_model_displays_starting_status(self):
        self.api.app.get_sessions.return_value = [types.SimpleNamespace(task_id=11, status="queued")]
        self.card._selected_value = 11
        self.card.update_nn()
        self.assertIn("Starting", self.card.status.get_json_data()["text"])
        self.assertTrue(self.card.settings.button.get_json_data()["disabled"])

    def test_deploy_hands_off_created_session_without_waiting_for_model(self):
        deploy = deployment.DeployAppByGeometry.__new__(deployment.DeployAppByGeometry)
        deploy._deploy_button = Button("Deploy")
        deploy._deployment_status = Text("")
        deploy._dialog = self.card.configuration.dialog
        deploy._nn_selector = Select(items=[Select.Item("model", "Model")])
        deploy._agent_selector = Mock()
        deploy._agent_selector.get_value.return_value = 3
        params = Mock(nn=types.SimpleNamespace(module_id=1))
        params.get_params.return_value = {}
        deploy._deploy_apps_parameters = {"model": params}
        deploy._on_deploy_callback = Mock()
        self.api.app.start.return_value = types.SimpleNamespace(task_id=22)
        deploy._deploy()
        self.api.app.wait.assert_not_called()
        deploy._on_deploy_callback.assert_called_once_with(22)
        self.api.app.start.assert_called_once_with(
            agent_id=3, module_id=1, workspace_id=8, is_branch=False,
            params={}, task_name="run-from-auto-track",
        )


if __name__ == "__main__":
    unittest.main()
