"""Workspace round-trip checks; runnable with unittest without extra dependencies."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

import workspace_config as config

ROOT = Path(__file__).resolve().parents[1]


def widget(app, kind, label):
    return next(w for w in getattr(app, kind) if w.label == label)


def fresh_app(saved=None):
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    if saved:
        app.file_uploader(key='workspace_upload').set_value(
            ('workspace.json', saved, 'application/json')).run()
        widget(app, 'button', 'Load config').click().run()
    return app


class ConfigFormatTests(unittest.TestCase):
    def test_roundtrip_filters_transient_state_and_preserves_types(self):
        state = {'data_folder': '/tmp/scans', 'probe_positions': [(2, 3)],
                 'curve_styles': {'curve': {'color': '#123abc', 'line_style': 'Dashed', 'width': 3.5}},
                 'angle_saved_rois': {'scan': (1, 4, 2, 8)},
                 'spectrum_sigma': 4.0, 'normalize_probe': False,
                 'workspace_upload': io.BytesIO(b'not serialized'),
                 'detector_click': {'clicked': True}, 'bg_apply': True}
        encoded = config.encode_config(state)
        decoded = config.decode_config(encoded)
        self.assertEqual(decoded['settings']['probe_positions'], [(2, 3)])
        self.assertEqual(decoded['settings']['angle_saved_rois']['scan'], (1, 4, 2, 8))
        self.assertEqual(decoded['settings']['curve_styles'], state['curve_styles'])
        self.assertNotIn('workspace_upload', decoded['settings'])
        self.assertNotIn('bg_apply', decoded['settings'])
        self.assertNotIn('detector_click', decoded['settings'])

    def test_invalid_config_does_not_change_existing_workspace(self):
        document = json.loads(config.encode_config({'data_folder': '/good', 'spectrum_stride': 3}))
        invalid = [b'not JSON', b'[]']
        for field, value in [('spectrum_stride', 0), ('spectrum_stride', True),
                             ('intensity_display', 'bogus'), ('spectrum_sigma', float('nan')),
                             ('line_color:curve', 'red'), ('bg_apply', True)]:
            changed = dict(document, settings={**document['settings'], field: value})
            invalid.append(json.dumps(changed).encode())
        invalid.append(json.dumps(dict(document, version=999)).encode())
        for data in invalid:
            with self.subTest(data=data):
                state = {'data_folder': '/original', 'spectrum_stride': 5,
                         'workspace_upload': io.BytesIO(data)}
                with patch.object(config.st, 'session_state', state):
                    config._load_upload()
                self.assertEqual(state['data_folder'], '/original')
                self.assertEqual(state['spectrum_stride'], 5)
                self.assertEqual(state['workspace_message'][0], 'error')

    def test_load_callback_replaces_stale_settings(self):
        saved = config.encode_config({'data_folder': '/restored', 'spectrum_stride': 7})
        state = {'data_folder': '/old', 'line_color:old': '#ffffff',
                 'background_state': object(), 'folder_browser_open': True,
                 'workspace_upload': io.BytesIO(saved)}
        with patch.object(config.st, 'session_state', state):
            config._load_upload()
        self.assertEqual(state['data_folder'], '/restored')
        self.assertEqual(state['spectrum_stride'], 7)
        self.assertNotIn('line_color:old', state)
        self.assertNotIn('background_state', state)
        self.assertNotIn('folder_browser_open', state)
        self.assertEqual(state['workspace_message'][0], 'success')


class WorkspaceAppTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='eels-workspace-test-')
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        # Both input formats, unequal energy lengths and multiple probes.
        np.save(self.folder / 'a.npy', np.arange(1., 316.).reshape(35, 3, 3))
        np.save(self.folder / 'b.npy', np.ones((1, 41, 2, 2, 3, 3)))

    def open_scans(self):
        app = fresh_app()
        widget(app, 'text_input', 'Data folder').set_value(str(self.folder)).run()
        self.assertHealthy(app)
        return app

    def assertHealthy(self, app):
        self.assertFalse(app.exception, str(app.exception))
        self.assertFalse(app.error, str(app.error))

    def test_restore_selected_scan_parameters_labels_and_line_style(self):
        app = self.open_scans()
        widget(app, 'multiselect', 'Files to compare').set_value([str(self.folder / 'b.npy')]).run()
        widget(app, 'text_input', 'b.npy').set_value('Custom spectrum').run()
        app.multiselect(key='probe_positions').set_value([(1, 1)]).run()
        app.checkbox(key='broaden_spectrum').check().run()
        app.number_input(key='detector_radius').set_value(1.5)
        app.number_input(key='offset_px').set_value(1)
        app.number_input(key='spectrum_timestep').set_value(4.0)
        app.number_input(key='spectrum_stride').set_value(2)
        app.number_input(key='spectrum_sigma').set_value(3.0)
        app.selectbox(key='intensity_display').select('Intensity × E²')
        app.number_input(key='spectrum_x_min').set_value(-80.0)
        app.number_input(key='spectrum_x_max').set_value(100.0)
        widget(app, 'color_picker', 'Line color').set_value('#aabbcc')
        widget(app, 'selectbox', 'Line style').select('Dashed')
        widget(app, 'number_input', 'Line width').set_value(4.0)
        app.selectbox(key='angle_direction').select('Vertical (px)')
        app.selectbox(key='angle_color_scale').select('Linear')
        app.number_input(key='angle_energy_min').set_value(-90.)
        widget(app, 'number_input', 'Row min (px)').set_value(1)
        app.run()
        self.assertHealthy(app)
        saved = config.encode_config(dict(app.session_state))
        restored = fresh_app(saved)
        self.assertHealthy(restored)
        self.assertEqual(widget(restored, 'text_input', 'Data folder').value, str(self.folder))
        self.assertEqual(widget(restored, 'multiselect', 'Files to compare').value, [str(self.folder / 'b.npy')])
        self.assertEqual(widget(restored, 'text_input', 'b.npy').value, 'Custom spectrum')
        self.assertEqual(restored.multiselect(key='probe_positions').value, [(1, 1)])
        for key, value in [('detector_radius', 1.5), ('offset_px', 1), ('spectrum_timestep', 4.),
                           ('spectrum_stride', 2), ('spectrum_sigma', 3.), ('spectrum_x_min', -80.),
                           ('spectrum_x_max', 100.)]:
            self.assertEqual(restored.number_input(key=key).value, value)
        self.assertTrue(restored.checkbox(key='broaden_spectrum').value)
        self.assertEqual(restored.selectbox(key='intensity_display').value, 'Intensity × E²')
        self.assertEqual(widget(restored, 'color_picker', 'Line color').value, '#aabbcc')
        self.assertEqual(widget(restored, 'selectbox', 'Line style').value, 'Dashed')
        self.assertEqual(widget(restored, 'number_input', 'Line width').value, 4.)
        self.assertEqual(restored.selectbox(key='angle_direction').value, 'Vertical (px)')
        self.assertEqual(restored.selectbox(key='angle_color_scale').value, 'Linear')
        self.assertEqual(restored.number_input(key='angle_energy_min').value, -90.)
        self.assertEqual(widget(restored, 'number_input', 'Row min (px)').value, 1)
        self.assertTrue(any(w.label == 'Save config' for w in restored.get('download_button')))

    def test_view_switch_keeps_spectrum_and_map_settings(self):
        app = self.open_scans()
        app.number_input(key='spectrum_timestep').set_value(4.5).run()
        widget(app, 'color_picker', 'Line color').set_value('#abcdef').run()
        app.radio(key='visualization').set_value('2D scan map').run()
        app.number_input(key='map_timestep').set_value(6.)
        app.number_input(key='map_columns').set_value(2)
        app.text_input(key='map_energies').set_value('0, 10')
        app.run()
        self.assertHealthy(app)
        restored = fresh_app(config.encode_config(dict(app.session_state)))
        self.assertHealthy(restored)
        self.assertEqual(restored.radio(key='visualization').value, '2D scan map')
        self.assertEqual(restored.number_input(key='map_timestep').value, 6.)
        self.assertEqual(restored.number_input(key='map_columns').value, 2)
        self.assertEqual(restored.text_input(key='map_energies').value, '0, 10')
        restored.radio(key='visualization').set_value('Spectra & detector').run()
        self.assertHealthy(restored)
        self.assertEqual(restored.number_input(key='spectrum_timestep').value, 4.5)
        self.assertEqual(widget(restored, 'color_picker', 'Line color').value, '#abcdef')

    def test_missing_scans_and_folder_leave_config_controls_available(self):
        app = self.open_scans()
        saved = config.encode_config(dict(app.session_state))
        (self.folder / 'a.npy').unlink()
        restored = fresh_app(saved)
        self.assertHealthy(restored)
        self.assertTrue(any('Saved scans unavailable' in w.value for w in restored.warning))
        self.assertEqual(widget(restored, 'multiselect', 'Files to compare').value, [str(self.folder / 'b.npy')])
        bad_folder = config.encode_config({'data_folder': str(self.folder / 'missing')})
        restored = fresh_app(bad_folder)
        self.assertFalse(restored.exception)
        self.assertTrue(restored.error)
        self.assertTrue(any(w.label == 'Load config' for w in restored.button))
        self.assertTrue(any(w.label == 'Save config' for w in restored.get('download_button')))

    def test_load_replaces_existing_style_and_handles_invalid_upload(self):
        app = self.open_scans()
        widget(app, 'color_picker', 'Line color').set_value('#112233').run()
        saved = config.encode_config(dict(app.session_state))
        widget(app, 'color_picker', 'Line color').set_value('#ffffff').run()
        app.file_uploader(key='workspace_upload').set_value(
            ('workspace.json', saved, 'application/json')).run()
        widget(app, 'button', 'Load config').click().run()
        self.assertHealthy(app)
        self.assertEqual(widget(app, 'color_picker', 'Line color').value, '#112233')
        app.file_uploader(key='workspace_upload').set_value(
            ('broken.json', b'{', 'application/json')).run()
        widget(app, 'button', 'Load config').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Could not load config' in w.value for w in app.error))
        self.assertEqual(widget(app, 'color_picker', 'Line color').value, '#112233')

    def test_restore_zarr_directory(self):
        import zarr
        store = self.folder / 'scan.zarr'
        array = zarr.open_array(str(store), mode='w', shape=(21, 3, 3),
                                chunks=(7, 3, 3), dtype='f8')
        array[:] = 1.
        app = fresh_app()
        widget(app, 'text_input', 'Data folder').set_value(str(store)).run()
        widget(app, 'color_picker', 'Line color').set_value('#abcdef').run()
        restored = fresh_app(config.encode_config(dict(app.session_state)))
        self.assertHealthy(restored)
        self.assertEqual(widget(restored, 'multiselect', 'Files to compare').value, [str(store)])
        self.assertEqual(widget(restored, 'color_picker', 'Line color').value, '#abcdef')

    def test_background_fit_is_recomputed_on_fresh_session(self):
        app = self.open_scans()
        app.selectbox(key='bg_method').select('SNIP').run()
        app.selectbox(key='bg_domain').select('Full recorded spectrum').run()
        app.button(key='bg_apply').click().run()
        self.assertHealthy(app)
        before = app.session_state['background_state']
        self.assertTrue(before.applied_results)
        expected = {key: result.corrected.copy() for key, result in before.applied_results.items()}
        restored = fresh_app(config.encode_config(dict(app.session_state)))
        self.assertHealthy(restored)
        after = restored.session_state['background_state']
        self.assertEqual(before.applied_config, after.applied_config)
        self.assertEqual(restored.radio(key='bg_signal').value, 'Corrected')
        for key in expected:
            np.testing.assert_allclose(after.applied_results[key].corrected, expected[key], equal_nan=True)


if __name__ == '__main__':
    unittest.main()
