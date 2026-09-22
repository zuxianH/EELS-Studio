"""Paths pasted from a terminal or formatted text should resolve predictably."""
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np
import pytest

from scan_sources import resolve_data_directory

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('style', ['plain', 'whitespace', 'single_quote', 'double_quote',
                                  'backticks', 'escaped', 'quoted_escaped'])
def test_pasted_path_formats(tmp_path, style):
    folder = tmp_path / 'scan_data (300 K).zarray'
    folder.mkdir()
    raw = str(folder)
    escaped = raw.replace('_', r'\_').replace(' ', r'\ ').replace('.', r'\.')
    value = {'plain': raw, 'whitespace': '\n ' + raw + '\t',
             'single_quote': "'" + raw + "'", 'double_quote': '"' + raw + '"',
             'backticks': '`' + raw + '`', 'escaped': escaped,
             'quoted_escaped': ' "' + escaped + '"\n'}[style]
    assert resolve_data_directory(value) == folder.resolve()


def test_existing_literal_backslashes_and_spaces_are_preserved(tmp_path):
    for name in [r'scan\_data', ' trailing space ']:
        folder = tmp_path / name
        folder.mkdir()
        assert resolve_data_directory(str(folder)) == folder


def test_home_paths():
    assert resolve_data_directory(' "~" ') == Path.home().resolve()


def test_missing_file_and_permission_errors_are_specific(tmp_path):
    missing = tmp_path / 'missing'
    with pytest.raises(ValueError) as exc:
        resolve_data_directory(str(missing))
    assert str(missing) in str(exc.value)
    assert socket.gethostname() in str(exc.value)
    file = tmp_path / 'scan.npy'
    file.touch()
    with pytest.raises(ValueError, match='got a file'):
        resolve_data_directory(str(file))
    with patch.object(Path, 'stat', side_effect=PermissionError):
        with pytest.raises(ValueError, match='Permission denied'):
            resolve_data_directory(str(tmp_path))
    with pytest.raises(ValueError, match='Folder not found'):
        resolve_data_directory(' \n ')


def test_app_accepts_escaped_quoted_path(tmp_path):
    from streamlit.testing.v1 import AppTest
    folder = tmp_path / 'scan_data'
    folder.mkdir()
    np.save(folder / 'scan.npy', np.ones((9, 5, 6)))
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=45).run()
    pasted = ' "' + str(folder).replace('_', r'\_') + '" '
    app.text_input(key='data_folder').set_value(pasted).run()
    assert not app.exception and not app.error
    assert app.metric[0].value == '1'
    app.text_input(key='data_folder').set_value(str(folder / 'missing')).run()
    assert not app.exception
    assert any('Folder not found on' in error.value for error in app.error)
