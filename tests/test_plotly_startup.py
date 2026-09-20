"""Cold-process coverage for Plotly inspecting a concurrently importing pandas."""
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def test_startup_waits_for_pandas_before_plotly_validation():
    # A fresh interpreter avoids pandas already being initialized by another test.
    script = textwrap.dedent('''
        import ast
        import importlib.abc
        import importlib.machinery
        from pathlib import Path
        import sys
        import threading

        tree = ast.parse(Path(sys.argv[1]).read_text())
        startup = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == 'eels_core':
                break
            startup.append(node)
        bootstrap = ast.Module(body=startup, type_ignores=[])
        # Warm other imports so their disk I/O cannot hide the race. Leave pandas
        # unimported, exactly as Plotly's optional dependency lookup permits.
        warm = ast.Module(body=[node for node in startup if not (
            isinstance(node, ast.Import) and any(a.name == 'pandas' for a in node.names)
        )], type_ignores=[])
        namespace = {}
        exec(compile(warm, sys.argv[1], 'exec'), namespace)
        namespace['go'].Scatter(x=[0., 1.], y=[1., 2.], hovertemplate='%{y:.6g}')
        assert 'pandas' not in sys.modules

        started, release = threading.Event(), threading.Event()
        errors = []
        class Loader(importlib.abc.Loader):
            def __init__(self, original): self.original = original
            def create_module(self, spec): return self.original.create_module(spec)
            def exec_module(self, module):
                started.set()
                assert release.wait(20), 'pandas import was never released'
                self.original.exec_module(module)
        class Finder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == 'pandas':
                    spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                    spec.loader = Loader(spec.loader)
                    return spec
        def load_pandas():
            try:
                __import__('pandas')
            except BaseException as exc:
                errors.append(exc)
        sys.meta_path.insert(0, Finder())
        thread = threading.Thread(target=load_pandas)
        thread.start()
        assert started.wait(20)
        timer = threading.Timer(1, release.set)
        timer.start()
        try:
            # The real application bootstrap must wait for pandas, otherwise
            # Scatter's string validation sees a module with no Series/Index.
            exec(compile(bootstrap, sys.argv[1], 'exec'), namespace)
            import plotly.graph_objects as go
            trace = go.Scatter(x=[0., 1.], y=[1., 2.],
                              hovertemplate='%{y:.6g}<extra>%{fullData.name}</extra>')
            assert trace.hovertemplate.startswith('%{y:')
            assert not errors
        finally:
            release.set()
            timer.cancel()
            thread.join(30)
    ''')
    subprocess.run([sys.executable, '-c', script, str(ROOT / 'app.py')],
                   check=True, capture_output=True, text=True, timeout=90)
