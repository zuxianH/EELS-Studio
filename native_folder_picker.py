"""Open a desktop folder chooser outside Streamlit's worker thread."""
from pathlib import Path
import os
import shutil
import subprocess
import sys


# Tk must own its main thread; a child process also isolates its event loop from
# Streamlit reruns. On Windows/macOS, askdirectory uses the native folder dialog.
_TK_PICKER = """
import sys
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
try:
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(
        parent=root, title="Choose your EELS data folder",
        initialdir=sys.argv[1], mustexist=True)
    sys.stdout.write(folder or "")
finally:
    root.destroy()
"""


def choose_directory(initial):
    """Return the chosen Path, None on cancel, or raise if no dialog can open."""
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("No desktop display is available.")
    title = "Choose your EELS data folder"
    if sys.platform.startswith("linux") and (program := shutil.which("zenity")):
        command = [program, "--file-selection", "--directory", f"--title={title}",
                   f"--filename={str(initial).rstrip(os.sep) + os.sep}"]
        desktop_tool = True
    elif sys.platform.startswith("linux") and (program := shutil.which("kdialog")):
        command = [program, "--getexistingdirectory", str(initial), "--title", title]
        desktop_tool = True
    else:
        command = [sys.executable, "-c", _TK_PICKER, str(initial)]
        desktop_tool = False
    result = subprocess.run(command, capture_output=True, text=True)
    if "cannot open display" in result.stderr.lower():
        raise RuntimeError("The desktop display is unavailable.")
    if desktop_tool and result.returncode == 1:
        return None  # The user closed or cancelled the desktop dialog.
    if result.returncode != 0:
        raise RuntimeError("The system folder-selection window could not open.")
    selected = result.stdout.rstrip("\r\n") if desktop_tool else result.stdout
    return Path(selected) if selected else None
