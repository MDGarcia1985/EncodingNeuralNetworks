# Purpose: Start the Tkinter Titanic survival app from a script entry point.
# Design: Add the project root to sys.path so direct script execution can import src.
# Workflow: Resolve the project root, import the app root, and enter the Tkinter main loop.
#
# Michael Garcia
# michael@mandedesign.studio
# https://mandedesign.studio
#
# License: MIT

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.app import root


if __name__ == "__main__":
    root.mainloop()
