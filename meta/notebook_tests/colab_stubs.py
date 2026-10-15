"""IPython startup file for the test kernel: replaces Colab features that need a human.

Copied into the Colab image's IPython startup folder, so it runs before the
first cell of every tested notebook.
"""

import os

from IPython import get_ipython
from google.colab import userdata


def _get_secret(key: str) -> str:
    # The Colab secrets sidebar, filled from the Modal secret instead.
    try:
        return os.environ[key]
    except KeyError:
        raise userdata.SecretNotFoundError(key) from None


def _press_enter(prompt: str = "") -> str:
    # The human just presses Enter: accept whatever is asked.
    print(prompt + "[test runner: pressed Enter]")
    return ""


userdata.get = _get_secret
# ipykernel binds builtins.input to kernel.raw_input before each cell.
get_ipython().kernel.raw_input = _press_enter
