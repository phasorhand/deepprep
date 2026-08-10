"""The sandbox's module whitelist must actually work (Sec 2.2.8, ExeCode).

``_audit`` deliberately permits ``import pandas`` and friends, but the namespace
handed to ``exec`` had no ``__import__``, so every permitted import died at run
time with ``ImportError: __import__ not found``.  The whitelist was unreachable
code.  It zeroed the paper's CodeGen baseline outright -- LLM-written pandas
programs open with ``import pandas as pd``, so *every* generated program failed
before its first statement and the baseline scored 0.00 completion for a reason
that had nothing to do with the model.
"""

from __future__ import annotations

import pandas as pd
import pytest

from deepprep.operators.sandbox import SandboxError, compile_callable, exec_code


def _ns() -> dict:
    return {"movies": pd.DataFrame({"id": [1, 2], "score": [3.0, 4.0]})}


# --------------------------------------------------------------------------- #
# permitted imports now run
# --------------------------------------------------------------------------- #
def test_a_whitelisted_import_executes():
    out = exec_code("import pandas as pd\nresult = pd.DataFrame({'a': [1, 2]})", _ns())
    assert list(out["result"]["a"]) == [1, 2]


def test_from_import_of_a_whitelisted_module_executes():
    out = exec_code("from numpy import nan\nresult = nan", _ns())
    assert out["result"] != out["result"]  # NaN


def test_the_codegen_baseline_shape_of_program_runs():
    """What an LLM actually emits for these tasks."""
    code = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "df = movies.drop_duplicates(subset=['id'])\n"
        "out = df[df['score'] > 3.0]\n"
    )
    out = exec_code(code, _ns())
    assert list(out["out"]["id"]) == [2]


@pytest.mark.parametrize("module", ["pandas", "numpy", "re", "math", "datetime", "json",
                                    "statistics"])
def test_every_advertised_module_is_importable(module):
    """The audit's error message names these; each must really be reachable."""
    out = exec_code(f"import {module}\nresult = {module}", _ns())
    assert out["result"] is not None


# --------------------------------------------------------------------------- #
# the restriction still holds
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("code", [
    "import os\nresult = os.getcwd()",
    "import subprocess",
    "from os import path",
    "import sys",
])
def test_a_non_whitelisted_import_is_still_rejected(code):
    with pytest.raises(SandboxError):
        exec_code(code, _ns())


def test_dunder_import_is_still_unreachable():
    with pytest.raises(SandboxError):
        exec_code("result = __import__('os')", _ns())


def test_importlib_cannot_be_reached_through_a_permitted_module():
    """A whitelisted module must not become a bridge to the rest of the runtime."""
    with pytest.raises(SandboxError):
        exec_code("import json\nresult = json.__loader__", _ns())


def test_relative_imports_are_rejected():
    with pytest.raises(SandboxError):
        exec_code("from . import os", _ns())


def test_lambda_parameters_are_unaffected():
    fn = compile_callable("lambda x: str(x).strip().lower()")
    assert fn("  AB ") == "ab"
