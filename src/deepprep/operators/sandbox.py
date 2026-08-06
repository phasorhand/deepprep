"""Restricted execution for user-defined function / code parameters.

Several operators take a *user-defined function* (``ValueTransform``, ``Filter``,
``AddNewColumn``, ``ErrorDetection``, ``SplitColumn``, ``Concatenate``,
``CalculateStatistic``) and ``ExeCode`` takes an arbitrary Python program — the
paper's "Program Synthesis" operator (Sec 2.2.8) that exists precisely to handle
"transformations that cannot be expressed using the above predefined operators".

.. warning::
   Executing model-generated code is inherently dangerous.  The namespace
   restriction below blocks the obvious footguns (``import os``, ``open``,
   ``eval``, ``__subclasses__`` traversal) but **it is not a security sandbox**
   and must not be treated as one.  For untrusted inputs run the environment
   inside a container or set ``DEEPPREP_ALLOW_EXEC=0`` to disable code
   parameters entirely.
"""

from __future__ import annotations

import ast
import builtins
import datetime as _datetime
import math
import os
import re
import statistics
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["SandboxError", "compile_callable", "exec_code", "safe_globals"]


class SandboxError(Exception):
    """Raised when generated code is rejected or fails to compile."""


# Names that are never reachable from generated code.
_FORBIDDEN_NAMES = {
    "__import__",
    "__builtins__",
    "__subclasses__",
    "__globals__",
    "__code__",
    "__class__",
    "__bases__",
    "__mro__",
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "memoryview",
    "exit",
    "quit",
}

_ALLOWED_MODULES = {"pandas", "pd", "numpy", "np", "re", "math", "datetime", "json", "statistics"}

_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
        "float", "format", "frozenset", "getattr", "hasattr", "hash", "int",
        "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min",
        "next", "print", "range", "repr", "reversed", "round", "set", "setattr",
        "slice", "sorted", "str", "sum", "tuple", "type", "zip",
        "ValueError", "TypeError", "KeyError", "IndexError", "AttributeError",
        "ZeroDivisionError", "Exception", "StopIteration", "True", "False", "None",
    )
    if hasattr(builtins, name)
}


def allow_exec() -> bool:
    """Whether code/lambda parameters may be executed at all."""
    return os.environ.get("DEEPPREP_ALLOW_EXEC", "1") not in ("0", "false", "False")


def safe_globals(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the restricted namespace shared by lambdas and ``ExeCode``."""
    g: dict[str, Any] = {
        "__builtins__": dict(_SAFE_BUILTINS),
        "pd": pd,
        "pandas": pd,
        "np": np,
        "numpy": np,
        "re": re,
        "math": math,
        "statistics": statistics,
        "datetime": _datetime,
        "nan": float("nan"),
        "NaN": float("nan"),
    }
    if extra:
        g.update(extra)
    return g


def _audit(source: str) -> None:
    """Reject syntactically dangerous constructs before compilation."""
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        # `mode="exec"` fails for bare expressions such as a lambda; retry.
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as e:
            raise SandboxError(f"SyntaxError in generated code: {e.msg} (line {e.lineno})") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = (
                [a.name.split(".")[0] for a in node.names]
                if isinstance(node, ast.Import)
                else [(node.module or "").split(".")[0]]
            )
            bad = [m for m in mods if m not in _ALLOWED_MODULES]
            if bad:
                raise SandboxError(
                    f"import of {bad[0]!r} is not allowed. Allowed modules: "
                    f"{sorted(_ALLOWED_MODULES)}"
                )
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_NAMES:
            raise SandboxError(f"attribute access to {node.attr!r} is not allowed")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise SandboxError(f"use of {node.id!r} is not allowed")


def compile_callable(source: str) -> Any:
    """Compile a function-valued parameter (typically a ``lambda``).

    Accepts a lambda expression, a ``def`` statement, or any expression that
    evaluates to a callable (e.g. ``str.strip``).
    """
    if not allow_exec():
        raise SandboxError(
            "code parameters are disabled (DEEPPREP_ALLOW_EXEC=0); "
            "use predefined operators instead"
        )
    src = source.strip()
    _audit(src)
    g = safe_globals()
    try:
        return eval(compile(ast.parse(src, mode="eval"), "<deepprep-param>", "eval"), g)  # noqa: S307
    except SyntaxError:
        pass
    except SandboxError:
        raise
    except Exception as e:  # noqa: BLE001
        raise SandboxError(f"failed to evaluate parameter: {type(e).__name__}: {e}") from e

    # Statement form: `def f(x): ...` — execute and return the defined function.
    local: dict[str, Any] = {}
    try:
        exec(compile(src, "<deepprep-param>", "exec"), g, local)  # noqa: S102
    except Exception as e:  # noqa: BLE001
        raise SandboxError(f"failed to compile parameter: {type(e).__name__}: {e}") from e
    funcs = [v for v in local.values() if callable(v)]
    if not funcs:
        raise SandboxError(f"parameter did not define a callable: {src[:120]}")
    return funcs[-1]


def exec_code(code: str, namespace: dict[str, Any]) -> dict[str, Any]:
    """Execute an ``ExeCode`` program against ``namespace``; return the new bindings.

    ``namespace`` maps table names to DataFrames.  Whatever the program assigns
    is returned so the operator can pick up its declared target table.
    """
    if not allow_exec():
        raise SandboxError(
            "ExeCode is disabled (DEEPPREP_ALLOW_EXEC=0); use predefined operators instead"
        )
    _audit(code)
    g = safe_globals()
    local: dict[str, Any] = dict(namespace)
    try:
        exec(compile(code, "<deepprep-execode>", "exec"), g, local)  # noqa: S102
    except SandboxError:
        raise
    except Exception as e:  # noqa: BLE001
        raise SandboxError(f"{type(e).__name__}: {e}") from e
    return local
