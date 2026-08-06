"""Sec 2.2.8 — Program Synthesis (1), plus the ``Terminate`` control operator.

    "ExeCode(tables, target, func) generates a target table from input tables by
     executing Python code synthesized by an LLM, enabling transformations that
     cannot be expressed using the above predefined operators."

``Terminate`` appears in Figure 4 as the last element of the extracted pipeline
(``... Terminate([result_table])``).  It carries no data transformation: it marks
which table of the final state is the answer ``T_hat``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..types import Table, TableSet
from .base import Operator, OperatorError, ParamSpec
from .util import as_list, require_table

__all__ = ["ExeCode", "Terminate"]

#: Attribute key used to mark the answer table on a TableSet.
ANSWER_ATTR = "_deepprep_answer"


class ExeCode(Operator):
    NAME = "ExeCode"
    CATEGORY = "program"
    DOC = (
        "Generate a target table from input tables by executing synthesized Python code. "
        "Input tables are bound as pandas DataFrames under their table names; the code "
        "must assign the result to a variable named exactly like 'target'."
    )
    PARAMS = (
        ParamSpec("tables", kind="tables", doc="names of the input tables to expose to the code"),
        ParamSpec("target", kind="table", doc="name of the produced table AND of the result variable"),
        ParamSpec("func", kind="code",
                  doc="Python source; pandas is available as 'pd' and numpy as 'np'"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        from .sandbox import SandboxError, exec_code

        names = [str(n) for n in as_list(p["tables"])]
        inputs = {n: require_table(tables, n).df.copy(deep=True) for n in names}
        target = str(p["target"])
        code = p["func"]
        if callable(code):
            raise OperatorError(
                "ExeCode: 'func' must be Python source code (a string or a ``` fenced block ```), "
                "not a lambda. Use ValueTransform/AddNewColumn for lambda-shaped transformations."
            )
        if not isinstance(code, str) or not code.strip():
            raise OperatorError("ExeCode: 'func' must be a non-empty Python source string.")

        try:
            result_ns = exec_code(code, inputs)
        except SandboxError as e:
            raise OperatorError(f"ExeCode: {e}") from e

        if target not in result_ns:
            assigned = [
                k for k, v in result_ns.items()
                if isinstance(v, pd.DataFrame) and k not in inputs
            ]
            raise OperatorError(
                f"ExeCode: the code did not assign a variable named '{target}'. "
                f"DataFrames it did assign: {assigned or 'none'}. "
                f"End the code with `{target} = <your dataframe>`."
            )

        out = result_ns[target]
        if isinstance(out, pd.Series):
            out = out.to_frame()
        if not isinstance(out, pd.DataFrame):
            raise OperatorError(
                f"ExeCode: variable '{target}' is a {type(out).__name__}, expected a "
                f"pandas DataFrame."
            )
        out = out.reset_index(drop=True)
        out.columns = [str(c) for c in out.columns]
        return tables.replace(Table(name=target, df=out))


class Terminate(Operator):
    NAME = "Terminate"
    CATEGORY = "control"
    DOC = (
        "Mark a table of the current state as the final target table T_hat. "
        "This is the last operator of an extracted pipeline; it performs no transformation."
    )
    PARAMS = (
        ParamSpec("table", kind="tables", doc="name of the table holding the final result"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        names = [str(n) for n in as_list(p["table"])]
        if len(names) != 1:
            raise OperatorError(
                f"Terminate: expected exactly one result table, got {names}. "
                f"The pipeline must converge to a single target table."
            )
        t = require_table(tables, names[0])
        out = tables.copy()
        setattr(out, ANSWER_ATTR, t.name)
        return out


def answer_table_name(tables: TableSet) -> str | None:
    """Return the table marked by ``Terminate``, if any."""
    return getattr(tables, ANSWER_ATTR, None)
