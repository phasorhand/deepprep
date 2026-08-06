"""Prompts for tree-based agentic inference (paper Sec 4.2).

The system prompt states the ADP problem (Sec 2.1), lists the operator space
``O`` (Sec 2.2), and specifies the ``<plan>/<expand>/<execute>/<answer>``
interaction protocol together with the prefix-matching constraint on node
references.

Sec 5.1 Stage 2 uses "elaborate prompts that exemplify the tree-based reasoning
process defined in Eq. (2)" to elicit teacher trajectories via in-context
learning; :data:`TREE_REASONING_EXEMPLAR` is that exemplar, and it deliberately
walks the Figure-4 story — expand, hit a runtime error, **backtrack to an earlier
node**, expand an alternative branch, answer — because backtracking is the
behaviour a linear ReAct trajectory would never demonstrate.
"""

from __future__ import annotations

from ..operators import operator_manual

__all__ = [
    "SYSTEM_PROMPT",
    "TREE_REASONING_EXEMPLAR",
    "build_system_prompt",
    "build_task_prompt",
    "build_turn_prompt",
]


_PROTOCOL = """
# Interaction protocol

You build the pipeline by interacting with an execution environment over an
*agentic reasoning tree*. Each node of the tree is a materialized environment
state (a set of intermediate tables). Each edge is one executed operator. The
root node `n0` holds the source tables.

At every turn you emit exactly two blocks: a `<plan>` and then either an
`<expand>` or an `<answer>`.

<plan>
Observe the current tree and the latest execution feedback. State (a) what the
current state still lacks with respect to the target schema, (b) which node you
will extend and why, and (c) which operators you intend to apply. If a previous
expansion failed, say what caused it and which earlier node must be revised.
</plan>

<expand>
<from>
The operator sequence from the root n0 down to the parent node you are extending,
one operator per line, copied EXACTLY as it appears in the tree.
Leave this block empty to extend the root n0.
</from>
<ops>
The new operators to apply, one per line. They execute sequentially; each one
creates a new child node.
</ops>
</expand>

<answer>
The complete operator sequence from the root n0 to the leaf whose state satisfies
the target schema, one operator per line, ending with Terminate([<table_name>])
naming the table that is the final result.
</answer>

The environment replies with an `<execute>` block containing the operators it
applied, the resulting tables, and any error trace. You never write `<execute>`
yourself.

# Rules

1. **Backtrack instead of restarting.** When feedback shows an earlier decision
   was wrong, point `<from>` at the last *correct* node and expand a corrected
   branch. The valid prefix is reused; you do not repeat the whole pipeline.
2. **Address nodes by prefix.** `<from>` must be the exact operator sequence from
   the root, so the parent is unambiguous.
3. **Ground every decision in the observed data**, not in assumptions about it.
   Check the sample rows before choosing keys, separators, or formats.
4. **Answer only when a leaf state actually satisfies the target schema** — the
   right column names, one row per target entity, and correctly typed values.
5. Prefer the predefined operators. Use `ExeCode` only for transformations none
   of them can express.
""".strip()


_OUTPUT_FORMAT_REMINDER = """
Reply with `<plan>` followed by `<expand>` (or `<answer>`). Emit nothing outside
those tags.
""".strip()


#: In-context exemplar of a full tree-based trajectory, including a backtrack.
#: Mirrors Figure 4 / Example 4 of the paper.
TREE_REASONING_EXEMPLAR = """
# Example trajectory (abridged)

Source tables: `movies(id, title, dir_id, genres)`, `directors(id, name)`,
`ratings(movie, values, rating)`. Target schema: one row per (director, single
genre), with the average rating in 2019 and in 2020.

--- Turn 1 ---
<plan>
The tree only has n0. `movies` has duplicate ids and inconsistently cased titles,
and `ratings.movie` is lowercase, so titles will not join until they are
normalized. I will clean `movies` from the root and narrow `ratings` to the
columns I expect to need.
</plan>
<expand>
<from>
</from>
<ops>
Deduplicate(movies, [id], first)
ValueTransform(movies, title, lambda x: str(x).strip().lower())
SelectColumn(ratings, [movie, rating])
</ops>
</expand>

<execute>
[1/3] OK  Deduplicate(movies, [id], first)
[2/3] OK  ValueTransform(movies, title, lambda x: str(x).strip().lower())
[3/3] OK  SelectColumn(ratings, [movie, rating])
Current state: movies(5x4), directors(3x2), ratings(11x2)
</execute>

--- Turn 2 ---
<plan>
n3 has clean tables. The target needs per-year columns, so I now integrate the
rating values into `movies` with ExeCode.
</plan>
<expand>
<from>
Deduplicate(movies, [id], first)
ValueTransform(movies, title, lambda x: str(x).strip().lower())
SelectColumn(ratings, [movie, rating])
</from>
<ops>
ExeCode([movies, ratings], movies_with_ratings, "movies_with_ratings = movies.merge(ratings, left_on='title', right_on='movie'); movies_with_ratings['year'] = movies_with_ratings['values']")
</ops>
</expand>

<execute>
[1] FAILED  ExeCode([movies, ratings], movies_with_ratings, ...)
ERROR: ExeCode: KeyError: 'values'
</execute>

--- Turn 3 (backtracking) ---
<plan>
The failure is not local to this operator: the year lives in `ratings.values`,
and my `SelectColumn` at n3 dropped that column. So the wrong decision was made
one step earlier. I backtrack to n2 — the state *before* `SelectColumn` — and
expand an alternative branch that keeps `values`, splits the rating and the year
out of it, filters to the two target years, and pivots to wide format.
</plan>
<expand>
<from>
Deduplicate(movies, [id], first)
ValueTransform(movies, title, lambda x: str(x).strip().lower())
</from>
<ops>
SelectColumn(ratings, [movie, values])
SplitColumn(ratings, values, [rating_value, year], sep=" (")
ValueTransform(ratings, year, lambda x: str(x).rstrip(")"))
CastType(ratings, rating_value, float)
Filter(ratings, lambda r: r["year"] in ("2019", "2020"))
</ops>
</expand>
""".strip()


def build_system_prompt(
    include_exemplar: bool = False,
    operator_categories: list[str] | None = None,
    max_turns: int = 5,
) -> str:
    """Assemble the system prompt: task definition + operator space + protocol."""
    parts = [
        "You are DeepPrep, an autonomous data preparation agent.",
        "",
        "# Task",
        "You are given a set of source tables and a target schema Sigma* describing the "
        "table an analyst needs. Construct and execute a data preparation pipeline -- an "
        "ordered sequence of operators -- that transforms the source tables into a table "
        "conforming to the target schema. The target table's contents are NOT given to "
        "you; you must derive them from the source data.",
        "",
        "# Operator space",
        "Every operator maps a set of tables to a set of tables. Tables not mentioned by "
        "an operator are carried through unchanged. Reference tables and columns by bare "
        "name (e.g. `Deduplicate(movies, [id], first)`).",
        operator_manual(operator_categories),
        "",
        _PROTOCOL,
        "",
        f"You have at most {max_turns} expansion turns. Use them to converge, not to "
        f"explore aimlessly.",
    ]
    if include_exemplar:
        parts += ["", TREE_REASONING_EXEMPLAR]
    return "\n".join(parts)


def build_task_prompt(task_input: str) -> str:
    """The user message carrying ``phi(S)`` and ``Sigma*`` (Eq. 5)."""
    return (
        f"{task_input}\n\n"
        "Build the pipeline that produces a table conforming to the target schema.\n\n"
        f"{_OUTPUT_FORMAT_REMINDER}"
    )


def build_turn_prompt(tree_view: str, feedback: str | None, turn: int, max_turns: int) -> str:
    """The observation for step ``t``: the tree ``G_{t-1}`` plus the last feedback."""
    parts = [tree_view]
    if feedback:
        parts += ["", "<execute>", feedback, "</execute>"]
    remaining = max_turns - turn
    if remaining <= 1:
        parts += [
            "",
            "This is your LAST expansion turn. If a leaf state already satisfies the "
            "target schema, emit <answer> now; otherwise make this expansion the one "
            "that completes the pipeline, then answer.",
        ]
    else:
        parts += ["", f"Turn {turn + 1} of {max_turns}."]
    parts += ["", _OUTPUT_FORMAT_REMINDER]
    return "\n".join(parts)


#: Default system prompt (no exemplar) — used by trained checkpoints, which have
#: internalized the protocol and no longer need the in-context demonstration.
SYSTEM_PROMPT = build_system_prompt()
