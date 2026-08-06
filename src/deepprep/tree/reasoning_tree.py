"""The agentic reasoning tree ``G = (N, E)`` (paper Sec 4.1).

    "Node (N). Each node n in N corresponds to an *environment state*,
     represented by the current set of intermediate tables maintained by the
     environment.  In particular, the root node n_0 represents the initial state
     initialized with the source tables S.

     Edge (E). A directed edge e = (n_i -o-> n_j) in E represents a state
     transition induced by executing an *operator instance* o ... orchestrated by
     the agent at state n_i."

    "In the agentic reasoning tree, each edge corresponds to a single operator
     instance, while a path from a node to another node represents an operator
     sequence, i.e., a data preparation pipeline."

This is what distinguishes DeepPrep from ReAct: because every intermediate state
stays materialized, "execution feedback [can] be precisely attributed to earlier
decision points, allowing the agent to backtrack and revise upstream choices when
downstream failures occur" (Sec 1).

Node addressing
---------------
Sec 4.2, ``<expand>``:

    "To ensure consistent node selection, the parent node is specified using a
     *prefix-matching constraint*: the agent must reference the full operator
     sequence from the root to that node.  This constraint avoids ambiguous or
     index-based node references and enforces alignment between the proposed
     expansion and the recorded transformation history."

:meth:`ReasoningTree.resolve_parent` implements that constraint, with a tolerant
matcher (whitespace/quote-insensitive, then operator-name-and-key-argument
fuzzy) so a cosmetically different but semantically identical reference still
lands on the right node instead of silently restarting from the root.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache

from ..operators import OperatorCall
from ..serialize import serialize_table_set
from ..types import TableSet

__all__ = ["Node", "NodeStatus", "ReasoningTree"]


class NodeStatus:
    OK = "ok"
    FAILED = "failed"


@dataclass
class Node:
    """One node of ``G``: a materialized environment state, or a failure marker."""

    id: str
    parent: Node | None = None
    #: The operator instance labelling the edge ``parent -> self``.
    op: OperatorCall | None = None
    #: Materialized intermediate tables.  ``None`` for failure nodes: Sec 4.2 says
    #: on a runtime exception "no new state node is created".
    state: TableSet | None = None
    status: str = NodeStatus.OK
    #: Error trace recorded on a failure node (Figure 4 annotates n_4 this way).
    error: str | None = None
    children: list[Node] = field(default_factory=list)
    #: Turn index t at which this node was created, for tree rendering.
    turn: int = 0

    # -- structure ---------------------------------------------------------- #
    @property
    def depth(self) -> int:
        return 0 if self.parent is None else self.parent.depth + 1

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def failed(self) -> bool:
        return self.status == NodeStatus.FAILED

    @property
    def edge_source(self) -> str:
        return self.op.to_source() if self.op is not None else ""

    def path(self) -> list[Node]:
        """Root-to-self node chain."""
        out: list[Node] = []
        n: Node | None = self
        while n is not None:
            out.append(n)
            n = n.parent
        return list(reversed(out))

    def path_ops(self) -> list[OperatorCall]:
        """The pipeline ``P`` from the root to this node (Sec 4.1)."""
        return [n.op for n in self.path() if n.op is not None]

    def pipeline_source(self, joiner: str = "\n") -> str:
        return joiner.join(op.to_source() for op in self.path_ops())

    def descendants(self) -> Iterator[Node]:
        for c in self.children:
            yield c
            yield from c.descendants()

    # -- display ------------------------------------------------------------ #
    def state_summary(self) -> str:
        if self.state is None:
            return "no state (execution failed)"
        return ", ".join(f"{t.name}({t.n_rows}x{len(t.columns)})" for t in self.state) or "empty"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Node {self.id} {self.status} {self.edge_source[:40]!r}>"


def _normalize(src: str) -> str:
    """Whitespace/quote-insensitive normal form; the fallback when parsing fails."""
    s = src.strip().rstrip(",;")
    s = s.replace('"', "'")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def _norm_value(v: object) -> str:
    """Stable textual form of a bound parameter value."""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_norm_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(f"{_norm_value(k)}:{_norm_value(x)}" for k, x in sorted(
            v.items(), key=lambda kv: str(kv[0])
        )) + "}"
    return repr(v)


@lru_cache(maxsize=8192)
def _canonical(src: str) -> str:
    """Canonical form of an operator call, used for prefix matching.

    Matching on raw text is too brittle: the same edge can be written
    ``Deduplicate(movies, [id], first)`` or
    ``Deduplicate(table='movies', subset=['id'], keep='first')``, and the agent
    has no reason to reproduce its earlier formatting byte for byte.  Since the
    paper makes node identity depend on the operator sequence, a cosmetic
    difference silently restarting the agent from the root would be a real loss
    of the backtracking capability.

    So the reference is parsed and re-serialized from its *bound parameters*,
    which collapses positional/keyword form, quoting and spacing.  Callable
    parameters keep their normalized source, since two different lambdas are two
    different operators.
    """
    try:
        from ..operators import parse_operator_call

        call = parse_operator_call(src)
    except Exception:  # noqa: BLE001 - unparseable references fall back to text
        return _normalize(src)

    parts: list[str] = []
    for spec in type(call.op).PARAMS:
        if spec.kind in ("func", "code") and spec.name in call.raw_params:
            value = _normalize(call.raw_params[spec.name])
        else:
            v = call.params.get(spec.name)
            value = "<callable>" if callable(v) else _norm_value(v)
        parts.append(f"{spec.name}={value}")
    return f"{call.name.lower()}({','.join(parts)})"


def _coarse_key(src: str) -> str:
    """Fuzzy key: operator name plus its first argument.

    Two references to the same expansion nearly always agree on these even when
    the model re-formats optional keyword arguments.
    """
    m = re.match(r"\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", src.strip(), flags=re.DOTALL)
    if not m:
        return _normalize(src)
    name, args = m.group(1), m.group(2)
    first = ""
    depth = 0
    for ch in args:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            break
        first += ch
    return f"{name.lower()}({_normalize(first)})"


class ReasoningTree:
    """The materialized search tree over data preparation pipelines."""

    def __init__(self, root_state: TableSet) -> None:
        self.root = Node(id="n0", state=root_state, status=NodeStatus.OK)
        self.nodes: dict[str, Node] = {"n0": self.root}
        self._counter = 0

    # -- growth ------------------------------------------------------------- #
    def _next_id(self) -> str:
        self._counter += 1
        return f"n{self._counter}"

    def add_state(
        self, parent: Node, op: OperatorCall, state: TableSet, turn: int = 0
    ) -> Node:
        """Record a successful transition ``parent -o-> child`` (Sec 4.2)."""
        node = Node(
            id=self._next_id(), parent=parent, op=op, state=state,
            status=NodeStatus.OK, turn=turn,
        )
        parent.children.append(node)
        self.nodes[node.id] = node
        return node

    def add_failure(
        self, parent: Node, op: OperatorCall | None, source: str, error: str, turn: int = 0
    ) -> Node:
        """Record a failed transition.

        The node carries *no* materialized state, matching Sec 4.2 ("no new state
        node is created"), but it is kept in the tree so the error trace stays
        attributed to the exact decision point — as Figure 4 shows for ``n_4``.
        """
        node = Node(
            id=self._next_id(), parent=parent, op=op, state=None,
            status=NodeStatus.FAILED, error=error, turn=turn,
        )
        if op is None:
            # Parsing failed, so there is no OperatorCall; keep the raw source.
            node.__dict__["_raw_source"] = source
        parent.children.append(node)
        self.nodes[node.id] = node
        return node

    # -- queries ------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[Node]:
        yield self.root
        yield from self.root.descendants()

    def get(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def leaves(self, ok_only: bool = True) -> list[Node]:
        return [n for n in self if n.is_leaf and (not ok_only or not n.failed)]

    def ok_nodes(self) -> list[Node]:
        return [n for n in self if not n.failed]

    def failures(self) -> list[Node]:
        return [n for n in self if n.failed]

    def max_depth(self) -> int:
        return max((n.depth for n in self), default=0)

    # -- node addressing (Sec 4.2 prefix-matching constraint) ---------------- #
    def resolve_parent(self, prefix: list[str] | str | None) -> tuple[Node, str | None]:
        """Resolve a parent reference to a node.

        ``prefix`` is the operator sequence from the root, as required by the
        paper's prefix-matching constraint.  Returns ``(node, warning)`` where
        ``warning`` is non-``None`` when the reference had to be matched loosely
        or could not be fully resolved — the caller surfaces it as feedback so
        the agent learns to address nodes precisely.
        """
        if prefix is None:
            return self.root, None
        if isinstance(prefix, str):
            from ..operators import split_calls

            text = prefix.strip()
            # A bare node id is accepted as a fallback: Figure 4's plans say
            # things like "rollback to n_2", so refusing it would waste turns.
            if re.fullmatch(r"n\d+", text):
                node = self.nodes.get(text)
                if node is None:
                    return self.root, (
                        f"UnknownNode: there is no node '{text}'. Existing nodes: "
                        f"{sorted(self.nodes, key=lambda s: int(s[1:]))}. Falling back to n0."
                    )
                if node.failed:
                    return node.parent or self.root, (
                        f"Node {text} is a failure node with no materialized state; "
                        f"expanding from its parent {(node.parent or self.root).id} instead."
                    )
                return node, None
            steps = split_calls(text) if text else []
        else:
            steps = [s for s in (str(x).strip() for x in prefix) if s]

        if not steps:
            return self.root, None

        node = self.root
        for i, step in enumerate(steps):
            child = self._match_child(node, step)
            if child is None:
                return node, (
                    f"PrefixMismatch: no recorded edge from {node.id} matches operator "
                    f"#{i + 1} of the referenced prefix: {step!r}. Available edges from "
                    f"{node.id}: {[c.edge_source for c in node.children] or 'none'}. "
                    f"Expanding from {node.id} (the longest matching prefix) instead."
                )
            if child.failed:
                return node, (
                    f"The referenced prefix ends in the failed transition {step!r}, which has "
                    f"no materialized state. Expanding from {node.id} instead."
                )
            node = child
        return node, None

    def resolve_longest_prefix(self, steps: list[str]) -> tuple[Node, list[str]]:
        """Walk ``steps`` as far as the tree allows.

        Returns the deepest matching node and the operators that still have to be
        executed.  This is how ``<answer>`` is realized without recomputation:
        Sec 4.2 extracts "the root-to-leaf operator path associated with that
        node", and any suffix the agent invented but never executed is run from
        the matched node rather than from the root.
        """
        node = self.root
        for i, step in enumerate(steps):
            child = self._match_child(node, step)
            if child is None or child.failed:
                return node, steps[i:]
            node = child
        return node, []

    @staticmethod
    def _match_child(node: Node, step: str) -> Node | None:
        """Match one operator of a prefix against the edges leaving ``node``."""
        target = _canonical(step)
        for c in node.children:
            if _canonical(c.edge_source) == target:
                return c
        coarse = _coarse_key(step)
        matches = [c for c in node.children if _coarse_key(c.edge_source) == coarse]
        # Only accept the fuzzy match when it is unambiguous.
        return matches[0] if len(matches) == 1 else None

    # -- rendering ---------------------------------------------------------- #
    def render(
        self,
        detail_nodes: list[Node] | None = None,
        max_rows: int = 5,
        max_nodes: int = 60,
    ) -> str:
        """Serialize ``G`` for the agent's observation.

        The structure of every node is always shown (that is what enables
        backtracking), but tuple data is materialized only for ``detail_nodes``
        — otherwise the context would grow with the whole search history.
        """
        detail_ids = {n.id for n in (detail_nodes or [])}
        lines: list[str] = []
        shown = 0
        truncated = False

        def walk(node: Node, indent: str, last: bool) -> None:
            nonlocal shown, truncated
            if shown >= max_nodes:
                truncated = True
                return
            shown += 1
            if node.is_root:
                lines.append(f"n0 [root: source tables]  -> {node.state_summary()}")
            else:
                branch = "`-- " if last else "|-- "
                raw = node.__dict__.get("_raw_source", "")
                src = node.edge_source or raw or "<unparsed>"
                if node.failed:
                    lines.append(f"{indent}{branch}{node.id} x {src}")
                    lines.append(f"{indent}{'    ' if last else '|   '}  ERROR: {node.error}")
                else:
                    mark = " *" if node.id in detail_ids else ""
                    lines.append(
                        f"{indent}{branch}{node.id}{mark} {src}  -> {node.state_summary()}"
                    )
            child_indent = indent + ("    " if last else "|   ") if not node.is_root else ""
            for i, c in enumerate(node.children):
                walk(c, child_indent, i == len(node.children) - 1)

        walk(self.root, "", True)
        if truncated:
            lines.append(f"... (tree truncated at {max_nodes} nodes)")

        out = ["## Agentic Reasoning Tree", "```", *lines, "```"]

        for node in detail_nodes or []:
            if node.state is None:
                continue
            out.append(f"\n### State of node {node.id}")
            out.append(f"Pipeline from root: {node.pipeline_source(joiner=' -> ') or '(none)'}")
            out.append(serialize_table_set(node.state, max_rows=max_rows, with_description=False))
        return "\n".join(out)
