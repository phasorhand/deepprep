"""Agentic interaction actions (paper Sec 4.2).

    "We regulate this interaction using control tags aligned with the components
     in Eq. (2) and Eq. (3)."

    * ``<plan>``    — "produces a high-level logical plan for the next reasoning
      step ... determines whether to continue expanding the tree or terminate ...
      it identifies the parent node to extend and the candidate operations to apply."
    * ``<expand>``  — "generates a candidate operator pipeline expanded from a
      selected parent node n_parent ... the parent node is specified using a
      prefix-matching constraint."
    * ``<execute>`` — written by the *environment*, not the policy.  Sec 5.2
      masks these tokens out of the policy gradient.
    * ``<answer>``  — "terminates the reasoning process and returns the final
      pipeline ... The root-to-leaf operator path associated with that node is
      extracted as the final pipeline P*."

    "We extend the LLM vocabulary with the action tags defined above, so that our
     tree-based reasoning can be expressed as a structured language modeling process."

Concrete surface syntax
-----------------------
``<expand>`` carries the prefix reference in a nested ``<from>`` block and the new
operators in ``<ops>``::

    <expand>
    <from>
    Deduplicate(movies, [id], first)
    ValueTransform(movies, title, lambda x: str(x).strip().lower())
    </from>
    <ops>
    SelectColumn(ratings, [movie, values])
    SplitColumn(ratings, values, [rating_value, year], sep=" (")
    </ops>
    </expand>

An empty or omitted ``<from>`` addresses the root ``n_0``.

The parser is deliberately forgiving — a malformed action costs a turn but must
never crash a run, and the diagnostic is fed back to the agent verbatim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "ACTION_TAGS",
    "AgentTurn",
    "AnswerAction",
    "ExpandAction",
    "PlanAction",
    "extract_block",
    "parse_agent_output",
    "strip_think",
]

#: The tags added to the LLM vocabulary (Sec 4.2, "Agentic Inference Procedure").
ACTION_TAGS = ("plan", "expand", "execute", "answer")


def _block_re(tag: str) -> re.Pattern[str]:
    return re.compile(rf"<{tag}\s*>(.*?)</{tag}\s*>", re.DOTALL | re.IGNORECASE)


def _open_re(tag: str) -> re.Pattern[str]:
    return re.compile(rf"<{tag}\s*>(.*)", re.DOTALL | re.IGNORECASE)


def extract_block(text: str, tag: str, allow_unclosed: bool = True) -> str | None:
    """Return the contents of ``<tag>...</tag>``, or ``None`` if absent.

    When ``allow_unclosed`` and the generation was cut off mid-block, the text
    after the opening tag is returned; truncated output is common at the token
    limit and discarding it wholesale would waste the turn.
    """
    m = _block_re(tag).search(text)
    if m:
        return m.group(1).strip()
    if allow_unclosed:
        m = _open_re(tag).search(text)
        if m:
            body = m.group(1)
            # Stop at the next opening tag so blocks don't swallow each other.
            nxt = re.search(r"<(?:" + "|".join(ACTION_TAGS) + r")\s*>", body, re.IGNORECASE)
            if nxt:
                body = body[: nxt.start()]
            body = body.strip()
            if body:
                return body
    return None


def strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` spans emitted by reasoning models (e.g. Qwen3)."""
    text = re.sub(r"<think\s*>.*?</think\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # An unclosed <think> means the model never left the reasoning phase.
    text = re.sub(r"<think\s*>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


@dataclass
class PlanAction:
    """``psi_t`` — the high-level logical plan for step ``t``."""

    text: str

    #: Whether the plan announces termination.  Only advisory: the actual
    #: terminating action is ``<answer>``.
    @property
    def signals_terminate(self) -> bool:
        t = self.text.lower()
        return any(k in t for k in ("terminate", "final answer", "we are done", "is complete"))


@dataclass
class ExpandAction:
    """``P_t`` — a candidate operator pipeline plus its parent-node reference."""

    #: The root-to-parent operator sequence (the prefix-matching constraint).
    from_prefix: list[str] = field(default_factory=list)
    #: The operators to append.
    ops_source: str = ""
    #: Raw block text, kept for trajectory serialization.
    raw: str = ""


@dataclass
class AnswerAction:
    """``A`` — the final pipeline ``P*`` and the table it produces."""

    #: The full root-to-leaf operator sequence the agent claims is the answer.
    pipeline_source: str = ""
    raw: str = ""


@dataclass
class AgentTurn:
    """The parsed output of one policy generation."""

    raw: str
    plan: PlanAction | None = None
    expand: ExpandAction | None = None
    answer: AnswerAction | None = None
    #: Populated when nothing actionable could be parsed.
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.answer is not None


_FROM_RE = _block_re("from")
_OPS_RE = _block_re("ops")
_PARENT_LINE_RE = re.compile(
    r"^\s*(?:parent|from|parent_node|node)\s*[:=]\s*(.+)$", re.IGNORECASE | re.MULTILINE
)


def _parse_expand(block: str) -> ExpandAction:
    """Split an ``<expand>`` block into its prefix reference and its operators."""
    raw = block
    m_from = _FROM_RE.search(block)
    m_ops = _OPS_RE.search(block)

    prefix_text = ""
    if m_from:
        prefix_text = m_from.group(1).strip()
    else:
        # Tolerate `parent: n2` / `from: Deduplicate(...)` header lines.
        m_line = _PARENT_LINE_RE.search(block)
        if m_line:
            prefix_text = m_line.group(1).strip()
            block = block[: m_line.start()] + block[m_line.end() :]

    if m_ops:
        ops_source = m_ops.group(1).strip()
    else:
        ops_source = _FROM_RE.sub("", block).strip()

    from ..operators import split_calls

    if prefix_text and re.fullmatch(r"n\d+", prefix_text.strip()):
        prefix = [prefix_text.strip()]  # node-id fallback, resolved by the tree
    else:
        prefix = split_calls(prefix_text) if prefix_text else []

    return ExpandAction(from_prefix=prefix, ops_source=ops_source, raw=raw)


def parse_agent_output(text: str) -> AgentTurn:
    """Parse one policy generation into its actions.

    ``<answer>`` takes precedence: it is the terminating action, so a turn that
    emits both a plan and an answer is treated as terminal.
    """
    cleaned = strip_think(text or "")
    turn = AgentTurn(raw=text or "")

    plan_body = extract_block(cleaned, "plan")
    if plan_body:
        turn.plan = PlanAction(text=plan_body)

    answer_body = extract_block(cleaned, "answer")
    if answer_body:
        turn.answer = AnswerAction(pipeline_source=answer_body, raw=answer_body)
        return turn

    expand_body = extract_block(cleaned, "expand")
    if expand_body:
        turn.expand = _parse_expand(expand_body)
        return turn

    # Nothing parseable.  Distinguish "no tags at all" from "a plan but no action",
    # because they need different corrective feedback.
    if turn.plan is not None:
        turn.error = (
            "MissingAction: the output contained <plan> but neither <expand> nor <answer>. "
            "Every turn must end with exactly one <expand>...</expand> (to grow the tree) "
            "or <answer>...</answer> (to finish)."
        )
    else:
        turn.error = (
            "MissingAction: no <plan>/<expand>/<answer> tags were found in the output. "
            "Respond with <plan>...</plan> followed by <expand>...</expand>, or with "
            "<answer>...</answer> when a leaf state already satisfies the target schema."
        )
    return turn
