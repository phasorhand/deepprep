"""Agentic reasoning tree and inference loop (paper Sec 4).

The load-bearing test in this file is
:func:`test_agent_backtracks_to_an_earlier_node_and_reuses_the_valid_prefix`,
which is the paper's core claim made executable.
"""

from __future__ import annotations

import pandas as pd
import pytest

from deepprep.agent import DeepPrepAgent, parse_agent_output
from deepprep.agent.llm import ScriptedClient
from deepprep.env import Environment, EnvironmentLimits
from deepprep.eval import table_match
from deepprep.operators import parse_pipeline
from deepprep.tree import ReasoningTree
from deepprep.types import Table, TableSet


# --------------------------------------------------------------------------- #
# Environment (Sec 3)
# --------------------------------------------------------------------------- #
def test_environment_keeps_the_states_of_a_successful_prefix(simple_tables):
    """Sec 4.2: operators execute sequentially; the successful prefix survives a
    later failure. This is what makes "reuse valid operator prefixes" possible."""
    env = Environment(simple_tables)
    result = env.execute(
        env.initial_state(),
        parse_pipeline(
            "Deduplicate(people, [id], first)\n"
            "DropColumn(people, [tags])\n"
            "SelectColumn(people, [does_not_exist])"
        ),
    )
    assert not result.success
    assert result.n_applied == 2
    assert len(result.states) == 2
    assert "MissingColumn" in result.error


def test_environment_feedback_reports_the_prefix_and_the_error(simple_tables):
    env = Environment(simple_tables)
    result = env.execute(env.initial_state(), "Deduplicate(people)\nSelectColumn(people, [nope])")
    fb = env.render_feedback(result)
    assert "[1/1] OK" in fb
    assert "FAILED" in fb and "ERROR:" in fb
    assert "were applied and their" in fb  # tells the agent the prefix is kept


def test_environment_reports_parse_errors_as_feedback(simple_tables):
    env = Environment(simple_tables)
    result = env.execute(env.initial_state(), "NotAnOperator(x)")
    assert not result.success
    assert "ParseError" in result.error
    assert result.n_applied == 0


def test_environment_rejects_an_empty_expansion(simple_tables):
    env = Environment(simple_tables)
    result = env.execute(env.initial_state(), "just some prose")
    assert "EmptyPipeline" in result.error


def test_environment_enforces_a_state_size_limit():
    big = pd.DataFrame({"k": [1] * 400})
    ts = TableSet([Table("a", big), Table("b", big.copy())])
    env = Environment(ts, limits=EnvironmentLimits(max_total_rows=1000))
    result = env.execute(env.initial_state(), "Join(a, b, on=k, how=inner, target=x)")
    assert not result.success
    assert "StateTooLarge" in result.error


# --------------------------------------------------------------------------- #
# Tree structure (Sec 4.1)
# --------------------------------------------------------------------------- #
def test_root_holds_the_source_tables(simple_tables):
    tree = ReasoningTree(simple_tables.copy())
    assert tree.root.id == "n0"
    assert tree.root.is_root and tree.root.state is not None
    assert tree.root.path_ops() == []


def test_path_from_root_to_a_node_is_the_pipeline(simple_tables):
    tree = ReasoningTree(simple_tables.copy())
    env = Environment(simple_tables)
    ops = parse_pipeline("Deduplicate(people, [id], first)\nDropColumn(people, [tags])")
    result = env.execute(tree.root.state, ops)

    node = tree.root
    for op, state in zip(result.applied, result.states, strict=False):
        node = tree.add_state(node, op, state)

    assert node.depth == 2
    assert [o.name for o in node.path_ops()] == ["Deduplicate", "DropColumn"]
    assert "Deduplicate" in node.pipeline_source()


def test_failure_node_has_no_materialized_state(simple_tables):
    """Sec 4.2: "no new state node is created; instead, the parent node is
    annotated with a failure state that records the error trace"."""
    tree = ReasoningTree(simple_tables.copy())
    failed = tree.add_failure(tree.root, None, "SelectColumn(x, [y])", "MissingColumn: ...")
    assert failed.failed
    assert failed.state is None
    assert failed in tree.failures()
    assert failed not in tree.leaves(ok_only=True)


# --------------------------------------------------------------------------- #
# Prefix-matching node addressing (Sec 4.2)
# --------------------------------------------------------------------------- #
@pytest.fixture
def two_branch_tree(simple_tables):
    """n0 -> n1(Deduplicate) -> {n2(DropColumn tags), n3(DropColumn dept)}."""
    tree = ReasoningTree(simple_tables.copy())
    env = Environment(simple_tables)

    ops = parse_pipeline("Deduplicate(people, [id], first)")
    r = env.execute(tree.root.state, ops)
    n1 = tree.add_state(tree.root, r.applied[0], r.states[0])

    for cols in ("[tags]", "[dept]"):
        r2 = env.execute(n1.state, parse_pipeline(f"DropColumn(people, {cols})"))
        tree.add_state(n1, r2.applied[0], r2.states[0])
    return tree


def test_empty_prefix_resolves_to_the_root(two_branch_tree):
    node, warn = two_branch_tree.resolve_parent([])
    assert node.id == "n0" and warn is None


def test_exact_prefix_resolves_to_the_right_branch(two_branch_tree):
    node, warn = two_branch_tree.resolve_parent(
        ["Deduplicate(people, [id], first)", "DropColumn(people, [dept])"]
    )
    assert warn is None
    assert node.edge_source == "DropColumn(people, [dept])"


def test_prefix_matching_tolerates_whitespace_and_quote_differences(two_branch_tree):
    node, warn = two_branch_tree.resolve_parent(
        ['Deduplicate( "people" , [ id ] , first )']
    )
    assert warn is None
    assert node.id == "n1"


def test_ambiguous_fuzzy_match_is_not_guessed(two_branch_tree):
    """Both branches are DropColumn(people, ...); the coarse key alone cannot
    disambiguate them, so resolution must fall back rather than pick one."""
    node, warn = two_branch_tree.resolve_parent(
        ["Deduplicate(people, [id], first)", "DropColumn(people, [something_else])"]
    )
    assert warn is not None and "PrefixMismatch" in warn
    assert node.id == "n1"  # the longest matching prefix


def test_prefix_mismatch_reports_the_available_edges(two_branch_tree):
    node, warn = two_branch_tree.resolve_parent(["Sort(people, [id])"])
    assert "PrefixMismatch" in warn
    assert "Available edges from n0" in warn
    assert node.id == "n0"


def test_node_id_reference_is_accepted_as_a_fallback(two_branch_tree):
    # Figure 4's plans say things like "rollback to n2", so refusing an id
    # reference outright would waste turns.
    node, warn = two_branch_tree.resolve_parent("n1")
    assert warn is None and node.id == "n1"

    node, warn = two_branch_tree.resolve_parent("n99")
    assert "UnknownNode" in warn and node.id == "n0"


def test_resolve_longest_prefix_returns_the_unexecuted_remainder(two_branch_tree):
    node, remaining = two_branch_tree.resolve_longest_prefix(
        ["Deduplicate(people, [id], first)", "DropColumn(people, [tags])", "Sort(people, [id])"]
    )
    assert node.edge_source == "DropColumn(people, [tags])"
    assert remaining == ["Sort(people, [id])"]


def test_render_shows_structure_and_error_traces(two_branch_tree):
    two_branch_tree.add_failure(
        two_branch_tree.root, None, "Bad(x)", "MissingColumn: No column named 'year'"
    )
    out = two_branch_tree.render()
    assert "n0 [root" in out
    assert "ERROR: MissingColumn" in out
    assert "DropColumn(people, [tags])" in out


# --------------------------------------------------------------------------- #
# Action parsing (Sec 4.2)
# --------------------------------------------------------------------------- #
def test_parses_plan_and_expand_with_prefix():
    turn = parse_agent_output(
        "<plan>do the thing</plan>\n"
        "<expand>\n<from>\nDeduplicate(t)\n</from>\n<ops>\nSort(t, [a])\n</ops>\n</expand>"
    )
    assert turn.plan.text == "do the thing"
    assert turn.expand.from_prefix == ["Deduplicate(t)"]
    assert "Sort(t, [a])" in turn.expand.ops_source


def test_empty_from_block_means_the_root():
    turn = parse_agent_output("<plan>p</plan><expand><from>\n</from><ops>\nSort(t,[a])\n</ops></expand>")
    assert turn.expand.from_prefix == []


def test_expand_without_explicit_blocks_is_still_usable():
    turn = parse_agent_output("<plan>p</plan>\n<expand>\nSort(t, [a])\n</expand>")
    assert turn.expand.from_prefix == []
    assert "Sort(t, [a])" in turn.expand.ops_source


def test_parent_header_line_is_tolerated():
    turn = parse_agent_output("<plan>p</plan>\n<expand>\nparent: n2\nSort(t, [a])\n</expand>")
    assert turn.expand.from_prefix == ["n2"]
    assert "Sort" in turn.expand.ops_source
    assert "parent:" not in turn.expand.ops_source


def test_answer_takes_precedence_over_expand():
    turn = parse_agent_output("<plan>p</plan><expand>Sort(t,[a])</expand><answer>Terminate([t])</answer>")
    assert turn.is_terminal
    assert turn.expand is None


def test_think_blocks_are_stripped():
    turn = parse_agent_output("<think>reasoning models emit this</think><plan>p</plan><expand>Sort(t,[a])</expand>")
    assert turn.plan.text == "p"


def test_unclosed_block_from_a_truncated_generation_is_recovered():
    turn = parse_agent_output("<plan>a plan that got cut off")
    assert turn.plan is not None
    assert "MissingAction" in turn.error


def test_output_with_no_tags_yields_actionable_feedback():
    turn = parse_agent_output("Sure! Here is some code.")
    assert turn.error is not None and "MissingAction" in turn.error
    assert "<expand>" in turn.error


# --------------------------------------------------------------------------- #
# End-to-end inference (Sec 4.2, Figure 4)
# --------------------------------------------------------------------------- #
def test_agent_solves_the_running_example(demo_task, figure4_trajectory):
    agent = DeepPrepAgent(llm=ScriptedClient(figure4_trajectory), max_turns=5)
    result = agent.solve(demo_task)

    assert result.stop_reason == "answered"
    assert result.completed
    assert table_match(result.table, demo_task.target_table)


def test_agent_backtracks_to_an_earlier_node_and_reuses_the_valid_prefix(
    demo_task, figure4_trajectory
):
    """The paper's central claim, made executable.

    Sec 1: "Instead of restarting the entire process, it can backtrack to the
    execution state where the incorrect decision was made, expand an alternative
    branch with the corrected operation, and reuse valid operator prefixes from
    existing partial pipelines to produce a correct result."
    """
    result = DeepPrepAgent(llm=ScriptedClient(figure4_trajectory), max_turns=5).solve(demo_task)
    tree = result.tree

    # A failure was recorded, with no materialized state (Sec 4.2).
    failures = tree.failures()
    assert len(failures) == 1
    assert failures[0].state is None
    assert "year" in failures[0].error

    # The agent then expanded from an ANCESTOR of the failed branch, not from the
    # frontier -- that is the non-local revision.
    assert result.n_backtracks == 1
    backtrack_step = next(s for s in result.trajectory if s.is_backtrack)
    reused = tree.get(backtrack_step.parent_node_id)
    assert reused.depth == 2, "should have returned to n2, before the wrong SelectColumn"

    # The abandoned branch is still in the tree (states are preserved, not undone)
    # and the corrected branch is a sibling of it.
    dead_branch = next(
        c for c in reused.children if c.edge_source.startswith("SelectColumn(ratings, [movie, rating]")
    )
    assert dead_branch.children and dead_branch.children[0].failed
    assert len(reused.children) == 2

    # The valid prefix was reused rather than recomputed: the final pipeline
    # starts with exactly the operators leading to the backtrack node.
    prefix = [op.to_source() for op in reused.path_ops()]
    final = [op.to_source() for op in result.pipeline]
    assert final[: len(prefix)] == prefix


def test_agent_reports_max_turns_and_keeps_a_fallback(demo_task):
    """Sec 6.1: exceeding the interaction limit counts as INCOMPLETE."""
    responses = [
        "<plan>step</plan><expand><from></from><ops>Deduplicate(movies, [id], first)</ops></expand>"
    ] * 3
    result = DeepPrepAgent(llm=ScriptedClient(responses), max_turns=3).solve(demo_task)

    assert result.stop_reason == "max_turns"
    assert not result.completed
    assert result.table is None
    assert result.fallback_table is not None  # still gives RL a partial signal


def test_agent_survives_a_malformed_turn_and_says_why(demo_task):
    responses = [
        "I'll just describe it in prose.",
        "<plan>p</plan><answer>Terminate([movies])</answer>",
    ]
    result = DeepPrepAgent(llm=ScriptedClient(responses), max_turns=2).solve(demo_task)
    assert result.trajectory[0].error is not None
    assert "MissingAction" in result.trajectory[0].error
    assert result.stop_reason == "answered"


def test_agent_survives_an_llm_outage(demo_task):
    class Dead:
        model = "dead"

        def generate(self, *a, **k):
            raise RuntimeError("connection refused")

    result = DeepPrepAgent(llm=Dead(), max_turns=3).solve(demo_task)
    assert result.stop_reason == "llm_error"
    assert not result.completed
    assert "connection refused" in result.error


def test_empty_answer_table_is_rejected_as_incomplete(demo_task):
    """Sec 6.1: "produces an empty result" counts as incomplete."""
    responses = [
        "<plan>p</plan><answer>\n"
        "Filter(movies, lambda r: False)\n"
        "Terminate([movies])\n"
        "</answer>",
        "<plan>giving up</plan><expand><from></from><ops>Sort(movies, [id])</ops></expand>",
    ]
    result = DeepPrepAgent(llm=ScriptedClient(responses), max_turns=2).solve(demo_task)
    assert not result.completed
    assert any("EmptyResult" in (s.error or "") for s in result.trajectory)


def test_ground_truth_never_reaches_the_model(demo_task, figure4_trajectory):
    """Sec 2.1: T* "is used only for evaluation".

    A value the agent computes itself is not a leak, so the canary has to be a
    sentinel that exists *only* in the ground-truth table and is underivable from
    the sources.
    """
    canary = "ZZ_GROUND_TRUTH_CANARY_ZZ"
    demo_task.target_table = demo_task.target_table.copy()
    demo_task.target_table["director_name"] = canary

    client = ScriptedClient(figure4_trajectory)
    DeepPrepAgent(llm=client, max_turns=5).solve(demo_task)

    everything = "\n".join(m["content"] for call in client.calls for m in call)
    assert canary not in everything


def test_prefix_matching_is_invariant_to_positional_vs_keyword_form(two_branch_tree):
    """The same edge can be written either way; a cosmetic difference must not
    silently restart the agent from the root."""
    node, warn = two_branch_tree.resolve_parent(
        ["Deduplicate(table='people', subset=['id'], keep='first')"]
    )
    assert warn is None
    assert node.id == "n1"


def test_trajectory_separates_agent_and_environment_tokens(demo_task, figure4_trajectory):
    """Sec 5.2 masks environment tokens out of the policy gradient, so the
    trajectory must keep the two cleanly apart."""
    result = DeepPrepAgent(llm=ScriptedClient(figure4_trajectory), max_turns=5).solve(demo_task)
    for step in result.trajectory:
        assert "<plan>" in step.response or "<answer>" in step.response
        assert "<execute>" not in step.response
    # The environment's output is delivered as a user message.
    assert all(m["role"] in ("system", "user", "assistant") for m in result.messages)
    for m in result.messages:
        if m["role"] == "assistant":
            assert "Current state:" not in m["content"]
