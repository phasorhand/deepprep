"""The closing <answer> turn (paper Sec 4.3).

Sec 6.1 caps *exploration* at 5 turns.  Emitting ``<answer>`` is not exploration:
it selects an already-materialized leaf.  Before this behaviour existed, the
last-turn prompt told the model to "make this expansion the one that completes
the pipeline, then answer" -- and then the loop returned immediately, so the
"then answer" it asked for was impossible.  A model that spent its whole budget
exploring was scored INCOMPLETE even when a leaf state already matched the
target exactly, which is what DeepSeek-chat does on the paper's own Figure-2
example.
"""

from __future__ import annotations

from deepprep.agent import DeepPrepAgent
from deepprep.agent.llm import ScriptedClient

_EXPAND = (
    "<plan>keep exploring</plan>"
    "<expand><from></from><ops>Deduplicate(movies, [id], first)</ops></expand>"
)
_ANSWER = "<plan>done</plan><answer>Terminate([movies])</answer>"


def test_expansion_budget_is_followed_by_one_closing_answer_turn(demo_task):
    """A model that explores to the last turn still gets to commit an answer."""
    client = ScriptedClient([_EXPAND] * 3 + [_ANSWER])
    result = DeepPrepAgent(llm=client, max_turns=3).solve(demo_task)

    assert len(client.calls) == 4, "the closing turn must be an extra LLM call"
    assert result.stop_reason == "answered"
    assert result.completed
    assert result.table is not None


def test_the_closing_turn_forbids_further_expansion(demo_task):
    """The budget is a budget: a 4th <expand> must not grow the tree."""
    client = ScriptedClient([_EXPAND] * 3 + [_EXPAND])
    result = DeepPrepAgent(llm=client, max_turns=3).solve(demo_task)

    assert result.stop_reason == "max_turns"
    assert not result.completed
    assert result.fallback_table is not None
    # Three expansions were requested and three nodes exist besides the root.
    assert len(result.tree.nodes) == 1 + 3


def test_the_closing_prompt_asks_for_an_answer_and_offers_the_leaves(demo_task):
    client = ScriptedClient([_EXPAND] * 2 + [_ANSWER])
    DeepPrepAgent(llm=client, max_turns=2).solve(demo_task)

    closing = client.calls[-1][-1]["content"]
    assert closing.count("<answer>") >= 1
    assert "<expand>" not in closing


def test_a_model_that_answers_in_budget_gets_no_extra_call(demo_task):
    """The closing turn is a fallback, not an unconditional tax on every run."""
    client = ScriptedClient([_EXPAND, _ANSWER])
    result = DeepPrepAgent(llm=client, max_turns=5).solve(demo_task)

    assert result.stop_reason == "answered"
    assert len(client.calls) == 2


def test_the_closing_turn_can_be_disabled(demo_task):
    client = ScriptedClient([_EXPAND] * 2 + [_ANSWER])
    result = DeepPrepAgent(llm=client, max_turns=2, final_answer_turn=False).solve(demo_task)

    assert len(client.calls) == 2
    assert result.stop_reason == "max_turns"
