# DeepPrep

An implementation of **"DeepPrep: An LLM-Powered Agentic System for Autonomous Data Preparation"**
([arXiv:2602.07371](https://arxiv.org/abs/2602.07371), PVLDB Vol. 19 — Fan et al., Renmin University of China & ByteDance).

Autonomous Data Preparation (ADP): given a set of messy **source tables** `S` and a
natural-language **target schema** `Σ*`, automatically synthesize and execute an
operator **pipeline** `P` that produces an analysis-ready target table `T̂`.

```
                       ┌──────────────────────────────────────────┐
  Source tables  S ───▶ │  ⟨plan⟩ → ⟨expand⟩ → ⟨execute⟩  (loop)   │ ───▶  T̂
  Target schema Σ* ───▶ │      over an agentic reasoning tree      │       + pipeline P
                       └──────────────────────────────────────────┘
                                     ▲            │
                                     └─ feedback ─┘  (materialized tables / error traces)
```

The key idea of the paper is that pipeline construction is **not linear**. Code-generation
methods decide without grounding in intermediate results; ReAct-style agents ground in
execution feedback but cannot revise an early decision once later operators depend on it.
DeepPrep keeps every materialized intermediate state in a **tree**, so when execution
feedback invalidates an early choice the agent can *backtrack to that node*, expand an
alternative branch, and reuse the valid operator prefix.

---

## What is implemented

| Paper section | Component | Module |
|---|---|---|
| §2.1 | Problem formulation: `S_i=(Σ_i,D_i)`, `Σ=(τ,C)`, pipeline `P` | `deepprep.types` |
| §2.2 | **All 31 operators** in 8 categories | `deepprep.operators` |
| §3 | Execution environment (materializes states, returns runtime feedback) | `deepprep.env` |
| §4.1 | Agentic reasoning tree `G=(N,E)` with prefix-matching node references | `deepprep.tree` |
| §4.2 | `⟨plan⟩ / ⟨expand⟩ / ⟨execute⟩ / ⟨answer⟩` inference procedure | `deepprep.agent` |
| §5.1 | Cold-start curriculum: operator-syntax SFT (Eq. 4) + reasoning SFT (Eq. 5) | `deepprep.training` |
| §5.2 | Multi-turn GRPO with hybrid reward `R = αR_out + βR_part + γR_llm` (Eq. 6–8) | `deepprep.training` |
| §5.3 | Data synthesis: NL2SQL → ADP + **reversible** noise injection | `deepprep.synthesis` |
| §6.1 | Metrics: permutation-invariant exact match, completion rate, cost | `deepprep.eval` |
| §6.1 | Baselines: CodeGen, Plan-and-Solve, ReAct, MCTS-OP | `deepprep.baselines` |

### The 31 operators (§2.2)

| Category | Operators |
|---|---|
| 2.2.1 Data Cleaning (5) | `DropNA` `MissingValueImputation` `Deduplicate` `ErrorDetection` `OutlierDetection` |
| 2.2.2 Value Normalization (3) | `ValueTransform` `StandardizeDatetime` `CastType` |
| 2.2.3 Schema Editing (7) | `RenameColumn` `AddNewColumn` `DropColumn` `SplitColumn` `Concatenate` `SelectColumn` `Subtitle` |
| 2.2.4 Row Selection (3) | `Filter` `Sort` `TopK` |
| 2.2.5 Aggregation (3) | `GroupBy` `Count` `CalculateStatistic` |
| 2.2.6 Table Combination (3) | `Join` `Union` `Append` |
| 2.2.7 Table Reshaping (5) | `Pivot` `Stack` `WideToLong` `Transpose` `Explode` |
| 2.2.8 Program Synthesis (1) | `ExeCode` |
| Control (1) | `Terminate` — closes the pipeline (Figure 4) |

---

## Install

```bash
uv venv --python 3.12 && uv pip install -e ".[dev,llm]"
# for the training stack (torch, transformers, peft):
uv pip install -e ".[train]"
```

## Quickstart

Run the paper's Figure-2 example (movies / ratings / directors) end to end:

```bash
deepprep demo                       # deterministic scripted agent, no API key needed
deepprep run examples/movies_demo/task.json --model gpt-4o-mini
```

Programmatic use:

```python
from deepprep import DeepPrepAgent, ADPTask
from deepprep.agent.llm import OpenAIClient

task   = ADPTask.load("examples/movies_demo/task.json")
agent  = DeepPrepAgent(llm=OpenAIClient(model="gpt-4o-mini"), max_turns=5)
result = agent.solve(task)

print(result.pipeline_source)   # the extracted root-to-leaf pipeline P*
print(result.table)             # the produced target table T̂
```

Evaluate on a dataset:

```bash
deepprep eval data/synth_spider/test.jsonl --model gpt-4o-mini --out runs/spider.json
```

## Training (§5)

The three stages of the *Progressive Agentic Training* framework:

```bash
# Stage 1 -- operator syntax learning (Eq. 4)
deepprep train-op-syntax  --tasks data/synth_spider/train.jsonl --model Qwen/Qwen3-8B

# Stage 2 -- reasoning procedure learning (Eq. 5), distilled from a teacher
deepprep distill          --tasks data/synth_spider/train.jsonl --teacher deepseek-reasoner
deepprep train-reasoning  --trajectories data/trajectories.jsonl --model <stage1-ckpt>

# Stage 3 -- multi-turn GRPO with the hybrid reward (Eq. 6)
deepprep train-grpo       --tasks data/synth_spider/train.jsonl --model <stage2-ckpt>
```

## Data synthesis (§5.3)

```bash
deepprep synthesize --spider data/spider --split train --out data/synth_spider/train.jsonl
```

Converts NL2SQL cases into ADP tasks, then applies **reversible noise injection**: each
corruption is the *inverse* of a cleaning operator and is kept only if applying that
cleaning operator restores the previous table state exactly.

---

## Safety note

`ExeCode` and the function-valued operator parameters execute model-generated Python.
`deepprep.operators.sandbox` restricts the namespace (no `os`, `open`, `eval`, no dunder
traversal) but **this is a guardrail, not a security boundary**. Run untrusted workloads in
a container, or disable code execution entirely:

```bash
export DEEPPREP_ALLOW_EXEC=0
```

## Tests

```bash
uv run pytest -q          # no network access required; uses a scripted mock LLM
```

## License

Apache-2.0. This is an independent reimplementation from the paper text; it is not
affiliated with the authors.
