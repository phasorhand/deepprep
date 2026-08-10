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
# Stage 1 -- operator syntax learning (Eq. 4): build D_op, then fine-tune on it
deepprep build-op-syntax --tasks data/synth_spider/train.jsonl --out data/sft/op_syntax.jsonl
deepprep train-sft       --data data/sft/op_syntax.jsonl --model Qwen/Qwen2.5-0.5B-Instruct \
                         --out checkpoints/sft-stage1 --lora

# Stage 2 -- reasoning procedure learning (Eq. 5), distilled from a teacher
deepprep distill         --tasks data/synth_spider/train.jsonl --out data/sft/distilled.jsonl \
                         --teacher deepseek-reasoner --judge deepseek-chat
deepprep train-sft       --data data/sft/distilled.jsonl --model checkpoints/sft-stage1 \
                         --out checkpoints/sft-stage2 --lora

# Stage 3 -- multi-turn GRPO with the hybrid reward (Eq. 6)
deepprep train-grpo      --tasks data/synth_spider/train.jsonl --model checkpoints/sft-stage2
```

`train-grpo` exposes the paper's settings (group 8, 2048 new tokens, bf16), which assume a
GPU. `scripts/grpo_smoke.py` shrinks every dimension until one step fits on a laptop CPU —
useful for checking the rollout → reward → masked-policy-gradient loop actually runs.

## Data synthesis (§5.3)

```bash
deepprep synthesize \
  --db-root data/spider/database \
  --spec    data/spider/train_spider.json \
  --out     data/synth_spider/train.jsonl
```

Converts NL2SQL cases into ADP tasks: the gold SQL is executed to obtain `T*`, translated
into an operator pipeline, and the **shortest candidate that exactly reproduces `T*`** is
kept. Then **reversible noise injection** dirties the sources — each corruption is the
*inverse* of a cleaning operator and is retained only if applying that cleaning operator
restores the previous table state exactly. The final gold pipeline is the cleaning pipeline
concatenated with the task pipeline.

Everything runs offline; pass `--use-llm --model ...` to use an LLM at the three points the
paper does (target schema spec, candidate pipelines, inverse transformation logic).

Spider is a 1.3 GB download. `examples/mini_spider/build_db.py` writes a fixture with the
same shape — `database/<db_id>/<db_id>.sqlite` plus a spec JSON — so the whole synthesis
path is runnable in a second:

```bash
python examples/mini_spider/build_db.py data/mini_spider
deepprep synthesize --db-root data/mini_spider/database \
                    --spec data/mini_spider/spec.json --out data/tasks/synth_train.jsonl
```

---

## Fidelity notes

Where the paper is silent or under-specified, these are the choices made and why. They are
all reachable from code comments too; this is the index.

| Topic | Paper | Here |
|---|---|---|
| Operator count | "31 operators" in 8 categories, but §2.2 enumerates 30 (5+3+7+3+3+3+5+1) | The 31st is `Terminate`, the control operator Figure 4 shows closing an extracted pipeline |
| `α, β, γ` (Eq. 6) | not published | `0.7 / 0.3 / 0.15`, chosen so a correct trajectory *always* outranks an incorrect one — otherwise `R_llm` would itself be hackable |
| `R_llm` judge | "an instruction-tuned LLM (e.g. GPT-4o)" | `LLMProcessJudge` does exactly that; `HeuristicProcessJudge` is a deterministic structural approximation so RL and the tests run without a second model |
| `S_cnt` (Eq. 8) | positional: `1[D_hat[c]_i = D*[c]_i]` | rows are canonically sorted first, otherwise a pure row permutation would score ≈0 despite being an exact match |
| Exact match | "invariant to row and column **permutations**" | column *names* are required by default (a rename is not a permutation, and §5.2 names renaming as the canonical reward hack). `MatchOptions(require_column_names=False)` restores value-signature matching |
| Float equality | "exact cell-value equality" | quantized to `float_tol=1e-6`; two pipelines computing the same mean in a different operator order differ in the last ULP |
| `ErrorDetection` | `(table, column, func)`, "*identifies* invalid records" | defaults to `action='flag'`; `'remove'`/`'null'` are opt-in extensions |
| `Explode` | "a column containing **list-valued** entries" | string splitting requires an explicit `sep`, so a plain text column is never silently shredded |
| `Count` / `CalculateStatistic` | described as returning scalars | materialized as 1×1 tables, since every operator must be `T → T` (§2.1) |
| `<execute>` transport | trajectory `r_t` "encapsulates both the current tree state and the agent's generated response" | environment output is a *user* message, so §5.2's "mask tokens inside `<execute>`" reduces to masking to assistant tokens. `refine_mask_to_action_tags` handles the inlined form |
| Node addressing | prefix-matching constraint | a bare `n2` reference is also accepted (Figure 4's plans say "rollback to n2"); an inexact prefix resolves but **always** returns a warning, and `<answer>` resolution is exact-only |
| Turn budget | "maximum exploration turns … set to 5" (§6.1) | 5 turns bound *exploration*; a model that spends all of them still gets one closing turn in which only `<answer>` is honoured. Without it the last-turn prompt asks for an answer the loop never allows, and a run whose leaf already matches `T*` is scored INCOMPLETE. `DeepPrepAgent(final_answer_turn=False)` restores the strict reading |
| Self-joins | not discussed | the rule-based SQL→pipeline translator refuses them rather than emitting a wrong pipeline; `--use-llm` is the fallback, as in the paper |

## Safety note

`ExeCode` and the function-valued operator parameters execute model-generated Python.
`deepprep.operators.sandbox` restricts the namespace to a whitelist — `pandas`, `numpy`,
`re`, `math`, `datetime`, `json`, `statistics` are importable; `os`, `open`, `eval` and
dunder traversal are not — but **this is a guardrail, not a security boundary**. Run
untrusted workloads in a container, or disable code execution entirely:

```bash
export DEEPPREP_ALLOW_EXEC=0
```

## Tests

```bash
uv run pytest -q          # no network access required; uses a scripted mock LLM
```

The suite never calls an API. The SFT loop tests build a randomly initialized model in
process and are skipped unless the `train` extra is installed; everything else runs with
`pandas` alone.

## License

Apache-2.0. This is an independent reimplementation from the paper text; it is not
affiliated with the authors.
