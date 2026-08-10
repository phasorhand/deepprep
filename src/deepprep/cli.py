"""Command-line interface.

Covers the full lifecycle described in the paper:

    inference   ``demo`` ``run`` ``eval`` ``operators``
    data (5.3)  ``synthesize``
    training    ``build-op-syntax`` (5.1 Stage 1) ``distill`` (5.1 Stage 2)
                ``train-sft`` ``train-grpo`` (5.2)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .types import ADPTask, load_tasks

__all__ = ["main"]

_METHODS = ("deepprep", "codegen", "pas", "react", "mcts")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _build_llm(args: argparse.Namespace) -> Any:
    from .agent.llm import OpenAIClient

    return OpenAIClient(
        model=args.model,
        base_url=getattr(args, "base_url", None),
        api_key=getattr(args, "api_key", None),
    )


def _build_solver(args: argparse.Namespace, llm: Any) -> Any:
    method = args.method.lower()
    if method == "deepprep":
        from .agent import DeepPrepAgent

        return DeepPrepAgent(
            llm=llm,
            max_turns=args.max_turns,
            temperature=args.temperature,
            include_exemplar=args.exemplar,
            verbose=getattr(args, "verbose", False),
        )
    from .baselines import CodeGenAgent, MCTSOperatorAgent, PlanAndSolveAgent, ReActAgent

    table = {
        "codegen": CodeGenAgent,
        "pas": PlanAndSolveAgent,
        "react": ReActAgent,
        "mcts": MCTSOperatorAgent,
    }
    cls = table[method]
    try:
        return cls(llm=llm, max_turns=args.max_turns, temperature=args.temperature)
    except TypeError:
        return cls(llm=llm, temperature=args.temperature)


def _add_llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="gpt-4o-mini", help="backbone model id")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (e.g. a vLLM server)")
    p.add_argument("--api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.01,
                   help="Sec 6.1 sets 0.01 for all prompting methods")
    p.add_argument("--max-turns", type=int, default=5,
                   help="Sec 6.1: maximum exploration turns is 5")
    p.add_argument("--method", default="deepprep", choices=_METHODS)
    p.add_argument("--exemplar", action="store_true",
                   help="include the in-context tree-reasoning exemplar (for untrained backbones)")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_operators(args: argparse.Namespace) -> int:
    from .operators import OPERATORS, operator_manual

    print(f"# DeepPrep operator space O ({len(OPERATORS)} operators, paper Sec 2.2)\n")
    print(operator_manual())
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Replay the paper's Figure-4 trajectory offline."""
    from .agent import DeepPrepAgent
    from .agent.llm import ScriptedClient
    from .eval import table_match

    root = Path(__file__).resolve().parents[2]
    demo_dir = root / "examples" / "movies_demo"
    sys.path.insert(0, str(demo_dir))
    try:
        from scripted_trajectory import FIGURE_4_TRAJECTORY  # type: ignore
    except ImportError:
        print(f"error: demo assets not found under {demo_dir}", file=sys.stderr)
        return 1

    task_path = demo_dir / "task.json"
    if not task_path.exists():
        from build_task import build_task  # type: ignore

        task = build_task()
    else:
        task = ADPTask.load(task_path)

    print("=" * 78)
    print("DeepPrep demo -- the running example of arXiv:2602.07371 (Figures 2 and 4)")
    print("=" * 78)
    print(f"\nTarget schema:\n  {task.target_schema.description}")
    print("  columns: " + ", ".join(c.name for c in task.target_schema.columns))

    agent = DeepPrepAgent(llm=ScriptedClient(FIGURE_4_TRAJECTORY), max_turns=5, verbose=args.verbose)
    result = agent.solve(task)

    print(f"\n{result.tree.render()}")
    print("\n--- extracted pipeline P* ---")
    print(result.pipeline_source)
    print("\n--- produced table T_hat ---")
    print(result.table.to_string(index=False) if result.table is not None else "(none)")

    ok = table_match(result.table, task.target_table)
    print(
        f"\nturns={result.n_turns}  backtracks={result.n_backtracks}  "
        f"failed expansions={result.n_failed_expansions}  "
        f"completed={result.completed}  exact match={ok}"
    )
    if result.n_backtracks == 0:
        print("warning: the demo did not backtrack, which is the behaviour it exists to show")
    return 0 if ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    from .eval import table_match

    task = ADPTask.load(args.task)
    solver = _build_solver(args, _build_llm(args))
    result = solver.solve(task)

    if result.tree is not None:
        print(result.tree.render())
    print("\n--- pipeline ---")
    print(result.pipeline_source or "(none)")
    print("\n--- table ---")
    table = result.table if result.table is not None else result.fallback_table
    print(table.to_string(index=False) if table is not None else "(none)")
    print(
        f"\nstop_reason={result.stop_reason}  turns={result.n_turns}  "
        f"completed={result.completed}  usage={result.usage.to_dict()}"
    )
    if task.target_table is not None:
        print(f"exact match: {table_match(result.table, task.target_table)}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result.to_dict(), indent=2, default=str))
        print(f"wrote {args.out}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval import evaluate

    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    solver = _build_solver(args, _build_llm(args))
    report = evaluate(
        solver,
        tasks,
        method=args.method,
        dataset=Path(args.tasks).stem,
        cost_mode=args.cost_mode,
        max_workers=args.workers,
    )
    print()
    print(report.summary())
    if args.out:
        report.save(args.out)
        print(f"wrote {args.out}")
    return 0


def cmd_synthesize(args: argparse.Namespace) -> int:
    from .synthesis import SynthesisConfig, load_benchmark, synthesize_dataset, write_jsonl

    cases = load_benchmark(args.spec, limit=args.limit)
    print(f"loaded {len(cases)} NL2SQL cases")
    llm = _build_llm(args) if args.use_llm else None
    cfg = SynthesisConfig(dataset=args.dataset, split=args.split, seed=args.seed)
    tasks, stats = synthesize_dataset(
        cases, db_root=args.db_root, config=cfg, llm=llm, verbose=True
    )
    print(stats.summary() if hasattr(stats, "summary") else stats)
    write_jsonl(tasks, args.out)
    print(f"wrote {len(tasks)} tasks to {args.out}")
    return 0


def cmd_build_op_syntax(args: argparse.Namespace) -> int:
    from .training import build_op_syntax_dataset, write_jsonl

    tasks = load_tasks(args.tasks)
    examples = build_op_syntax_dataset(
        tasks, max_span=args.max_span, max_per_task=args.max_per_task, seed=args.seed
    )
    write_jsonl(examples, args.out)
    print(f"wrote {len(examples)} Stage-1 examples to {args.out}")
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    from .training import DistillConfig, distill_trajectories, write_jsonl
    from .training.rewards import HeuristicProcessJudge, LLMProcessJudge

    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    teacher = _build_llm(argparse.Namespace(**{**vars(args), "model": args.teacher}))
    judge = (
        LLMProcessJudge(_build_llm(argparse.Namespace(**{**vars(args), "model": args.judge})))
        if args.judge
        else HeuristicProcessJudge()
    )
    examples, stats = distill_trajectories(
        tasks,
        teacher,
        DistillConfig(
            n_candidates=args.n_candidates,
            top_k=args.top_k,
            temperature=args.temperature,
            max_turns=args.max_turns,
        ),
        judge=judge,
    )
    write_jsonl(examples, args.out)
    print(f"wrote {len(examples)} Stage-2 trajectories to {args.out}")
    return 0


def cmd_train_sft(args: argparse.Namespace) -> int:
    from .training import read_jsonl
    from .training.sft import SFTConfig, SFTTrainer

    examples = list(read_jsonl(args.data))
    print(f"loaded {len(examples)} SFT examples from {args.data}")
    cfg = SFTConfig(
        model_name_or_path=args.model,
        output_dir=args.out,
        learning_rate=args.lr,
        n_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        use_lora=args.lora,
    )
    SFTTrainer(cfg).train(examples)
    return 0


def cmd_train_grpo(args: argparse.Namespace) -> int:
    from .training.grpo import GRPOConfig, GRPOTrainer

    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    cfg = GRPOConfig(
        model_name_or_path=args.model,
        output_dir=args.out,
        learning_rate=args.lr,
        group_size=args.group_size,
        tasks_per_batch=args.tasks_per_batch,
        n_epochs=args.epochs,
        kl_coef=args.kl_coef,
        max_turns=args.max_turns,
        use_lora=args.lora,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        bf16=not args.no_bf16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    GRPOTrainer(cfg).train(tasks)
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deepprep",
        description="DeepPrep: agentic autonomous data preparation (arXiv:2602.07371)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("operators", help="print the operator manual (Sec 2.2)")
    q.set_defaults(func=cmd_operators)

    q = sub.add_parser("demo", help="replay the paper's Figure-4 trajectory offline")
    q.add_argument("--verbose", action="store_true")
    q.set_defaults(func=cmd_demo)

    q = sub.add_parser("run", help="solve one ADP task")
    q.add_argument("task", help="path to a task JSON")
    q.add_argument("--out", default=None)
    q.add_argument("--verbose", action="store_true")
    _add_llm_args(q)
    q.set_defaults(func=cmd_run)

    q = sub.add_parser("eval", help="evaluate on a task set (Acc. / Comp. / Cost)")
    q.add_argument("tasks", help="path to a tasks JSONL")
    q.add_argument("--out", default=None)
    q.add_argument("--limit", type=int, default=None)
    q.add_argument("--workers", type=int, default=4)
    q.add_argument("--cost-mode", default="auto", choices=("auto", "api", "gpu"))
    _add_llm_args(q)
    q.set_defaults(func=cmd_eval)

    q = sub.add_parser("synthesize", help="build ADP tasks from an NL2SQL benchmark (Sec 5.3)")
    q.add_argument("--db-root", required=True, help="directory of the benchmark's sqlite databases")
    q.add_argument("--spec", required=True, help="benchmark JSON/JSONL of {db_id, question, query}")
    q.add_argument("--out", required=True)
    q.add_argument("--limit", type=int, default=None)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--dataset", default="Synth-Spider", help="name recorded in task metadata")
    q.add_argument("--split", default="train", choices=("train", "dev", "test"))
    q.add_argument("--use-llm", action="store_true", help="use an LLM where the paper does")
    _add_llm_args(q)
    q.set_defaults(func=cmd_synthesize)

    q = sub.add_parser("build-op-syntax", help="Stage 1 dataset D_op (Eq. 4)")
    q.add_argument("--tasks", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--max-span", type=int, default=3)
    q.add_argument("--max-per-task", type=int, default=6)
    q.add_argument("--seed", type=int, default=0)
    q.set_defaults(func=cmd_build_op_syntax)

    q = sub.add_parser("distill", help="Stage 2 teacher trajectories (Eq. 5)")
    q.add_argument("--tasks", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--teacher", default="deepseek-reasoner")
    q.add_argument("--judge", default=None, help="model id for the R_llm judge; omit for heuristic")
    q.add_argument("--n-candidates", type=int, default=4)
    q.add_argument("--top-k", type=int, default=1)
    q.add_argument("--limit", type=int, default=None)
    _add_llm_args(q)
    q.set_defaults(func=cmd_distill, temperature=0.7)

    q = sub.add_parser("train-sft", help="run cold-start SFT on a Stage-1 or Stage-2 dataset")
    q.add_argument("--data", required=True)
    q.add_argument("--model", required=True)
    q.add_argument("--out", default="checkpoints/sft")
    q.add_argument("--lr", type=float, default=6e-5)
    q.add_argument("--epochs", type=int, default=2)
    q.add_argument("--batch-size", type=int, default=1)
    q.add_argument("--grad-accum", type=int, default=8)
    q.add_argument("--max-length", type=int, default=8192)
    q.add_argument("--lora", action="store_true")
    q.set_defaults(func=cmd_train_sft)

    q = sub.add_parser("train-grpo", help="multi-turn GRPO with the hybrid reward (Sec 5.2)")
    q.add_argument("--tasks", required=True)
    q.add_argument("--model", required=True)
    q.add_argument("--out", default="checkpoints/grpo")
    q.add_argument("--lr", type=float, default=1e-6)
    q.add_argument("--group-size", type=int, default=8)
    q.add_argument("--tasks-per-batch", type=int, default=4)
    q.add_argument("--epochs", type=int, default=1)
    q.add_argument("--kl-coef", type=float, default=0.04)
    q.add_argument("--max-turns", type=int, default=5)
    q.add_argument("--limit", type=int, default=None)
    q.add_argument("--lora", action="store_true")
    # The defaults above are the paper's and assume a GPU; these make a run fit
    # on smaller hardware without editing the config in code.
    q.add_argument("--max-new-tokens", type=int, default=2048,
                   help="generation budget per turn during rollout")
    q.add_argument("--max-length", type=int, default=8192)
    q.add_argument("--no-bf16", action="store_true",
                   help="train in fp32 (bf16 matmuls are very slow on CPU)")
    q.add_argument("--no-gradient-checkpointing", action="store_true")
    q.set_defaults(func=cmd_train_grpo)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
