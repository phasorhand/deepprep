"""Data synthesis for training (paper Sec 5.3).

    "Since such ADP task instances are not available at scale, we construct them
     through synthesis by converting SQL benchmarks into ADP tasks consisting of
     source tables, a target table, and a corresponding transformation pipeline."

The module addresses the paper's two stated challenges directly:

* *"the generated pipelines should correspond to meaningful analytical
  transformations rather than arbitrary operator chains"* — pipelines are derived
  from real analytical SQL (:mod:`.pipeline_search`), never sampled at random, and
  are kept only when execution reproduces ``T*`` exactly.
* *"the synthesized source tables should include diverse data quality issues while
  preserving the intended transformation logic"* — corruption is the inverse of a
  cleaning operator and is accepted only when that operator restores the previous
  state (:mod:`.noise`).

Everything runs offline; an :class:`~deepprep.agent.llm.LLMClient` is optional and
is used exactly where the paper uses one (target schema specification, candidate
pipeline generation, inverse transformation logic).
"""

from __future__ import annotations

from .nl2sql import (
    CleanInstance,
    NL2SQLCase,
    build_clean_instance,
    infer_target_schema,
    load_benchmark,
)
from .noise import (
    BUILTIN_NOISE_PAIRS,
    Corruption,
    LLMInverseProposer,
    NoiseConfig,
    NoisePair,
    NoiseResult,
    inject_noise,
    state_signature,
    try_corruption,
)
from .pipeline_search import (
    PipelineSearchResult,
    TranslationError,
    TranslationResult,
    execute_pipeline,
    propose_llm_pipelines,
    search_pipeline,
    translate_sql,
    verify_pipeline,
)
from .synthesize import (
    SynthesisConfig,
    SynthesisStats,
    synthesize_dataset,
    synthesize_task,
    write_jsonl,
)

__all__ = [
    "BUILTIN_NOISE_PAIRS",
    "CleanInstance",
    "Corruption",
    "LLMInverseProposer",
    "NL2SQLCase",
    "NoiseConfig",
    "NoisePair",
    "NoiseResult",
    "PipelineSearchResult",
    "SynthesisConfig",
    "SynthesisStats",
    "TranslationError",
    "TranslationResult",
    "build_clean_instance",
    "execute_pipeline",
    "infer_target_schema",
    "inject_noise",
    "load_benchmark",
    "propose_llm_pipelines",
    "search_pipeline",
    "state_signature",
    "synthesize_dataset",
    "synthesize_task",
    "translate_sql",
    "try_corruption",
    "verify_pipeline",
    "write_jsonl",
]
