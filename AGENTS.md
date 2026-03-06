# AGENTS.md

Guide for AI agents working on the CodeEvolve codebase.

## Project Overview

CodeEvolve is an LLM-driven evolutionary algorithm framework that discovers and optimizes code solutions. Programs evolve across distributed islands via LLM-generated SEARCH/REPLACE diffs, evaluated in sandboxed environments.

## Repository Layout

```
src/codeevolve/
  cli.py            # CLI entry point, arg parsing, island spawning
  runner.py         # Process management, signal handling, log daemon
  evolution.py      # Main evolutionary loop and component setup
  database.py       # Program dataclass, ProgramDatabase, MAP-Elites
  evaluator.py      # Sandboxed execution with time/memory limits
  scheduler.py      # Exploration rate schedulers
  lm/               # LLM interfaces (base.py, openai.py)
  islands/          # Topology (graph.py), sync (sync.py), migration (migration.py)
  prompt/           # Prompt templates (template.py), conversation builder (sampler.py)
  utils/            # Constants, checkpointing, config, diff parsing, logging, locking
tests/              # pytest suite mirroring src/ structure
configs/            # Example YAML configs (mock, qwen, gemini)
problems/           # Benchmark problems and problem_template
```

## Build & Test

- **Python**: >=3.13.5, managed via conda (`environment.yml`)
- **Install**: `conda env create -f environment.yml && conda activate codeevolve`
- **Run tests**: `pytest tests/` (use `pytest tests/ -v` for verbose)
- **Formatting**: `black` (line-length 100, target py313) and `isort` (profile "black")
- **Async tests**: use `@pytest.mark.asyncio` decorator; pytest-asyncio is configured in strict mode

## Code Conventions

- **Type hints**: use throughout; prefer `Optional[X]` over `X | None` for new code
- **Docstrings**: Google-style with Args/Returns/Raises sections
- **File headers**: each source file starts with the Apache-2.0 license block
- **Constants**: centralized in `utils/constants.py`; do not scatter magic values
- **Dataclasses**: `Program` is a dataclass in `database.py`; `depth` field tracks evolutionary lineage depth
- **Config**: YAML-based; top-level keys (`SYS_MSG`, `CODEBASE_PATH`, etc.), `BUDGET_CONFIG` dict for resource limits/timeout scheduling, and `EVOLVE_CONFIG` dict for evolutionary parameters (with nested `selection`, `migration`, `markers`, `exploration_scheduler` sub-dicts)

## Key Patterns

### Evolutionary Loop (`evolution.py`)

The loop in `codeevolve_loop()` delegates to focused helper functions:
1. `select_parents()` — selection phase
2. `run_meta_prompting()` — optional prompt evolution
3. `generate_solution()` — LLM-driven code generation
4. `evaluate_and_store()` — sandboxed evaluation
5. `handle_migration()` — inter-island exchange

Component setup lives in `setup_codeevolve_components()` and `_create_*` factory functions.

### Prompt Assembly (`prompt/`)

System messages are assembled in `sampler.py:build()` as:
`[user SYS_MSG] + [eval_budget] + [task template]`

Task templates live in `template.py` as composable sections (`_CORE_RULES`, `_MODIFICATION_FORMAT`, etc.) assembled by `get_*_task_template()` factory functions.

### Evaluator (`evaluator.py`)

The `Evaluator` class runs programs via subprocess with timeout and optional memory monitoring. The `execute()` method takes a `Program` and returns `(returncode, output, warning, error, eval_metrics)`.

### Mock Models

Set `model_name` to `"MOCK"` in config to use `MockOpenAILM` for testing without API calls. The mock returns identity SEARCH/REPLACE diffs.

## Testing Guidelines

- Mirror the module structure: `test_<module>.py`
- Use class-based test organization (`class TestFeatureName`)
- Prefix helpers with `_make_*` for fixture-like factory methods
- Test both success and error/edge cases
- For async functions, use `@pytest.mark.asyncio`
