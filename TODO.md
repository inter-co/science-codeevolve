# CodeEvolve TODOs

## New Features

### Multiple file support

This feature has two independent steps: allowing the agent to read multiple input files when needed, and allowing the agent to modify multiple files. The former essentially boils down to implementing an agentic logic similar to what cursor or claude code does (allow agent to run bash scripts and see the output). This would require a significant refactor of the llm logic, and this would also be a challenge for smaller models like qwen to handle. The latter step is not that difficult: we could for instance ask the SEARCH/REPLACE blocks to also identify which file they are targeted at. This would however add further complexity to the instructions the LLM needs to do.

### Full rewrite

Allow the agent to do a full rewrite of the target file. This should be simple to implement.

## Systems

### Overhaul of the evaluation logic (Done)

Here's an overview of the current situation as potential issues: currently, each island has a separate process, and each process can call the evaluator function to run a given solution. Thus, if we have N islands, there may be N parallel processes competing for CPU/Mem resources to run the solution. The solution itself may want to use all CPUs, so this can get a bit messy and unfair, since our evaluator timeout measures wall-clock time.

**Fix 1 implemented:** The `resource_monitor` thread (previously `mem_monitor`) now also tracks accumulated CPU time (user + system) across the entire process tree via `psutil`. When the total CPU time exceeds `timeout_s`, the process is killed with a `CPUTimeExceededError`. This makes timeouts fairer under CPU contention — a program that is CPU-starved gets its full budget of actual computation rather than being killed early by a wall-clock timer. It also catches multi-threaded or multi-process evaluations that burn more CPU time than wall-clock time.

**Fix 2 implemented:** Users can now set `num_cpus_per_eval` in `BUDGET_CONFIG`. When set, CodeEvolve partitions the available CPUs (as reported by `os.sched_getaffinity`) into consecutive slices and pins each island process to its own slice via `os.sched_setaffinity` at startup. This eliminates cross-island CPU contention and makes wall-clock time roughly equal to CPU time, removing the need for a separate CPU-time budget. Requires Linux; falls back gracefully with a warning on other platforms or when insufficient CPUs are available.

### Better SEARCH/REPLACE

Even with more expensive LLMs, we get a lot of SEARCH/REPLACE errors, i.e., LLM trying to search for a block of code that does not perfectly match the parent program. We should think of a way of minimizing these kinds of errors (something that happened quite often with GEMINI 2.5 was it trying to search for a code block that almost matched the parent program, apart from an hallucinated comment).

## Quality-of-life

### More templates for system messages

Instead of asking the users to specify the evaluation budgets, installed packages, etc, we should automatically format those and add them to the system message.

### Better config structure

All configs within the code are dicts. This can get quite confusing and hard to read. We should implement dataclasses with these configs and defaults. This is conceptually easy, but would require a major refactor of the code.

### Better dependency handling for problems

Each benchmark problem should have its own dependencies, and they should be installed whenever the problem is first run. Currently, we bundle all dependencies into CodeEvolve itself, which isn't great.

## Unit tests

We currently only have a really simple test suite for the SEARCH/REPLACE operator. We need to vastly increase this: from basic tests of our classes, to more complicated simulated runs with the MOCK LLM setting.