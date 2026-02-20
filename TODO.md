# CodeEvolve TODOs

## New Features

### Dynamic Timeout

This consists of allowing the user to set a maximum evaluation timeout in seconds, but to vary the actual timeout afforded to certain solutions. This could be, for instance, proportional to the depth of the solution in the program tree (shallow programs get less time, deeper programs get more time).

In terms of the code itself, this is somewhat straightforward. We'll need to create a prompt template that formats the evaluation budget (we should have this regardless), and then dynamically change it once we sample a new parent solution.

### Multiple file support

This feature has two independent steps: allowing the agent to read multiple input files when needed, and allowing the agent to modify multiple files. The former essentially boils down to implementing an agentic logic similar to what cursor or claude code does (allow agent to run bash scripts and see the output). This would require a significant refactor of the llm logic, and this would also be a challenge for smaller models like qwen to handle. The latter step is not that difficult: we could for instance ask the SEARCH/REPLACE blocks to also identify which file they are targeted at. This would however add further complexity to the instructions the LLM needs to do.

### Full rewrite

Allow the agent to do a full rewrite of the target file. This should be simple to implement.

## Systems

### Overhaul of the evaluation logic

Here's an overview of the current situation as potential issues: currently, each island has a separate process, and each process can call the evaluator function to run a given solution. Thus, if we have N islands, there may be N parallel processes competing for CPU/Mem resources to run the solution. The solution itself may want to use all CPUs, so this can get a bit messy and unfair, since our evaluator timeout measures wall-clock time. 

The easiest way of improving this I can see would be for the user to specify how many CPUs a given solution is allowed to use, and then internally launch the solution process with taskset, in order to guarantee an independence of resource access. The might be better ways of improving this, so we should think it carefully.

### Better SEARCH/REPLACE

Even with more expensive LLMs, we get a lot of SEARCH/REPLACE errors, i.e., LLM trying to search for a block of code that does not perfectly match the parent program. We should think of a way of minimizing these kinds of errors (something that happened quite often with GEMINI 2.5 was it trying to search for a code block that almost matched the parent program, apart from an hallucinated comment).

## Quality-of-life

### More templates for system messages

Instead of asking the users to specify the evaluation budgets, installed packages, etc, we should automatically format those and add them to the system message.

### Better config structure

All configs within the code are dicts. This can get quite confusing and hard to read. We should implement dataclasses with these configs and defaults. This is conceptually easy, but would require a major refactor of the code.

## Unit tests

We currently only have a really simple test suite for the SEARCH/REPLACE operator. We need to vastly increase this: from basic tests of our classes, to more complicated simulated runs with the MOCK LLM setting.