# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements prompt templates for CodeEvolve.
#
# ===--------------------------------------------------------------------------------------===#

from typing import Optional

# ---------------------------------------------------------------------------
# Reusable template sections
# ---------------------------------------------------------------------------

_MODIFICATION_FORMAT = """
## MODIFICATION FORMAT:
Present your proposed code changes using the following structure:
    ```
    <<<<<<< SEARCH
    [exact original code STRICTLY WITHIN an EVOLVE-BLOCK]
    =======
    [your modified code]
    >>>>>>> REPLACE
    ```
* For multiple independent changes, provide each in a separate SEARCH/REPLACE block.
"""

_CORE_RULES_SCOPE = """
### Scope & Boundaries:
    1. **Target `EVOLVE-BLOCK` ONLY**: All code modifications **MUST** be confined to sections explicitly marked between `{start_marker}` and `{end_marker}` comments. Do NOT include these markers in your modifications.
    2. **External Code Usage**: You **MAY reference** code outside these `EVOLVE-BLOCK` regions, but you **MUST NOT modify** it.
    3. **New Imports**: If new imports are required, add them *within* an `EVOLVE-BLOCK`.
"""

_CORE_RULES_SEARCH = """
### SEARCH Block Requirements:
    1. **EXACT Match**: The content of each `<<<<<<< SEARCH` block **MUST EXACTLY MATCH** the original code, including all whitespace, indentation, formatting, and comments.
    2. **No Comment Alterations in SEARCH**: Do **NOT** add, remove, or modify comments within the `<<<<<<< SEARCH` block. Only make comment changes in the `======= REPLACE` block.
    3. **First Occurrence Precedence**: If multiple identical code sections exist in the original program, your SEARCH block will be applied to the *first occurrence* matching its content.
"""

_CORE_RULES_OUTPUT = """
### Output & Compatibility:
    1. **Preserve Functionality**: Your modifications **MUST NOT** break existing functionality, external dependencies, or expected program behavior.
    2. **Maintain Compatibility**: All changes **MUST** maintain compatibility with unmarked code and preserve existing function signatures and interfaces.
    3. **Internal Consistency**: If you propose multiple changes across different SEARCH/REPLACE blocks, ensure they are mutually consistent (e.g., if a new variable or function is introduced in one block, define it in another if necessary).
"""

_CORE_RULES = f"""
## CORE RULES FOR CODE MODIFICATION:
{_CORE_RULES_SCOPE}
{_CORE_RULES_SEARCH}
{_CORE_RULES_OUTPUT}
"""

_INSPIRATION_ANALYSIS = """
## INSPIRATION PROGRAMS ANALYSIS:
You WILL be provided with multiple inspiration programs that demonstrate various approaches to solving similar problems. **MANDATORY** analysis requirements:

### Learning from Inspirations:
    1. **Extract Promising Techniques**: Identify and adapt successful algorithms, data structures, optimization strategies, and design patterns from the inspiration programs.
    2. **Avoid Known Pitfalls**: Recognize and avoid bugs, inefficiencies, poor practices, or design flaws present in the inspiration programs.
    3. **Synthesize Best Practices**: Combine the most effective elements from multiple inspiration programs while avoiding their weaknesses.
    4. **Performance Insights**: Learn from the performance characteristics and metrics of inspiration programs to guide your optimization decisions.

### Inspiration Analysis Process:
    1. **Before Modification**: Analyze each inspiration program to identify:
        - Algorithmic approaches and their complexity
        - Effective optimization techniques
        - Common bugs or inefficiencies to avoid
        - Useful design patterns or code structures
    2. **Integration Strategy**: Explain how you will incorporate promising ideas from inspiration programs while avoiding their mistakes.
    3. **Comparative Reasoning**: Justify your choices by comparing different approaches seen in the inspiration programs.
"""

_EXPLORATION_GOALS = """
### Exploration Goals:
    1. **Distinct Pathways**: Do not simply optimize the current approach. Attempt to solve the problem using a fundamentally different algorithm or logic.
"""

_EXPLORATION_INSPIRATION_ANALYSIS = """
## INSPIRATION EXPLORATION ANALYSIS:
You **MUST** analyze the provided **Random Inspiration Programs** to find alternative logic.

### Exploration Goals:
    1. **Seek Novelty**: Do not perform incremental optimization (e.g., small variable name changes). Look for **structural changes** in the inspirations.
    2. **Semantic Crossover**: Synthesize a new solution that combines the problem definition of the Target with the **algorithmic logic** of the Inspirations.
    3. **Divergent Thinking**: If the Target and Inspirations are similar, try to combine their distinct features to create a hybrid that differs from both parents.

### Mandatory Analysis Steps:
    1. **Logic Extraction**: Identify the core algorithm used in each random inspiration.
    2. **Differentiation Strategy**: Explain how the inspiration's approach differs from the target's current approach.
    3. **Synthesis Plan**: Describe how you will replace the target's logic with the inspiration's logic to explore a new part of the search space.
"""

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

_EVOLVE_EXAMPLE = """
## EXAMPLE:
### YOUR INPUT
    IMPROVE THE TARGET PROGRAM.
    ----------TARGET PROGRAM---------
    ```python
    {start_marker}
    def exp(a: int, b: int) -> int:
        x: int = 1
        for i in range(b + 1):
            x = x * a
        return x
    {end_marker}
    if __name__ == '__main__':
        print(exp(5, 3))
    ```
    PERFORMANCE METRICS: {{'runtime':1}}
    RETURNCODE: 0
    WARNING: None
    ERROR: None

### YOUR OUTPUT
    <<<<<<< SEARCH
    def exp(a: int, b: int) -> int:
        x: int = 1
        for i in range(b + 1):
            x = x * a
        return x
    =======
    def exp(a: int, b: int) -> int:
        if b == 0:
            return 1
        if b == 1:
            return a
        
        # Use iterative binary exponentiation for O(log b) time, O(1) space
        result = 1
        base = a
        exponent = b
        while exponent > 0:
            if exponent % 2 == 1:
                result *= base
            base *= base
            exponent //= 2
        return result
    >>>>>>> REPLACE
"""

_EVOLVE_WITH_INSPIRATIONS_EXAMPLE = """
## EXAMPLE:
### YOUR INPUT
    ----------INSPIRATION PROGRAM 1---------
    ```python
    {start_marker}
    def exp(a: int, b: int) -> int:
        if b == 0:
            return 1
        return a * exp(a, b - 1)  # Simple recursion - clean but O(n) stack depth
    {end_marker}
    ```
    PERFORMANCE METRICS: {{'runtime': 0.8}}
    ----------INSPIRATION PROGRAM 2---------
    ```python
    {start_marker}  
    def exp(a: int, b: int) -> int:
        result = 1
        base = a
        exponent = b
        while exponent > 0:
            if exponent % 2 == 1:
                result *= base
            base *= base
            exponent //= 2  # Binary exponentiation - O(log n) but iterative
        return result
    {end_marker}
    ```
    PERFORMANCE METRICS: {{'runtime': 0.3}}
    ---------------------------------
    IMPROVE THE TARGET PROGRAM.
    ----------TARGET PROGRAM---------
    ```python
    {start_marker}
    def exp(a: int, b: int) -> int:
        x: int = 1
        for i in range(b + 1):
            x = x * a
        return x
    {end_marker}
    if __name__ == '__main__':
        print(exp(5, 3))
    ```
    PERFORMANCE METRICS: {{'runtime':1}}
    RETURNCODE: 0
    WARNING: None
    ERROR: None

### YOUR OUTPUT
    **INSPIRATION ANALYSIS:**
    - INSPIRATION 1: Shows proper base case handling (b==0) but uses simple recursion with O(n) time and O(n) stack space. Risk of stack overflow for large b.
    - INSPIRATION 2: Implements efficient binary exponentiation with O(log b) time and O(1) space. However, the variable naming could be clearer and edge cases aren't explicitly handled.
    
    **INTEGRATION STRATEGY:**
    I will combine the clear edge case handling from INSPIRATION 1 with the efficient binary exponentiation algorithm from INSPIRATION 2, while also fixing the off-by-one bug in the target program.

    <<<<<<< SEARCH
    def exp(a: int, b: int) -> int:
        x: int = 1
        for i in range(b + 1):
            x = x * a
        return x
    =======
    def exp(a: int, b: int) -> int:
        # Handle edge cases (inspired by INSPIRATION 1's approach)
        if b == 0:
            return 1
        if b == 1:
            return a
        
        # Use iterative binary exponentiation for O(log b) time, O(1) space
        # (adapted from INSPIRATION 2's algorithm with clearer variable names)
        result = 1
        base = a
        exponent = b
        while exponent > 0:
            if exponent % 2 == 1:
                result *= base
            base *= base
            exponent //= 2
        return result
    >>>>>>> REPLACE
"""

_EXPLORE_EXAMPLE = """
## EXAMPLE:
### YOUR INPUT
    EXPLORE NEW STRATEGIES FOR THE TARGET PROGRAM.
    ----------TARGET PROGRAM---------
    ```python
    {start_marker}
    def exp(a: int, b: int) -> int:
        x: int = 1
        for i in range(b + 1):
            x = x * a
        return x
    {end_marker}
    if __name__ == '__main__':
        print(exp(5, 3))
    ```
    PERFORMANCE METRICS: {{'runtime':1}}
    RETURNCODE: 0
    WARNING: None
    ERROR: None

### YOUR OUTPUT
    <<<<<<< SEARCH
    def exp(a: int, b: int) -> int:
        x: int = 1
        for i in range(b + 1):
            x = x * a
        return x
    =======
    def exp(a: int, b: int) -> int:
        if b == 0:
            return 1
        if b == 1:
            return a
        
        # EXPLORATION: Switching from linear iteration to Divide-and-Conquer (Binary Exponentiation)
        # This represents a distinct algorithmic pathway with O(log n) complexity.
        result = 1
        base = a
        exponent = b
        while exponent > 0:
            if exponent % 2 == 1:
                result *= base
            base *= base
            exponent //= 2
        return result
    >>>>>>> REPLACE
"""

_EXPLORE_WITH_INSPIRATIONS_EXAMPLE = """
## EXAMPLE:
### YOUR INPUT
    ----------RANDOM INSPIRATION 1---------
    ```python
    {start_marker}
    def solve(data): return sorted(data, reverse=True)[0] # Greedy/Sorting approach
    {end_marker}
    ```
    ----------TARGET PROGRAM---------
    ```python
    {start_marker}
    def solve(data):
        m = 0
        for x in data: m = max(m, x) # Iterative Linear Scan
        return m
    {end_marker}
    ```

### YOUR OUTPUT
    **INSPIRATION ANALYSIS:**
    - The Target uses an iterative linear scan (O(N)).
    - The Random Inspiration uses a sorting-based approach (O(N log N)).
    - **Strategy**: While sorting is slower here, it represents a distinct algorithmic pathway. I will explore the sorting-based logic to see if it simplifies downstream operations or enables different optimizations.
    
    **SYNTHESIS:**
    I will replace the iterative logic with the sorting logic from Inspiration 1 to shift the solution structure.

    <<<<<<< SEARCH
    def solve(data):
        m = 0
        for x in data: m = max(m, x) # Iterative Linear Scan
        return m
    =======
    def solve(data):
        # EXPLORATION: Adopting Sorting strategy from Inspiration 1
        if not data: return 0
        sorted_data = sorted(data, reverse=True)
        return sorted_data[0]
    >>>>>>> REPLACE
"""


# ---------------------------------------------------------------------------
# Template factory functions
# ---------------------------------------------------------------------------


def get_evolve_task_template(start_marker: str, end_marker: str) -> str:
    """Returns the evolution task template with configured markers.

    Args:
        start_marker: The marker indicating the start of an evolve block.
        end_marker: The marker indicating the end of an evolve block.

    Returns:
        Formatted evolution task template string.
    """
    core_rules = _CORE_RULES.format(start_marker=start_marker, end_marker=end_marker)
    example = _EVOLVE_EXAMPLE.format(start_marker=start_marker, end_marker=end_marker)

    return f"""
# TASK: CODE EVOLUTION
Your goal is to evolve the provided program by modifying specific sections.
You **MUST** adhere strictly to the **SEARCH/REPLACE format** described below for all modifications.
{_MODIFICATION_FORMAT}
{core_rules}
{example}
"""


def get_evolve_with_inspirations_task_template(start_marker: str, end_marker: str) -> str:
    """Returns the evolution task template with inspirations and configured markers.

    Args:
        start_marker: The marker indicating the start of an evolve block.
        end_marker: The marker indicating the end of an evolve block.

    Returns:
        Formatted evolution with inspirations task template string.
    """
    core_rules = _CORE_RULES.format(start_marker=start_marker, end_marker=end_marker)
    example = _EVOLVE_WITH_INSPIRATIONS_EXAMPLE.format(
        start_marker=start_marker, end_marker=end_marker
    )

    return f"""
# TASK: CODE EVOLUTION
Your goal is to evolve the provided program by modifying specific sections.
You **MUST** adhere strictly to the **SEARCH/REPLACE format** described below for all modifications.
{_MODIFICATION_FORMAT}
{core_rules}
{_INSPIRATION_ANALYSIS}
{example}
"""


def get_explore_task_template(start_marker: str, end_marker: str) -> str:
    """Returns the exploration task template with configured markers.

    Args:
        start_marker: The marker indicating the start of an evolve block.
        end_marker: The marker indicating the end of an evolve block.

    Returns:
        Formatted exploration task template string.
    """
    core_rules_scope = _CORE_RULES_SCOPE.format(start_marker=start_marker, end_marker=end_marker)
    example = _EXPLORE_EXAMPLE.format(start_marker=start_marker, end_marker=end_marker)

    return f"""
# TASK: CODE EXPLORATION & DIVERSIFICATION
Your goal is to evolve the provided program by implementing **novel strategies** and **distinct algorithmic pathways**.
Unlike standard optimization, you should avoid minor incremental fixes. Instead, aim to rewrite the logic using a different paradigm or mathematical approach to increase the diversity of the solution space.
You **MUST** adhere strictly to the **SEARCH/REPLACE format** described below for all modifications.
{_MODIFICATION_FORMAT}
## CORE RULES FOR CODE MODIFICATION:
{core_rules_scope}
{_CORE_RULES_SEARCH}

### Output & Compatibility:
    1. **Preserve Functionality**: Your modifications **MUST NOT** break existing functionality, external dependencies, or expected program behavior.
    2. **Maintain Compatibility**: All changes **MUST** maintain compatibility with unmarked code and preserve existing function signatures and interfaces.
    3. **Internal Consistency**: If you propose multiple changes across different SEARCH/REPLACE blocks, ensure they are mutually consistent.
{_EXPLORATION_GOALS}
{example}
"""


def get_explore_with_inspirations_task_template(start_marker: str, end_marker: str) -> str:
    """Returns the exploration task template with inspirations and configured markers.

    Args:
        start_marker: The marker indicating the start of an evolve block.
        end_marker: The marker indicating the end of an evolve block.

    Returns:
        Formatted exploration with inspirations task template string.
    """
    core_rules_scope = _CORE_RULES_SCOPE.format(start_marker=start_marker, end_marker=end_marker)
    example = _EXPLORE_WITH_INSPIRATIONS_EXAMPLE.format(
        start_marker=start_marker, end_marker=end_marker
    )

    return f"""
# TASK: CODE EXPLORATION & DIVERSIFICATION
Your goal is to evolve the provided program by implementing **novel strategies** and **distinct algorithmic pathways**.
Unlike standard optimization, you should avoid minor incremental fixes. Instead, aim to rewrite the logic using a different paradigm or mathematical approach to increase the diversity of the solution space.
You will be provided with the **Target Program** and a set of **Randomly Sampled Inspiration Programs**.
Unlike optimization tasks, your goal is **NOT** merely to refine the current code, but to synthesize a fundamentally different approach (a semantic crossover) that increases solution diversity.

You **MUST** adhere strictly to the **SEARCH/REPLACE format** described below for all modifications.
{_MODIFICATION_FORMAT}
## CORE RULES FOR CODE MODIFICATION:
{core_rules_scope}
{_CORE_RULES_SEARCH}

### Output & Compatibility:
    1. **Preserve Functionality**: Your modifications **MUST NOT** break existing functionality or external dependencies.
    2. **Maintain Compatibility**: All changes **MUST** maintain compatibility with unmarked code.
{_EXPLORATION_INSPIRATION_ANALYSIS}
{example}
"""


def get_evolve_prompt_task_template(start_marker: str, end_marker: str) -> str:
    """Returns the prompt evolution task template with configured markers.

    Args:
        start_marker: The marker indicating the start of a prompt block.
        end_marker: The marker indicating the end of a prompt block.

    Returns:
        Formatted prompt evolution task template string.
    """
    return f"""
# SETTING
You are an expert Prompt Engineer specializing in crafting instructions for advanced code-generating AI models.

# TASK: PROMPT EVOLUTION FOR DIVERSITY
Your goal is to evolve the provided **prompt** to foster solution diversity.
While the evolved prompt must still aim for valid code, it should encourage the AI assistant to explore **distinct algorithmic pathways** and **novel strategies** different from the current solution.
You will be given the original prompt, the code it generated, and the results of executing that code.
You **MUST** adhere strictly to the **SEARCH/REPLACE format** described below for all modifications.

## MODIFICATION FORMAT:
Present your proposed prompt changes using the following structure:
```
<<<<<<< SEARCH
[exact original text within a PROMPT-BLOCK]
=======
[your modified text]
>>>>>>> REPLACE
```
* For multiple independent changes, provide each in a separate SEARCH/REPLACE block.

## CORE RULES FOR PROMPT MODIFICATION:
### Scope & Boundaries:
    1. **Target `PROMPT-BLOCK` ONLY**: All modifications **MUST** be confined to sections of the prompt explicitly marked between `{start_marker}` and `{end_marker}` comments.
    2. **External Text Usage**: You **MAY reference** text outside these `PROMPT-BLOCK` regions, but you **MUST NOT modify** it.

### SEARCH Block Requirements:
    1. **EXACT Match**: The content of each `<<<<<<< SEARCH` block **MUST EXACTLY MATCH** the original text.

### Goal of Evolution:
    1. **Foster Diversity**: Analyze the strategy used in the `GENERATED CODE`. Modify the prompt to guide the LLM away from this specific implementation detail and toward alternative valid logic or mathematical approaches.
    2. **Enrich Context**: Enrich the prompt with higher-level conceptual guidance that opens up the search space. Add insights from literature or broad algorithmic patterns that differ from the current approach.
    3. **Avoid Over-fitting**: Do not make the prompt overly specific to fixing the current solution's bugs if it sacrifices generality. The goal is a *new* perspective, not just a patch.

## EXAMPLE:
### YOUR INPUT
    ... [Input omitted for brevity] ...
    ----------GENERATED PROGRAM---------
    [An O(n) iterative solution]
    ---------------------------------

### YOUR ANSWER
    The generated program uses a standard iterative approach. To foster diversity and find potentially better optima, I will evolve the prompt to explicitly encourage a divide-and-conquer strategy, which is a distinct algorithmic pathway.

    <<<<<<< SEARCH
    # SETTING
    You are an expert software developer. Your goal is to design an integer exponentiation function.
    =======
    # SETTING
    You are an expert mathematician. Your goal is to design an integer exponentiation function using recursive properties or binary decomposition.
    >>>>>>> REPLACE
"""


# ---------------------------------------------------------------------------
# Evaluation budget template
# ---------------------------------------------------------------------------

_EVAL_BUDGET_TEMPLATE = """
## COMPUTATIONAL BUDGET:
- **Time limit**: {timeout_s} seconds maximum execution time{mem_line}
"""


def format_eval_budget(timeout_s: int, max_mem_b: Optional[int] = None) -> str:
    """Formats the evaluation budget into a string for injection into the system prompt.

    This ensures the LLM is always informed of the actual resource limits
    that will be enforced during evaluation, keeping the prompt in sync
    with the evaluator configuration.

    Args:
        timeout_s: Maximum execution time in seconds.
        max_mem_b: Maximum memory usage in bytes. If None, the memory line
            is omitted from the output.

    Returns:
        A formatted budget string ready to be inserted into the system prompt.
    """
    mem_line = ""
    if max_mem_b is not None:
        if max_mem_b >= 1024**3:
            mem_display = f"{max_mem_b / (1024**3):.1f} GB"
        elif max_mem_b >= 1024**2:
            mem_display = f"{max_mem_b / (1024**2):.1f} MB"
        else:
            mem_display = f"{max_mem_b} bytes"
        mem_line = f"\n- **Memory limit**: {mem_display}"

    return _EVAL_BUDGET_TEMPLATE.format(timeout_s=timeout_s, mem_line=mem_line)


# ---------------------------------------------------------------------------
# Simple templates
# ---------------------------------------------------------------------------

PROG_TEMPLATE = """```{language}
{code}
```
PERFORMANCE METRICS: {eval_metrics}
RETURNCODE: {returncode}
WARNING: {warning}
ERROR: {error}
"""

EVOLVE_PROG_TEMPLATE = """IMPROVE THE TARGET PROGRAM.
----------TARGET PROGRAM---------
{program}
---------------------------------
"""

INSP_PROG_TEMPLATE = """-------INSPIRATION PROGRAM {counter}-------
{program}
---------------------------------
"""

EVOLVE_PROMPT_TEMPLATE = """IMPROVE THE TARGET PROMPT.
----------TARGET PROMPT---------
{prompt}
--------------------------------
----------GENERATED PROGRAM---------
{program}
------------------------------------
"""
