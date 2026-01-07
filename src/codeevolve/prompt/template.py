# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements prompt templates.
#
# ===--------------------------------------------------------------------------------------===#

# task: evolve solution
EVOLVE_PROG_TASK_TEMPLATE = """
# TASK: CODE EVOLUTION
Your goal is to evolve the provided program by modifying specific sections.
You **MUST** adhere strictly to the **SEARCH/REPLACE format** described below for all modifications.

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

## CORE RULES FOR CODE MODIFICATION:
### 1. Scope & Boundaries:
    1.1. **Target `EVOLVE-BLOCK` ONLY**: All code modifications **MUST** be confined to sections explicitly marked between `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` comments. Do NOT include these markers in your modifications.
    1.2. **External Code Usage**: You **MAY reference** code outside these `EVOLVE-BLOCK` regions, but you **MUST NOT modify** it.
    1.3. **New Imports**: If new imports are required, add them *within* an `EVOLVE-BLOCK`.

### 2. SEARCH Block Requirements:
    2.1. **EXACT Match**: The content of each `<<<<<<< SEARCH` block **MUST EXACTLY MATCH** the original code, including all whitespace, indentation, formatting, and comments.
    2.2. **No Comment Alterations in SEARCH**: Do **NOT** add, remove, or modify comments within the `<<<<<<< SEARCH` block. Only make comment changes in the `======= REPLACE` block.
    2.3. **First Occurrence Precedence**: If multiple identical code sections exist in the original program, your SEARCH block will be applied to the *first occurrence* matching its content.

### 3. Output & Compatibility:
    3.1. **Preserve Functionality**: Your modifications **MUST NOT** break existing functionality, external dependencies, or expected program behavior.
    3.2. **Maintain Compatibility**: All changes **MUST** maintain compatibility with unmarked code and preserve existing function signatures and interfaces.
    3.3. **Internal Consistency**: If you propose multiple changes across different SEARCH/REPLACE blocks, ensure they are mutually consistent (e.g., if a new variable or function is introduced in one block, define it in another if necessary).

## EXAMPLE:
### YOUR INPUT
    IMPROVE THE TARGET PROGRAM.
    ----------TARGET PROGRAM---------
    ```python
    # EVOLVE-BLOCK-START
    def exp(a: int, b: int) -> int:
        x: int = 1
        for i in range(b + 1):
            x = x * a
        return x
    # EVOLVE-BLOCK-END
    if __name__ == '__main__':
        print(exp(5, 3))
    ```
    PERFORMANCE METRICS: {'runtime':1}
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

EVOLVE_PROG_WINSP_TASK_TEMPLATE = """
# TASK: CODE EVOLUTION
Your goal is to evolve the provided program by modifying specific sections.
You **MUST** adhere strictly to the **SEARCH/REPLACE format** described below for all modifications.

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

## CORE RULES FOR CODE MODIFICATION:
### 1. Scope & Boundaries:
    1.1. **Target `EVOLVE-BLOCK` ONLY**: All code modifications **MUST** be confined to sections explicitly marked between `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` comments. Do NOT include these markers in your modifications.
    1.2. **External Code Usage**: You **MAY reference** code outside these `EVOLVE-BLOCK` regions, but you **MUST NOT modify** it.
    1.3. **New Imports**: If new imports are required, add them *within* an `EVOLVE-BLOCK`.

### 2. SEARCH Block Requirements:
    2.1. **EXACT Match**: The content of each `<<<<<<< SEARCH` block **MUST EXACTLY MATCH** the original code, including all whitespace, indentation, formatting, and comments.
    2.2. **No Comment Alterations in SEARCH**: Do **NOT** add, remove, or modify comments within the `<<<<<<< SEARCH` block. Only make comment changes in the `======= REPLACE` block.
    2.3. **First Occurrence Precedence**: If multiple identical code sections exist in the original program, your SEARCH block will be applied to the *first occurrence* matching its content.

### 3. Output & Compatibility:
    3.1. **Preserve Functionality**: Your modifications **MUST NOT** break existing functionality, external dependencies, or expected program behavior.
    3.2. **Maintain Compatibility**: All changes **MUST** maintain compatibility with unmarked code and preserve existing function signatures and interfaces.
    3.3. **Internal Consistency**: If you propose multiple changes across different SEARCH/REPLACE blocks, ensure they are mutually consistent (e.g., if a new variable or function is introduced in one block, define it in another if necessary).

## INSPIRATION PROGRAMS ANALYSIS:
You WILL be provided with multiple inspiration programs that demonstrate various approaches to solving similar problems. **MANDATORY** analysis requirements:

### 4. Learning from Inspirations:
    4.1. **Extract Promising Techniques**: Identify and adapt successful algorithms, data structures, optimization strategies, and design patterns from the inspiration programs.
    4.2. **Avoid Known Pitfalls**: Recognize and avoid bugs, inefficiencies, poor practices, or design flaws present in the inspiration programs.
    4.3. **Synthesize Best Practices**: Combine the most effective elements from multiple inspiration programs while avoiding their weaknesses.
    4.4. **Performance Insights**: Learn from the performance characteristics and metrics of inspiration programs to guide your optimization decisions.

### 5. Inspiration Analysis Process:
    5.1. **Before Modification**: Analyze each inspiration program to identify:
        - Algorithmic approaches and their complexity
        - Effective optimization techniques
        - Common bugs or inefficiencies to avoid
        - Useful design patterns or code structures
    5.2. **Integration Strategy**: Explain how you will incorporate promising ideas from inspiration programs while avoiding their mistakes.
    5.3. **Comparative Reasoning**: Justify your choices by comparing different approaches seen in the inspiration programs.

## EXAMPLE:
### YOUR INPUT
    ----------INSPIRATION PROGRAM 1---------
    ```python
    # EVOLVE-BLOCK-START
    def exp(a: int, b: int) -> int:
        if b == 0:
            return 1
        return a * exp(a, b - 1)  # Simple recursion - clean but O(n) stack depth
    # EVOLVE-BLOCK-END
    ```
    PERFORMANCE METRICS: {'runtime': 0.8}
    ----------INSPIRATION PROGRAM 2---------
    ```python
    # EVOLVE-BLOCK-START  
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
    # EVOLVE-BLOCK-END
    ```
    PERFORMANCE METRICS: {'runtime': 0.3}
    ---------------------------------
    IMPROVE THE TARGET PROGRAM.
    ----------TARGET PROGRAM---------
    ```python
    # EVOLVE-BLOCK-START
    def exp(a: int, b: int) -> int:
        x: int = 1
        for i in range(b + 1):
            x = x * a
        return x
    # EVOLVE-BLOCK-END
    if __name__ == '__main__':
        print(exp(5, 3))
    ```
    PERFORMANCE METRICS: {'runtime':1}
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

EXPLORE_PROG_TASK_TEMPLATE = """
# TASK: CODE EXPLORATION & DIVERSIFICATION
Your goal is to evolve the provided program by implementing **novel strategies** and **distinct algorithmic pathways**.
Unlike standard optimization, you should avoid minor incremental fixes. Instead, aim to rewrite the logic using a different paradigm or mathematical approach to increase the diversity of the solution space.
You **MUST** adhere strictly to the **SEARCH/REPLACE format** described below for all modifications.
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
## CORE RULES FOR CODE MODIFICATION:
### 1. Scope & Boundaries:
    1.1. **Target `EVOLVE-BLOCK` ONLY**: All code modifications **MUST** be confined to sections explicitly marked between `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` comments. Do NOT include these markers in your modifications.
    1.2. **External Code Usage**: You **MAY reference** code outside these `EVOLVE-BLOCK` regions, but you **MUST NOT modify** it.
    1.3. **New Imports**: If new imports are required, add them *within* an `EVOLVE-BLOCK`.
### 2. SEARCH Block Requirements:
    2.1. **EXACT Match**: The content of each `<<<<<<< SEARCH` block **MUST EXACTLY MATCH** the original code, including all whitespace, indentation, formatting, and comments.
    2.2. **No Comment Alterations in SEARCH**: Do **NOT** add, remove, or modify comments within the `<<<<<<< SEARCH` block. Only make comment changes in the `======= REPLACE` block.
    2.3. **First Occurrence Precedence**: If multiple identical code sections exist in the original program, your SEARCH block will be applied to the *first occurrence* matching its content.
### 3. Output & Compatibility:
    3.1. **Preserve Functionality**: Your modifications **MUST NOT** break existing functionality, external dependencies, or expected program behavior.
    3.2. **Maintain Compatibility**: All changes **MUST** maintain compatibility with unmarked code and preserve existing function signatures and interfaces.
    3.3. **Internal Consistency**: If you propose multiple changes across different SEARCH/REPLACE blocks, ensure they are mutually consistent.
### 4. Exploration Goals:
    4.1. **Distinct Pathways**: Do not simply optimize the current approach. Attempt to solve the problem using a fundamentally different algorithm or logic.
## EXAMPLE:
### YOUR INPUT
    EXPLORE NEW STRATEGIES FOR THE TARGET PROGRAM.
    ----------TARGET PROGRAM---------
    ```python
    # EVOLVE-BLOCK-START
    def exp(a: int, b: int) -> int:
        x: int = 1
        for i in range(b + 1):
            x = x * a
        return x
    # EVOLVE-BLOCK-END
    if __name__ == '__main__':
        print(exp(5, 3))
    ```
    PERFORMANCE METRICS: {'runtime':1}
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

EXPLORE_PROG_WINSP_TASK_TEMPLATE = """
# TASK: CODE EXPLORATION & DIVERSIFICATION
Your goal is to evolve the provided program by implementing **novel strategies** and **distinct algorithmic pathways**.
Unlike standard optimization, you should avoid minor incremental fixes. Instead, aim to rewrite the logic using a different paradigm or mathematical approach to increase the diversity of the solution space.
You will be provided with the **Target Program** and a set of **Randomly Sampled Inspiration Programs**.
Unlike optimization tasks, your goal is **NOT** merely to refine the current code, but to synthesize a fundamentally different approach (a semantic crossover) that increases solution diversity.

You **MUST** adhere strictly to the **SEARCH/REPLACE format** described below for all modifications.

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

## CORE RULES FOR CODE MODIFICATION:
### 1. Scope & Boundaries:
    1.1. **Target `EVOLVE-BLOCK` ONLY**: All code modifications **MUST** be confined to sections explicitly marked between `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` comments.
    1.2. **External Code Usage**: You **MAY reference** code outside these `EVOLVE-BLOCK` regions, but you **MUST NOT modify** it.
    1.3. **New Imports**: If new imports are required, add them *within* an `EVOLVE-BLOCK`.

### 2. SEARCH Block Requirements:
    2.1. **EXACT Match**: The content of each `<<<<<<< SEARCH` block **MUST EXACTLY MATCH** the original code.
    2.2. **No Comment Alterations in SEARCH**: Do **NOT** add, remove, or modify comments within the `<<<<<<< SEARCH` block.

### 3. Output & Compatibility:
    3.1. **Preserve Functionality**: Your modifications **MUST NOT** break existing functionality or external dependencies.
    3.2. **Maintain Compatibility**: All changes **MUST** maintain compatibility with unmarked code.

## INSPIRATION EXPLORATION ANALYSIS:
You **MUST** analyze the provided **Random Inspiration Programs** to find alternative logic.

### 4. Exploration Goals:
    4.1. **Seek Novelty**: Do not perform incremental optimization (e.g., small variable name changes). Look for **structural changes** in the inspirations.
    4.2. **Semantic Crossover**: Synthesize a new solution that combines the problem definition of the Target with the **algorithmic logic** of the Inspirations.
    4.3. **Divergent Thinking**: If the Target and Inspirations are similar, try to combine their distinct features to create a hybrid that differs from both parents.

### 5. Mandatory Analysis Steps:
    5.1. **Logic Extraction**: Identify the core algorithm used in each random inspiration.
    5.2. **Differentiation Strategy**: Explain how the inspiration's approach differs from the target's current approach.
    5.3. **Synthesis Plan**: Describe how you will replace the target's logic with the inspiration's logic to explore a new part of the search space.

## EXAMPLE:
### YOUR INPUT
    ----------RANDOM INSPIRATION 1---------
    ```python
    # EVOLVE-BLOCK-START
    def solve(data): return sorted(data, reverse=True)[0] # Greedy/Sorting approach
    # EVOLVE-BLOCK-END
    ```
    ----------TARGET PROGRAM---------
    ```python
    # EVOLVE-BLOCK-START
    def solve(data):
        m = 0
        for x in data: m = max(m, x) # Iterative Linear Scan
        return m
    # EVOLVE-BLOCK-END
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

PROG_TEMPLATE = """ 
```{language}
{code}
```
PERFORMANCE METRICS: {eval_metrics}
RETURNCODE: {returncode}
WARNING: {warning}
ERROR: {error}
"""

EVOLVE_PROG_TEMPLATE = """ 
IMPROVE THE TARGET PROGRAM.
----------TARGET PROGRAM---------
{program}
---------------------------------
"""
INSP_PROG_TEMPLATE = """ 
-------INSPIRATION PROGRAM {counter}-------
{program}
---------------------------------
"""

# task: evolve prompt
EVOLVE_PROMPT_TASK_TEMPLATE = """
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
[exact original text within an PROMPT-BLOCK]
=======
[your modified text]
>>>>>>> REPLACE
```
* For multiple independent changes, provide each in a separate SEARCH/REPLACE block.

## CORE RULES FOR PROMPT MODIFICATION:
### 1. Scope & Boundaries:
    1.1. **Target `PROMPT-BLOCK` ONLY**: All modifications **MUST** be confined to sections of the prompt explicitly marked between `PROMPT-BLOCK-START` and `PROMPT-BLOCK-END` comments.
    1.2. **External Text Usage**: You **MAY reference** text outside these `PROMPT-BLOCK` regions, but you **MUST NOT modify** it.
### 2. SEARCH Block Requirements:
    2.1. **EXACT Match**: The content of each `<<<<<<< SEARCH` block **MUST EXACTLY MATCH** the original text.
### 3. Goal of Evolution:
    3.1. **Foster Diversity**: Analyze the strategy used in the `GENERATED CODE`. Modify the prompt to guide the LLM away from this specific implementation detail and toward alternative valid logic or mathematical approaches.
    3.2. **Enrich Context**: Enrich the prompt with higher-level conceptual guidance that opens up the search space. Add insights from literature or broad algorithmic patterns that differ from the current approach.
    3.3. **Avoid Over-fitting**: Do not make the prompt overly specific to fixing the current solution's bugs if it sacrifices generality. The goal is a *new* perspective, not just a patch.
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

EVOLVE_PROMPT_TEMPLATE = """ 
IMPROVE THE TARGET PROMPT.
----------TARGET PROMPT---------
{prompt}
--------------------------------
----------GENERATED PROGRAM---------
{program}
------------------------------------
"""
