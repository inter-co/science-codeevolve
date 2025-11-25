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
Your task is to improve a given prompt in order to guide an LLM to generate more effective and efficient code.

# TASK: PROMPT EVOLUTION
Your goal is to evolve the provided **prompt**. 
The evolved prompt should be more likely to guide an AI assistant to generate correct and effective code. 
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
    1.1. **Target `PROMPT-BLOCK` ONLY**: All modifications **MUST** be confined to sections of the prompt explicitly marked between `PROMPT-BLOCK-START` and `PROMPT-BLOCK-END` comments. Do NOT include these markers in your modifications.
    1.2. **External Text Usage**: You **MAY reference** text outside these `PROMPT-BLOCK` regions, but you **MUST NOT modify** it.

### 2. SEARCH Block Requirements:
    2.1. **EXACT Match**: The content of each `<<<<<<< SEARCH` block **MUST EXACTLY MATCH** the original text, including all whitespace, formatting, and punctuation.
    2.2. **No Comment Alterations in SEARCH**: Do **NOT** add, remove, or modify comments within the `<<<<<<< SEARCH` block.

### 3. Goal of Evolution:
    3.1. **Incorporate Feedback**: Your modifications should incorporate insights from the provided `GENERATED CODE` and its `PERFORMANCE METRICS`. Use the errors, warnings, or performance issues as clues for how the prompt can be improved.
    3.2. **Enrich context**: Your modifications should enrich the prompt's context in order to give the LLM a broader and deeper understanding of the problem at hand. Add relevant information about the problem that will enrich the context, insights into the literature and state-of-the-art knowledge, guidance toward specific algorithmic patterns and strategies, etc. You can also remove redundant or misleading parts of the prompt in order to increase its quality.

## EXAMPLE:
### YOUR INPUT
    IMPROVE THE TARGET PROMPT.
    ----------TARGET PROMPT---------
    # PROMPT-BLOCK-START
    # SETTING
    You are an expert software developer. Your goal is to design an integer exponentiation function.
    # PROMPT-BLOCK-END
    # PERFORMANCE METRICS
        1. **correctness**: Ratio of tests passed by total number of tests
        2. **eval_time**: Execution time in seconds
    # COMPUTATIONAL RESOURCES
    You have access to the python standard library.
    You have a time limit of 1 second, and a memory limit of 100 KB.
    # REQUIREMENTS
    You function must be correct.
    # PROMPT-BLOCK-START
    # ADDITIONAL CONTEXT
    Exponentiation is defined as exp(a,b) = a*a*...*a (b times) when a and b are integers. 
    # PROMPT-BLOCK-END
    --------------------------------
    ----------GENERATED PROGRAM---------
    ```python
    def exp(a: int, b: int) -> int:
        x: int = 1
        for i in range(b):
            x = x * a
        return x
    ```
    PERFORMANCE METRICS: {'correctness':1,'eval_time':0.99}
    RETURNCODE: 0
    WARNING: None
    ERROR: None
    ---------------------------------
### YOUR ANSWER
    The `eval_time` of 0.99 is dangerously close to the 1-second limit, suggesting the O(b) iterative approach is too slow.
    The original prompt's description, `a*a*...*a (b times)`, likely guided the AI to this inefficient implementation. 
    I will evolve the prompt to be more specific about performance requirements and to provide clearer examples to prevent ambiguity. 
    This will guide the AI towards a more robust and efficient algorithm like binary exponentiation, which has O(log b) time complexity.
    <<<<<<< SEARCH
    # SETTING
    You are an expert software developer. Your goal is to design an integer exponentiation function.
    =======
    # SETTING
    You are an expert software engineer, with a deep knowledge of divide-and-conquer algorithms.
    Your goal is to design an efficient function to perform integer exponentiation.
    >>>>>>> REPLACE
    <<<<<<< SEARCH
    # ADDITIONAL CONTEXT
    Exponentiation is defined as exp(a,b) = a*a*...*a (b times) when a and b are integers. 
    =======
    # ADDITIONAL CONTEXT
    Exponentiation is defined as `exp(a,b) = a*a*...*a` (b times). For example, `exp(a, 3) = a * a * a`.
    A naive solution can be implemented in linear time O(b), however this is likely to time out for large values of b.
    The integer exponentiation problem is a famous example where a divide-and-conquer strategy can achieve an exponential speedup: O(log b) time.
    This is achieved by computing a^{b/2} recursively.
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
