# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the parsing functions for parsing language models responses to
# CodeEvolve's prompts.
#
# ===--------------------------------------------------------------------------------------===#

from typing import Dict, Tuple, List
import re


class SearchAndReplaceError(Exception):
    """Exception raised when a search pattern cannot be found in any evolve block.

    This exception is thrown when applying diff operations if a search pattern
    from a diff block cannot be located within any of the designated evolve blocks
    in the parent code.
    """

    pass


class DiffError(Exception):
    """Exception raised when no diff blocks are found in the diff string.

    This exception is thrown when parsing diff content if no valid diff blocks
    matching the expected format are discovered.
    """

    pass


class EvolveBlockError(Exception):
    """Exception raised when no evolve blocks are found in the parent code.

    This exception is thrown when searching for evolve blocks if no blocks
    matching the expected format are found in the parent code.
    """

    pass


def parse_diff_blocks(diff: str, diff_regex: str) -> List[Tuple[str, str]]:
    """Parses diff blocks from a diff string using regex pattern matching.

    This function extracts search-and-replace pairs from a diff string by finding
    all matches of the provided regex pattern and returning them as tuples.

    Args:
        diff: The diff string containing search and replace blocks.
        diff_regex: Regular expression pattern to match diff blocks. Should contain
                   two capture groups for search and replace text.

    Returns:
        List of tuples where each tuple contains (search_text, replace_text).

    Raises:
        DiffError: If no diff blocks are found matching the regex pattern.
    """
    diff_blocks: List[Tuple[str, str]] = re.findall(diff_regex, diff, re.DOTALL)

    if len(diff_blocks) == 0:
        raise DiffError("No DIFF blocks found.")

    return [(search.strip(), replace.strip()) for search, replace in diff_blocks]


def find_evolve_block_spans(parent_code: str, evolve_regex: str) -> List[Tuple[int, int]]:
    """Finds the character spans of all evolve blocks in the parent code.

    This function searches for evolve blocks in the parent code using regex
    and returns their start and end positions for later modification.

    Args:
        parent_code: The source code containing evolve blocks to be modified.
        evolve_regex: Regular expression pattern to match evolve blocks. Should
                     contain one capture group for the block content.

    Returns:
        List of tuples where each tuple contains (start_pos, end_pos) character
        indices of evolve block content (excluding markers).

    Raises:
        EvolveBlockError: If no evolve blocks are found in the parent code.
    """
    evolve_spans: List[Tuple[int, int]] = []

    for match in re.finditer(evolve_regex, parent_code, re.DOTALL):
        evolve_spans.append(match.span(1))

    if not evolve_spans:
        raise EvolveBlockError("No EVOLVE blocks found.")

    return evolve_spans


def assign_diffs_to_blocks(
    parent_code: str,
    diff_blocks: List[Tuple[str, str]],
    evolve_spans: List[Tuple[int, int]],
) -> Dict[int, List[int]]:
    """Assigns diff operations to their corresponding evolve blocks.

    This function determines which evolve block each diff operation should be
    applied to by searching for the search text within each block's content.

    Args:
        parent_code: The source code containing evolve blocks.
        diff_blocks: List of (search_text, replace_text) tuples from diff parsing.
        evolve_spans: List of (start_pos, end_pos) tuples for evolve blocks.

    Returns:
        Dictionary mapping evolve block indices to lists of diff block indices
        that should be applied to that block.

    Raises:
        SearchAndReplaceError: If a search pattern cannot be found in any evolve block.
    """

    block_assignments: Dict[int, List[int]] = {i: [] for i in range(len(evolve_spans))}

    for diff_idx, (search_text, _) in enumerate(diff_blocks):
        match_found: bool = False

        for block_idx, (start, end) in enumerate(evolve_spans):
            block_content = parent_code[start:end]

            if search_text in block_content:
                match_found = True
                block_assignments[block_idx].append(diff_idx)
                break

        if not match_found:
            raise SearchAndReplaceError(
                f"Search block not found in any evolve blocks: '{search_text}'"
            )

    return block_assignments


def apply_replacements(
    parent_code: str,
    diff_blocks: List[Tuple[str, str]],
    evolve_spans: List[Tuple[int, int]],
    block_assignments: Dict[int, List[int]],
) -> List[str]:
    """Applies search-and-replace operations to evolve blocks.

    This function performs the actual text replacements within each evolve block
    based on the assigned diff operations, processing them in sequence.

    Args:
        parent_code: The source code containing evolve blocks.
        diff_blocks: List of (search_text, replace_text) tuples.
        evolve_spans: List of (start_pos, end_pos) tuples for evolve blocks.
        block_assignments: Mapping of block indices to diff indices to apply.

    Returns:
        List of modified content strings for each evolve block after applying
        all assigned replacements.
    """
    modified_blocks: List[str] = []

    for block_idx, (start, end) in enumerate(evolve_spans):
        current_content: str = parent_code[start:end]

        for diff_idx in block_assignments[block_idx]:
            search_text, replace_text = diff_blocks[diff_idx]
            current_content = current_content.replace(search_text, replace_text, 1)

        modified_blocks.append(current_content)

    return modified_blocks


def apply_diff(
    parent_code: str,
    diff: str,
    evolve_regex: str = r"\s*EVOLVE-BLOCK-START\s*\n?(.*?)\n?\s*EVOLVE-BLOCK-END",
    diff_regex: str = r"<{7}\s*SEARCH\s*\n?(.*?)\n?\s*={7}\s*\n?(.*?)\n?\s*>{7}\s*REPLACE",
) -> str:
    """Applies diff operations to evolve blocks in parent code.

    This is the main function that orchestrates the diff application process.
    It parses the diff, finds evolve blocks, assigns operations to blocks,
    applies the replacements, and reconstructs the modified code.

    Args:
        parent_code: The source code containing EVOLVE-BLOCK markers.
        diff: The diff string containing search-and-replace operations.
        evolve_regex: Regex pattern to match evolve blocks (default handles
                     EVOLVE-BLOCK-START/END markers).
        diff_regex: Regex pattern to match diff operations (default handles
                   <<<<<<< SEARCH / ======= / >>>>>>> REPLACE format).

    Returns:
        The modified source code with diff operations applied to evolve blocks.

    Raises:
        DiffError: If no diff blocks are found.
        EvolveBlockError: If no evolve blocks are found.
        SearchAndReplaceError: If search patterns cannot be matched to blocks.
    """

    diff_blocks: List[Tuple[str, str]] = parse_diff_blocks(diff=diff, diff_regex=diff_regex)

    evolve_spans: List[Tuple[int, int]] = find_evolve_block_spans(
        parent_code=parent_code, evolve_regex=evolve_regex
    )

    block_assignments: Dict[int, List[int]] = assign_diffs_to_blocks(
        parent_code=parent_code, diff_blocks=diff_blocks, evolve_spans=evolve_spans
    )

    modified_blocks: List[str] = apply_replacements(
        parent_code=parent_code,
        diff_blocks=diff_blocks,
        evolve_spans=evolve_spans,
        block_assignments=block_assignments,
    )

    child_code_parts: List[str] = []
    last_end: int = 0
    for i, (start, end) in enumerate(evolve_spans):
        child_code_parts.append(parent_code[last_end:start])
        child_code_parts.append(modified_blocks[i])
        last_end = end

    child_code_parts.append(parent_code[last_end:])

    return "".join(child_code_parts)
