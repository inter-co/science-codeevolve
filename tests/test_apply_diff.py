# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements basic tests for parsing LM responses.
#
# ===--------------------------------------------------------------------------------------===#

import pytest

from codeevolve.utils.parsing_utils import (
    apply_diff,
    SearchAndReplaceError,
    DiffError,
    EvolveBlockError,
)


class TestDiff:
    """Test suite for the apply_diff function and related parsing utilities.

    This test class validates the functionality of diff application to evolve blocks,
    covering various scenarios including single and multiple blocks, different
    comment styles, multiline replacements, and error conditions.
    """

    # Positive test cases

    def test_single_block(self):
        """Tests basic diff application to a single evolve block with hash comments."""
        parent_code = """
# EVOLVE-BLOCK-START
old_code
# EVOLVE-BLOCK-END
"""
        diff = """
<<<<<<< SEARCH
old_code
=======
new_code
>>>>>>> REPLACE
"""
        child_code = apply_diff(parent_code, diff)
        assert (
            child_code
            == """
# EVOLVE-BLOCK-START
new_code
# EVOLVE-BLOCK-END
"""
        )

    def test_single_block2(self):
        """Tests basic diff application to a single evolve block with C-style comments."""
        parent_code = """
// EVOLVE-BLOCK-START
old_code
// EVOLVE-BLOCK-END
"""
        diff = """
<<<<<<< SEARCH
old_code
=======
new_code
>>>>>>> REPLACE
"""
        child_code = apply_diff(parent_code, diff, "// EVOLVE-BLOCK-START", "// EVOLVE-BLOCK-END")
        assert (
            child_code
            == """
// EVOLVE-BLOCK-START
new_code
// EVOLVE-BLOCK-END
"""
        )

    def test_single_block3(self):
        """Tests basic diff application to a single evolve block with hash comments within search."""
        parent_code = """
# EVOLVE-BLOCK-START
old_code
# EVOLVE-BLOCK-END
"""
        diff = """
<<<<<<< SEARCH
# EVOLVE-BLOCK-START
old_code
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
new_code
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
"""
        child_code = apply_diff(parent_code, diff)
        assert (
            child_code
            == """
# EVOLVE-BLOCK-START
new_code
# EVOLVE-BLOCK-END
"""
        )

    def test_multi_diff_block(self):
        """Tests multiple diff operations applied to a single evolve block in Python."""
        parent_code = """
# EVOLVE-BLOCK-START
def foo(x:int):
    return x+5
def bar(y:int):
    return y+6
# EVOLVE-BLOCK-END
"""
        diff = """
<<<<<<< SEARCH
def foo(x:int):
    return x+5
=======
def foobar(x:int):
    return x+5
>>>>>>> REPLACE
<<<<<<< SEARCH
def bar(y:int):
    return y+6
=======
def barfoo(y:int):
    return y+6
>>>>>>> REPLACE
"""
        child_code = apply_diff(parent_code, diff)
        assert (
            child_code
            == """
# EVOLVE-BLOCK-START
def foobar(x:int):
    return x+5
def barfoo(y:int):
    return y+6
# EVOLVE-BLOCK-END
"""
        )

    def test_multi_diff_block2(self):
        """Tests multiple diff operations applied to a single evolve block in C++."""
        parent_code = """
// EVOLVE-BLOCK-START
int foo(const int x){
    return x+5;
}
int bar(const int y){
    return y+6
}
// EVOLVE-BLOCK-END
"""
        diff = """
<<<<<<< SEARCH
int foo(const int x){
    return x+5;
}
=======
int foobar(const int x){
    return x+5;
}
>>>>>>> REPLACE
<<<<<<< SEARCH
int bar(const int y){
    return y+6
}
=======
int barfoo(const int y){
    return y+6
}
>>>>>>> REPLACE
"""
        child_code = apply_diff(parent_code, diff, "// EVOLVE-BLOCK-START", "// EVOLVE-BLOCK-END")
        assert (
            child_code
            == """
// EVOLVE-BLOCK-START
int foobar(const int x){
    return x+5;
}
int barfoo(const int y){
    return y+6
}
// EVOLVE-BLOCK-END
"""
        )

    def test_multi_evolve_block(self):
        """Tests diff application across multiple separate evolve blocks."""
        parent_code = """
# EVOLVE-BLOCK-START
def foo(x:int):
    return x+5
def bar(y:int):
    return y+6
# EVOLVE-BLOCK-END
def wont_change():
    print("This should not be evolved.")
# EVOLVE-BLOCK-START
def foo2(x:int):
    return x+5
def bar2(y:int):
    return y+6
# EVOLVE-BLOCK-END
"""
        diff = """
<<<<<<< SEARCH
def foo(x:int):
    return x+5
=======
def foobar(x:int):
    return x+5
>>>>>>> REPLACE
<<<<<<< SEARCH
def bar(y:int):
    return y+6
=======
def barfoo(y:int):
    return y+6
>>>>>>> REPLACE
<<<<<<< SEARCH
def foo2(x:int):
    return x+5
=======
def foobar2(x:int):
    return x+5
>>>>>>> REPLACE
<<<<<<< SEARCH
def bar2(y:int):
    return y+6
=======
def barfoo2(y:int):
    return y+6
>>>>>>> REPLACE
"""
        child_code = apply_diff(parent_code, diff)
        assert (
            child_code
            == """
# EVOLVE-BLOCK-START
def foobar(x:int):
    return x+5
def barfoo(y:int):
    return y+6
# EVOLVE-BLOCK-END
def wont_change():
    print("This should not be evolved.")
# EVOLVE-BLOCK-START
def foobar2(x:int):
    return x+5
def barfoo2(y:int):
    return y+6
# EVOLVE-BLOCK-END
"""
        )

    def test_first_match_only(self):
        """Tests that replacements only apply to the first matching occurrence in each block."""
        parent_code = """
# EVOLVE-BLOCK-START
def foo(x:int):
    return x+5
def bar(y:int):
    return y+6
# EVOLVE-BLOCK-END
def wont_change():
    print("This should not be evolved.")
# EVOLVE-BLOCK-START
def foo2(x:int):
    return x+5
def bar2(y:int):
    return y+6
# EVOLVE-BLOCK-END
"""
        diff = """
<<<<<<< SEARCH
    return x+5
=======
    return x+7
>>>>>>> REPLACE
<<<<<<< SEARCH
    return y+6
=======
    return y+8
>>>>>>> REPLACE
"""
        child_code = apply_diff(parent_code, diff)
        assert (
            child_code
            == """
# EVOLVE-BLOCK-START
def foo(x:int):
    return x+7
def bar(y:int):
    return y+8
# EVOLVE-BLOCK-END
def wont_change():
    print("This should not be evolved.")
# EVOLVE-BLOCK-START
def foo2(x:int):
    return x+5
def bar2(y:int):
    return y+6
# EVOLVE-BLOCK-END
"""
        )

    def test_multiline_replacement(self):
        """Tests diff application with multiline search and replace blocks."""
        parent_code = """
class Calculator:
    # EVOLVE-BLOCK-START
    def add(self, a, b):
        return a + b
    # EVOLVE-BLOCK-END
    
    def multiply(self, a, b):
        return a * b
"""
        diff = '''
<<<<<<< SEARCH
    def add(self, a, b):
        return a + b
=======
    def add(self, a, b):
        """Add two numbers with logging."""
        result = a + b
        print(f"Adding {a} + {b} = {result}")
        return result
>>>>>>> REPLACE
'''
        child_code = apply_diff(parent_code, diff)
        assert (
            child_code
            == '''
class Calculator:
    # EVOLVE-BLOCK-START
    def add(self, a, b):
        """Add two numbers with logging."""
        result = a + b
        print(f"Adding {a} + {b} = {result}")
        return result
    # EVOLVE-BLOCK-END
    
    def multiply(self, a, b):
        return a * b
'''
        )

    def test_empty_replace(self):
        """Tests diff application where the replacement text is empty (deletion)."""
        parent_code = """
# EVOLVE-BLOCK-START
    old_code
# EVOLVE-BLOCK-END
"""
        diff = """
<<<<<<< SEARCH
    old_code
=======
>>>>>>> REPLACE
        """
        child_code = apply_diff(parent_code, diff)
        assert (
            child_code
            == """
# EVOLVE-BLOCK-START
    
# EVOLVE-BLOCK-END
"""
        )

    # Negative test cases

    def test_diff_error(self):
        """Tests that DiffError is raised when no diff blocks are found."""
        parent_code = """
old_code
"""
        diff = ""

        with pytest.raises(DiffError):
            apply_diff(parent_code, diff)

    def test_evolve_block_error(self):
        """Tests that EvolveBlockError is raised when evolve blocks are malformed or missing."""
        parent_code1 = """
old_code
"""
        parent_code2 = """
# EVOLVE-BLOCK-START
old_code
"""
        parent_code3 = """
old_code
# EVOLVE-BLOCK-END
"""
        diff = """
<<<<<<< SEARCH
old_code
=======
new_code
>>>>>>> REPLACE
"""
        with pytest.raises(EvolveBlockError):
            child_code = apply_diff(parent_code1, diff)
        with pytest.raises(EvolveBlockError):
            child_code = apply_diff(parent_code2, diff)
        with pytest.raises(EvolveBlockError):
            child_code = apply_diff(parent_code3, diff)

    def test_search_and_replace_error(self):
        """Tests that SearchAndReplaceError is raised when search text is not found in any evolve block."""
        parent_code = """
class Calculator:
    # EVOLVE-BLOCK-START
    def add(self, a, b):
        return a + b
    # EVOLVE-BLOCK-END
    
    def multiply(self, a, b):
        return a * b
"""
        diff = """
<<<<<<< SEARCH
    def div(self, a, b):
        return a/b
=======
    def div_safe(self, a, b):
        if(b != 0):
            return a/b
        else:
            raise ValueError("Division by zero.")
>>>>>>> REPLACE
"""
        with pytest.raises(SearchAndReplaceError):
            child_code = apply_diff(parent_code, diff)

    def test_search_and_replace_error_multi(self):
        """Tests SearchAndReplaceError with multiple diff blocks where one search fails."""
        parent_code = """
class Calculator:
    # EVOLVE-BLOCK-START
    def add(self, a, b):
        return a + b
    # EVOLVE-BLOCK-END
    
    def multiply(self, a, b):
        return a * b
"""
        diff = '''
<<<<<<< SEARCH
    def add(self, a, b):
        return a + b
=======
    def add(self, a, b):
        """Add two numbers with logging."""
        result = a + b
        print(f"Adding {a} + {b} = {result}")
        return result
>>>>>>> REPLACE
<<<<<<< SEARCH
    def div(self, a, b):
        return a/b
=======
    def div_safe(self, a, b):
        if(b != 0):
            return a/b
        else:
            raise ValueError("Division by zero.")
>>>>>>> REPLACE
'''
        with pytest.raises(SearchAndReplaceError):
            child_code = apply_diff(parent_code, diff)
