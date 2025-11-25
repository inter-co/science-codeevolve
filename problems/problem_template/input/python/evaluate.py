# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements an example of an solution evaluator.
#
# ===--------------------------------------------------------------------------------------===#

import sys
import os
from importlib import __import__
import json


def evaluate(program_path: str, results_path: str) -> None:
    """
    Evaluate function.
    Should receive the path to the temporary file containing the current solution's code,
    and a path to where the json of results should be saved.
    """

    # add program_path to sys path and import
    abs_program_path = os.path.abspath(program_path)
    program_dir = os.path.dirname(abs_program_path)
    module_name = os.path.splitext(os.path.basename(program_path))[0]

    try:
        sys.path.insert(0, program_dir)
        program = __import__(module_name)
        output = program.run()
    except Exception as err:
        raise err
    finally:
        if program_dir in sys.path:
            sys.path.remove(program_dir)

    with open(results_path, "w") as f:
        json.dump({"output": output}, f, indent=4)


if __name__ == "__main__":
    program_path = sys.argv[1]
    results_path = sys.argv[2]

    evaluate(program_path, results_path)
