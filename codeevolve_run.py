# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the entry point script for CodeEvolve.
#
# ===--------------------------------------------------------------------------------------===#

import sys
from codeevolve.cli import main

if __name__ == "__main__":
    sys.exit(main())