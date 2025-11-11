# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file provides a template for executing CodeEvolve in the terminal using bash.
#
# ===--------------------------------------------------------------------------------------===#

#!/bin/bash

BASE_DIR="benchmarks/circle_packing_square/26"
INPT_DIR="${BASE_DIR}/input/"
CFG_PATH="${BASE_DIR}/configs/config.yaml"
OUT_DIR="${BASE_DIR}/experiments/test/"
LOAD_CKPT=-1
CPU_LIST="0-7"

taskset --cpu-list $CPU_LIST codeevolve --inpt_dir=$INPT_DIR --cfg_path=$CFG_PATH --out_dir=$RESULTS_DIR --load_ckpt=$LOAD_CKPT --terminal_logging