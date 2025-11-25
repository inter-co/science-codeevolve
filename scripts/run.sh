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

PROB_NAME="alphaevolve_math_problems/circle_packing_square/26"
BASE_DIR="problems/${PROB_NAME}"
INPT_DIR="${BASE_DIR}/input/"
CFG_PATH="${BASE_DIR}/configs/config_mp_insp.yaml"
OUT_DIR="experiments/${PROB_NAME}/test/"
LOAD_CKPT=-1
CPU_LIST=""

taskset --cpu-list $CPU_LIST codeevolve --inpt_dir=$INPT_DIR --cfg_path=$CFG_PATH --out_dir=$RESULTS_DIR --load_ckpt=$LOAD_CKPT --terminal_logging