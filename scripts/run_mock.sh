#!/bin/bash
# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file provides provides a bash script to execute a mock run of CodeEvolve.
#
# ===--------------------------------------------------------------------------------------===#

PROB_NAME="alphaevolve_math_problems/packing_problems/circle_packing_square/26"
BASE_DIR="problems/${PROB_NAME}"
INPT_DIR="${BASE_DIR}/input/"
CFG_PATH="configs/templates/config_mock.yaml"
OUT_DIR="debug/${PROB_NAME}/mock/"
LOAD_CKPT=0
CPU_LIST="0"

export API_BASE="test" 
export API_KEY="test"

taskset --cpu-list $CPU_LIST codeevolve \
    --inpt_dir=$INPT_DIR \
    --cfg_path=$CFG_PATH \
    --out_dir=$OUT_DIR \
    --load_ckpt=$LOAD_CKPT