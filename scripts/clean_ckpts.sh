# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file provides a script for deleting all but the most recent checkpoint file from
# a given experiment.
#
# ===--------------------------------------------------------------------------------------===#

#!/bin/bash
if [ -z "$1" ]; then
    echo "Error: Please provide the path to the main experiments folder."
    echo "Usage: bash clean_checkpoints.sh /path/to/your/experiments"
    exit 1
fi

BASE_DIR="$1"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: The path '$BASE_DIR' is not a valid directory."
    exit 1
fi

echo "Starting checkpoint cleanup in: $BASE_DIR"
echo "---"

for exp_dir in "$BASE_DIR"/*/; do

    if [ -d "$exp_dir" ]; then
        echo "Scanning Experiment: $(basename "$exp_dir")"

        for run_dir in "$exp_dir"*/; do

            ckpt_dir="${run_dir}ckpt"

            if [ -d "$ckpt_dir" ]; then
                echo "  -> Processing: $ckpt_dir"

                files_to_delete=$(ls -v "$ckpt_dir"/ckpt_*.pkl 2>/dev/null | head -n -1)

                if [ -n "$files_to_delete" ]; then
                    echo "     The following files will be deleted:"

                    echo "$files_to_delete" | xargs -n 1 basename
                    echo "$files_to_delete" | xargs rm

                    echo "     Successfully deleted old checkpoints."
                else
                    echo "     No old checkpoints to delete in this folder."
                fi
            fi
        done
        echo "---"
    fi
done

echo "Cleanup complete."