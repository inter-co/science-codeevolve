#!/bin/bash
# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file provides a script for recursively finding all ckpt folders and deleting all but
# the most recent checkpoint from each.
#
# ===--------------------------------------------------------------------------------------===#

if [ -z "$1" ]; then
    echo "Error: Please provide the path to the target folder."
    echo "Usage: bash clean_checkpoints.sh /path/to/your/folder"
    exit 1
fi

BASE_DIR="$1"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: The path '$BASE_DIR' is not a valid directory."
    exit 1
fi

echo "Starting recursive checkpoint cleanup in: $BASE_DIR"
echo "---"

# Find all directories named 'ckpt' recursively
find "$BASE_DIR" -type d -name "ckpt" | while read -r ckpt_dir; do
    echo "Processing: $ckpt_dir"
    
    # Find all ckpt_*.pkl files, sort them, and get all but the last one
    files_to_delete=$(ls -v "$ckpt_dir"/ckpt_*.pkl 2>/dev/null | head -n -1)
    
    if [ -n "$files_to_delete" ]; then
        echo "  The following files will be deleted:"
        echo "$files_to_delete" | xargs -n 1 basename
        echo "$files_to_delete" | xargs rm
        echo "  Successfully deleted old checkpoints."
    else
        echo "  No old checkpoints to delete (only 1 or 0 checkpoints found)."
    fi
    
    echo "---"
done

echo "Cleanup complete."