# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file centralizes magic constants used throughout CodeEvolve.
#
# ===--------------------------------------------------------------------------------------===#

from typing import Dict

# ---------------------------------------------------------------------------
# Code Block Markers
# ---------------------------------------------------------------------------

# Default markers for code evolution blocks
DEFAULT_EVOLVE_START_MARKER: str = "# EVOLVE-BLOCK-START"
DEFAULT_EVOLVE_END_MARKER: str = "# EVOLVE-BLOCK-END"

# Default markers for prompt evolution blocks (meta-prompting)
DEFAULT_PROMPT_START_MARKER: str = "# PROMPT-BLOCK-START"
DEFAULT_PROMPT_END_MARKER: str = "# PROMPT-BLOCK-END"


# ---------------------------------------------------------------------------
# Regex Patterns
# ---------------------------------------------------------------------------

# Pattern for parsing SEARCH/REPLACE diff blocks from LLM responses
DEFAULT_DIFF_REGEX: str = r"<{7}\s*SEARCH\s*\n?(.*?)\n?\s*={7}\s*\n?(.*?)\n?\s*>{7}\s*REPLACE"

# Pattern for identifying checkpoint files
CHECKPOINT_PATTERN: str = r"ckpt_(\d+)\.pkl$"  # must match CHECKPOINT_FILE_FORMAT defined below


# ---------------------------------------------------------------------------
# File Names
# ---------------------------------------------------------------------------

LOCK_FILE: str = ".codeevolve.lock"
ISLAND_LOG_FILE: str = "island_{time}.log"
CRASH_LOG_FILE: str = "crash.log"
RUN_METADATA_FILE: str = "run_metadata.json"
BEST_SOLUTION_FILE: str = "best_sol"
BEST_PROMPT_FILE: str = "best_prompt.txt"
CHECKPOINT_FILE_FORMAT: str = "ckpt_{epoch}.pkl"

# ---------------------------------------------------------------------------
# Default Configuration Values
# ---------------------------------------------------------------------------

# Evaluation defaults
DEFAULT_EVAL_TIMEOUT_S: int = 60
DEFAULT_MAX_MEM_BYTES: int = 1 * 1024 * 1024 * 1024
DEFAULT_MEM_CHECK_INTERVAL_S: float = 0.1

# Migration defaults
DEFAULT_MIGRATION_INTERVAL: int = 20
DEFAULT_MIGRATION_RATE: float = 0.1

# Logging defaults
DEFAULT_MAX_LOG_MSG_SIZE: int = 256


# ---------------------------------------------------------------------------
# Mock Model
# ---------------------------------------------------------------------------

# Prefix for mock model names
MOCK_MODEL_PREFIX: str = "MOCK"


# ---------------------------------------------------------------------------
# Language Extensions
# ---------------------------------------------------------------------------

LANGUAGE_TO_EXTENSION: Dict[str, str] = {
    "python": ".py",
    "javascript": ".js",
    "java": ".java",
    "cpp": ".cpp",
    "c": ".c",
    "csharp": ".cs",
    "go": ".go",
    "rust": ".rs",
    "typescript": ".ts",
    "php": ".php",
    "ruby": ".rb",
    "swift": ".swift",
    "kotlin": ".kt",
    "scala": ".scala",
    "r": ".r",
    "matlab": ".m",
    "shell": ".sh",
    "powershell": ".ps1",
    "sql": ".sql",
}

DEFAULT_EXTENSION: str = ".txt"

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

ASCII_NAME: str = """       
              ██████               ██         ██████                 ██                             
             ██    ██  █████   ██████  █████  ██      ██  ██  ████   ██ ██   ██  ████               
             ██       ██   ██ ██   ██ ██   ██ ██████  ██  █  ██   █  ██  ██ ██  █   ██              
             ██    ██ ██   ██ ██   ██ ██      ██       ████  ██   █  ██  ██ █  ██                   
               █████   █████   ██████  █████  ██████    ██    ████   ██   ███    ████               
                                                                                                                                                                                           
"""

ASCII_LOGO: str = """                                                                                                                                                                                            
                                         █████        █████                                         
                                     ██    ███████████████   ███                                    
                                  ██  ████████████ ███████████   ██                                 
                               ██  █████████████    ██████████████ ██                               
                             ██  ██████████████      ███████████████  █                             
                            █  ██████████  █████     ████  ███████████ ██                           
                          ██ ████████ ████ █   ██  ██   ██████  ███████  █                          
                         █  █████████    ███    █  █    ██ █    █████████ █                         
                        █  ██████████      █ █████ █████   █   ███████████ █                        
                       ██ █████████████████████ ██ ██ █████████████████████ █                       
                       █ ██████     █████  █   █ █  ██  ██  █████     ██████ █                      
                      █ ███████    ██ █   ██  ███  ███  ██   █████   ███████ █                      
                      █ ███████████████  █  ███  █ █ ███     ████████████████ █                     
                     █ ██████████     ██  █   ███   ██ █    ██     ██████████ █                     
                     █ ████████████   ███████ ███  ███████████    ███████████ █                     
                     █ ███████████████████      █  ██       █████████████████ █                     
                     █ ███████████   ██   ██████    ███████  ██   ███████████ █                     
                     █ ███████████████   ███████ ██  ██████   ███████████████ █                     
                     ██ ███████████████████████  ██  ████████████████████████ █                     
                      █ ██████████████████████  ████   ██████████████████ ██ ██                     
                      ██ █████████████████████  ████  ██████████████████████ █                      
                       █ ██████████████████████  ██  ██████████████████████ █                       
                        █ ██████████████████████ ██ ██████████████████████ ██                       
                         █ █████████████████████     ████████████████████ ██                        
                          █  ████████████████          █████████████████ ██                         
                           ██ ████ ██        █ ██  ██ █        ███████  █                           
                             ██ ███ ██ ███ ██ ██  █ █████ ███ ██ ████ ██                            
                               ██ ████████████    ██  ██ █████████  ██                              
                                 ██  ████  ████ █████ ████ █████ ██                                 
                                    ███  ███ ██████████ ████  ███                                   
                                        █████           █████                                       
                                                 ███                                                
                                                                                                                                                                                                                                                                                                                                       
"""
