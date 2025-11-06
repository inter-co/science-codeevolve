# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements many functions for data analysis, used in the jupyter notebooks
# for CodeEvolve.
#
# ===--------------------------------------------------------------------------------------===#

import os
import sys
import re
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import itertools
from importlib import __import__
from collections import defaultdict
import pickle as pkl

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np

from codeevolve.utils.ckpt_utils import load_ckpt
from codeevolve.database import ProgramDatabase


def get_total_runtime(
    log_file_path: str, start_marker: str = "=== EVOLVE ALGORITHM ==="
) -> timedelta:
    """ """

    def _parse_time(line: str) -> Optional[datetime]:
        """Extracts datetime from a log line.

        Args:
            line: Log line containing timestamp in the expected format.

        Returns:
            Parsed datetime object or None if parsing fails.
        """
        try:
            timestamp_str = line[11:].split(" | ", 1)[0]
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
        except (ValueError, IndexError):
            return None

    with open(log_file_path, "r") as f:
        lines = f.readlines()
    if not lines:
        return timedelta(0)

    start_indices = [i for i, line in enumerate(lines) if start_marker in line]
    if not start_indices:
        return timedelta(0), lines, start_indices

    total_duration = timedelta(0)
    for i, start_index in enumerate(start_indices):
        start_time = _parse_time(lines[start_index])
        if not start_time:
            continue
        if i + 1 < len(start_indices):
            end_index = start_indices[i + 1] - 1
        else:
            end_index = len(lines) - 1
        end_time = _parse_time(lines[end_index])
        if end_time and end_time >= start_time:
            segment_duration = end_time - start_time
            total_duration += segment_duration

    return total_duration


def get_experiment_df(
    experiment_res: Dict[str, Any],
    results_dir,
    config: Dict[Any, Any],
    model_names: List[str],
) -> pd.DataFrame:
    """ """
    df_rows = []

    for isl_id in experiment_res.keys():
        island_results_dir = results_dir + f"{isl_id}/"

        training_time = get_total_runtime(island_results_dir + "results.log")
        training_time = training_time.seconds / 3600

        evolve_state = experiment_res[isl_id]["evolve_state"]
        num_epochs = experiment_res[isl_id]["num_epochs"]

        sol_db = experiment_res[isl_id]["sol_db"]
        best_fitness = sol_db.programs[sol_db.best_prog_id].fitness

        epoch_best_found = float("inf")
        for prog in sol_db.programs.values():
            if prog.fitness == best_fitness:
                epoch_best_found = (
                    min(prog.iteration_found, epoch_best_found) if prog.iteration_found else 0
                )

        num_sr_errors = len(
            [
                error_info
                for error_info in evolve_state["errors"]
                if error_info["motive"] == "sr_evolve_prog"
            ]
        )
        num_eval_errors = len([prog for prog in sol_db.programs.values() if prog.error])
        num_eval_warnings = len([prog for prog in sol_db.programs.values() if prog.warning])

        model2tok_usage = defaultdict(lambda: defaultdict(int))
        for tok_info in evolve_state["tok_usage"]:
            if tok_info["motive"] == "generate_prog":
                model2tok_usage[tok_info["model_name"]]["prompt_tok"] += tok_info["prompt_tok"]
                model2tok_usage[tok_info["model_name"]]["compl_tok"] += tok_info["compl_tok"]

        tok_usage_row = []
        for model_name in model_names:
            tok_usage_row += [
                model2tok_usage[model_name]["prompt_tok"],
                model2tok_usage[model_name]["compl_tok"],
            ]

        prompt_db = experiment_res[isl_id]["prompt_db"]
        mp_best_fitness = prompt_db.programs[prompt_db.best_prog_id].fitness

        mp_epoch_best_found = float("inf")
        for prog in prompt_db.programs.values():
            if prog.fitness == mp_best_fitness:
                mp_epoch_best_found = (
                    min(prog.iteration_found, mp_epoch_best_found) if prog.iteration_found else 0
                )

        mp_num_sr_errors = len(
            [
                error_info
                for error_info in evolve_state["errors"]
                if error_info["motive"] == "sr_meta_prompt"
            ]
        )
        mp_num_eval_errors = len([prog for prog in prompt_db.programs.values() if prog.error])
        mp_num_eval_warnings = len([prog for prog in prompt_db.programs.values() if prog.warning])

        mp_model2tok_usage = defaultdict(lambda: defaultdict(int))
        for tok_info in evolve_state["tok_usage"]:
            if tok_info["motive"] == "meta_prompt":
                mp_model2tok_usage[tok_info["model_name"]]["prompt_tok"] += tok_info["prompt_tok"]
                mp_model2tok_usage[tok_info["model_name"]]["compl_tok"] += tok_info["compl_tok"]

        mp_tok_usage_row = []
        for model_name in model_names:
            mp_tok_usage_row += [
                mp_model2tok_usage[model_name]["prompt_tok"],
                mp_model2tok_usage[model_name]["compl_tok"],
            ]

        df_rows.append(
            [
                isl_id,
                num_epochs,
                training_time,
                best_fitness,
                epoch_best_found,
                num_sr_errors,
                num_eval_errors,
                num_eval_warnings,
            ]
            + [
                mp_best_fitness,
                mp_epoch_best_found,
                mp_num_sr_errors,
                mp_num_eval_errors,
                mp_num_eval_warnings,
            ]
            + tok_usage_row
            + mp_tok_usage_row
        )

    tok_usage_cols = [
        model_name + suffix
        for model_name, suffix in itertools.product(model_names, ["(prompt_tok)", "(compl_tok)"])
    ]
    mp_tok_usage_cols = [
        "mp_" + model_name + suffix
        for model_name, suffix in itertools.product(model_names, ["(prompt_tok)", "(compl_tok)"])
    ]

    return pd.DataFrame(
        df_rows,
        columns=[
            "isl_id",
            "num_epochs",
            "training_time (hours)",
            "best_fitness",
            "epoch_best_found",
            "num_sr_errors",
            "num_eval_errors",
            "num_eval_warnings",
        ]
        + [
            "mp_best_fitness",
            "mp_epoch_best_found",
            "mp_num_sr_errors",
            "mp_num_eval_errors",
            "mp_num_eval_warnings",
        ]
        + tok_usage_cols
        + mp_tok_usage_cols,
    )


def process_experiments(
    args: Dict, model_names: List[str], model2cost: Dict[str, Dict[str, float]]
):
    """ """
    experiments_res = {}
    for idx, result_path in enumerate(args["out_dirs"]):
        result_dir = args["inpt_dir"] + f"experiments/{result_path}"

        cfg_fname = [fname for fname in os.listdir(result_dir) if fname.endswith(".yaml")][0]
        with open(result_dir + cfg_fname, "r") as f:
            config = yaml.safe_load(f)

        experiment_res: Dict[int, Dict[str, Any]] = {}

        ckpt = -1 if not args.get("ckpts", None) else args["ckpts"][idx]

        for isl_id in range(0, config["EVOLVE_CONFIG"]["num_islands"]):
            island_results_dir: str = result_dir + f"{isl_id}/"
            ckpt_dir: str = island_results_dir + "ckpt/"
            try:
                prompt_db, sol_db, evolve_state = load_ckpt(ckpt, ckpt_dir)
            except:
                ckpts: List[str] = [
                    f for f in os.listdir(ckpt_dir) if re.match(r"ckpt_\d+\.pkl$", f)
                ]
                if len(ckpts):
                    ckpt = max([int(re.search(r"ckpt_(\d+)\.pkl$", f).group(1)) for f in ckpts])
                    prompt_db, sol_db, evolve_state = load_ckpt(ckpt, ckpt_dir)
                else:
                    raise ValueError(f"No ckpts were found for island {isl_id}.")

            experiment_res[isl_id] = {
                "prompt_db": prompt_db,
                "sol_db": sol_db,
                "evolve_state": evolve_state,
                "num_epochs": ckpt,
            }

        experiment_df = get_experiment_df(experiment_res, result_dir, config, model_names)

        tok_cols = experiment_df.columns[-8:]
        estimated_cost = 0
        for col in tok_cols:
            mult = "prompt_pm" if "prompt" in col else "compl_pm"
            for model_name in model_names:
                if model_name in col:
                    estimated_cost += (
                        (experiment_df[[col]] * model2cost[model_name][mult] * 1e-6).sum().sum()
                    )

        best_island = experiment_df["best_fitness"].idxmax()

        experiments_res[result_path] = {
            "res": experiment_res,
            "config": config,
            "df": experiment_df,
            "cost": estimated_cost,
        }

    return experiments_res


def get_experiment_sol(results_dir, sol_func_name, island_id: int):
    """ """
    program_path = results_dir + f"{island_id}/best_sol.py"
    if "best_sol" in sys.modules:
        del sys.modules["best_sol"]

    abs_program_path = os.path.abspath(program_path)
    program_dir = os.path.dirname(abs_program_path)
    module_name = os.path.splitext(os.path.basename(program_path))[0]
    try:
        sys.path.insert(0, program_dir)
        program = __import__(module_name)
        sol = getattr(program, sol_func_name)()
        pkl.dump(sol, open(results_dir + "best_sol.pkl", "wb"))
    except Exception as err:
        raise err
    finally:
        if program_dir in sys.path:
            sys.path.remove(program_dir)

    return sol


# plotting


def plot_experiments_statistical_summary(
    experiments: Dict[str, Dict[int, Any]],
    title,
    save_path: str = None,
    figsize: tuple = (6, 4),
    epsilon: float = 1e-3,
):
    """Plots statistical summary of fitness evolution across multiple experiments.

    This function creates a line plot showing the evolution of fitness values over
    epochs for multiple experiments. It uses a logarithmic transformation to better
    visualize convergence towards optimal fitness values and handles NaN values
    and variable-length histories across experiments.

    Args:
        experiments: Dictionary mapping experiment names to their results containing
                    fitness histories for each island/run.
        title: Title for the plot.
        save_path: Optional path to save the plot image. If None, plot is only displayed.
        figsize: Tuple specifying the figure size (width, height) in inches.
        epsilon: Small value used for numerical stability in logarithmic transformation.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    default_colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    default_markers = ["o", "s", "^", "v", "D", "p", "*", "h", "H", "+"]

    all_fitness_values = []
    experiment_data = {}

    for exp_name, results in experiments.items():
        all_best_hists = []

        for key, data in results.items():
            try:
                best_hist = data["evolve_state"]["best_fit_hist"]
                if best_hist:
                    all_best_hists.append(best_hist)
            except (KeyError, TypeError):
                continue

        if not all_best_hists:
            print(f"No valid data found for experiment: {exp_name}")
            continue

        experiment_data[exp_name] = all_best_hists

        for hist in all_best_hists:
            for value in hist:
                if not np.isnan(value):
                    all_fitness_values.append(value)

    if not all_fitness_values:
        print("No valid fitness values found across all experiments!")
        return

    MAX_FITNESS = max(all_fitness_values)

    for exp_idx, (exp_name, all_best_hists) in enumerate(experiment_data.items()):

        def pad_with_last_non_nan(hist):
            """Pad history with last non-NaN value, handling NaNs properly"""
            last_valid = None
            for i in range(len(hist) - 1, -1, -1):
                if not np.isnan(hist[i]):
                    last_valid = hist[i]
                    break
            if last_valid is None:
                return None

            cleaned_hist = []
            current_valid = None

            for value in hist:
                if not np.isnan(value):
                    current_valid = value
                    cleaned_hist.append(value)
                else:
                    cleaned_hist.append(current_valid if current_valid is not None else last_valid)

            return cleaned_hist

        cleaned_hists = []
        for hist in all_best_hists:
            cleaned = pad_with_last_non_nan(hist)
            if cleaned is not None:
                cleaned_hists.append(cleaned)

        if not cleaned_hists:
            print(f"No valid histories after cleaning NaNs for experiment: {exp_name}")
            continue

        max_len = max(len(hist) for hist in cleaned_hists)
        padded_best = []
        for hist in cleaned_hists:
            padded = hist + [hist[-1]] * (max_len - len(hist))
            padded_best.append(padded)

        best_array = np.array(padded_best)
        best_fitness = np.max(best_array, axis=0)
        std_best = np.std(best_array, axis=0)
        epochs = range(1, max_len + 1)

        y_data = np.maximum(MAX_FITNESS + epsilon - best_fitness, epsilon)

        color = default_colors[exp_idx]
        marker = default_markers[exp_idx % len(default_markers)]

        ax.plot(
            epochs,
            -np.log10(y_data),
            color=color,
            linewidth=2,
            marker=marker,
            markersize=4,
            markevery=5,
            label=f"{exp_name}",
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")

    ax.set_ylabel("Target metric")

    ax.axhline(
        y=-np.log10(MAX_FITNESS + epsilon - 1),
        linestyle=":",
        color="gray",
        alpha=0.7,
        label=f"AlphaEvolve",
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def create_db_tree(db: ProgramDatabase) -> nx.DiGraph:
    """ """
    G = nx.DiGraph()
    for prog_id, prog in db.programs.items():
        G.add_node(prog_id)

    G.graph["best_prog_id"] = db.best_prog_id

    G.graph["roots"] = []
    for prog_id, prog in db.programs.items():
        if prog.parent_id:
            G.add_edge(prog.parent_id, prog_id)
        else:
            G.graph["roots"].append(prog_id)

    return G


def plot_program_tree(
    G: nx.DiGraph,
    node_labels: Dict,
    color_name: str,
    node_colors: Dict,
    num_islands: int,
    node_to_island: Dict,
    save_path: str = None,
    figsize=(12, 10),
    node_size=160,
    font_size=8,
    title=None,
) -> None:
    """ """

    fig, ax = plt.subplots(figsize=figsize)

    pos = graphviz_layout(G, prog="dot")

    values = list(node_colors.values())
    vmin = min(values)
    vmax = max(values)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    node_colors_cm = plt.get_cmap("autumn")(norm(np.array(values, dtype=float)))

    island_shapes = ["p", "8", "s", "o", "h"]

    best_id = G.graph["best_prog_id"]

    island_nodes = {}
    for node in G.nodes():
        if node != best_id:
            island_id = node_to_island.get(node, 0)
            if island_id not in island_nodes:
                island_nodes[island_id] = []
            island_nodes[island_id].append(node)

    legend_handles = []

    for island_id, nodes in island_nodes.items():
        if not nodes:
            continue

        shape = island_shapes[island_id % len(island_shapes)]

        island_node_colors = [node_colors_cm[list(G.nodes()).index(node)] for node in nodes]

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=nodes,
            node_color=island_node_colors,
            node_size=node_size,
            node_shape=shape,
            edgecolors="black",
            linewidths=1,
            alpha=0.8,
        )

        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=shape,
                color="w",
                markerfacecolor="gray",
                markersize=8,
                label=f"Island {island_id}",
            )
        )

    best_island_id = node_to_island.get(best_id, 0)
    best_shape = island_shapes[best_island_id % len(island_shapes)]
    best_node_color = node_colors_cm[list(G.nodes()).index(best_id)]

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=[best_id],
        node_color=[best_node_color],
        node_size=node_size * 1.5,
        node_shape=best_shape,
        alpha=0.9,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=[best_id],
        node_color="none",
        node_size=node_size * 1.5,
        node_shape=best_shape,
        edgecolors="blue",
        linewidths=2,
        alpha=1.0,
    )

    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=8, edge_color="gray", alpha=0.8)

    nx.draw_networkx_labels(G, pos, ax=ax, labels=node_labels, font_size=font_size)

    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker=best_shape,
            color="w",
            markerfacecolor="white",
            markeredgecolor="blue",
            markersize=10,
            markeredgewidth=2,
            label=f"Best solution",
        )
    )

    legend_handles.sort(key=lambda x: x.get_label())
    ax.legend(handles=legend_handles, loc="lower right", fontsize=font_size)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("autumn"), norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar_label = color_name
    cbar.set_label(cbar_label)

    if title:
        ax.set_title(title, fontweight="bold")
    else:
        ax.set_title(f"Solution evolution forest of best island ({num_islands} islands)")

    plt.tight_layout()
    ax.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=fig.dpi, bbox_inches="tight")
    plt.show()
