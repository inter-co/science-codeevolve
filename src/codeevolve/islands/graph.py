# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file defines the graph structures for the distributed islands algorithm.
#
# ===--------------------------------------------------------------------------------------===#

import multiprocessing as mp
import multiprocessing.connection as mpc
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class PipeEdge:
    """Represents a directed communication edge between two islands.

    This class encapsulates a unidirectional pipe connection between two
    islands in a distributed evolutionary system, where one island can send
    data and the other can receive.

    Attributes:
        u: Source island ID (sender).
        v: Destination island ID (receiver).
        u_conn: Connection object for sending data from island u.
        v_conn: Connection object for receiving data at island v.
    """

    u: int
    v: int
    u_conn: mpc.Connection  # send only
    v_conn: mpc.Connection  # recv only


@dataclass
class IslandCommunicationData:
    """Contains communication data for an island in the distributed islands algorithm.

    This class stores the incoming and outgoing communication channels
    for an island in the distributed islands algorithm.

    Attributes:
        id: Unique identifier for the island.
        in_neigh: List of incoming pipe edges from neighboring islands (received migrants).
        out_neigh: List of outgoing pipe edges to neighboring islands (sent migrants).
    """

    id: int
    in_neigh: Optional[List[PipeEdge]]
    out_neigh: Optional[List[PipeEdge]]


def get_edge_list(num_islands: int, migration_topology: str) -> List[Tuple[int, int]]:
    """Generates edge list for island migration topology.

    Creates a list of directed edges representing the migration topology
    between islands in a distributed evolutionary system.

    Args:
        num_islands: Number of islands in the system.
        migration_topology: Name of the topology pattern to use.
            Supported topologies: 'directed_ring', 'ring', 'complete',
            'inward_star', 'outward_star', 'star', 'empty'.

    Returns:
        List of tuples representing directed edges (source, destination).

    Raises:
        ValueError: If migration_topology is not supported.
    """
    edge_list: List[Tuple[int, int]] = []
    if num_islands > 1:
        match migration_topology:
            case "directed_ring":
                edge_list = [(i, (i + 1) % num_islands) for i in range(num_islands)]
            case "ring":
                edge_list = [(i, (i + 1) % num_islands) for i in range(num_islands)] + [
                    ((i + 1) % num_islands, i) for i in range(num_islands)
                ]
            case "complete":
                for i in range(num_islands):
                    for j in range(i + 1, num_islands):
                        edge_list.append((i, j))
                        edge_list.append((j, i))
            case "inward_star":
                edge_list = [(i, 0) for i in range(1, num_islands)]
            case "outward_star":
                edge_list = [(0, i) for i in range(1, num_islands)]
            case "star":
                edge_list = [(0, i) for i in range(1, num_islands)] + [
                    (i, 0) for i in range(1, num_islands)
                ]
            case "empty":
                pass
            case _:
                raise ValueError(f"Unsupported migration topology: {migration_topology}.")

    return list(set(edge_list))


def get_pipe_graph(
    num_nodes: int, edge_list: List[Tuple[int, int]]
) -> Tuple[Dict[int, List[PipeEdge]], Dict[int, List[PipeEdge]]]:
    """Creates pipe communication graph from edge list.

    Converts a list of directed edges into actual pipe communication channels
    between islands, creating PipeEdge objects with multiprocessing pipes.

    Args:
        num_nodes: Number of nodes (islands) in the graph.
        edge_list: List of directed edges as (source, destination) tuples.

    Returns:
        A tuple containing:
            - Dictionary mapping node IDs to incoming PipeEdge objects
            - Dictionary mapping node IDs to outgoing PipeEdge objects
    """
    out_adj: Dict[int, List[PipeEdge]] = {u: [] for u in range(num_nodes)}
    in_adj: Dict[int, List[PipeEdge]] = {u: [] for u in range(num_nodes)}

    for u, v in edge_list:
        v_conn, u_conn = mp.Pipe(duplex=False)
        pedge = PipeEdge(u, v, u_conn, v_conn)

        out_adj[u].append(pedge)
        in_adj[v].append(pedge)

    return (in_adj, out_adj)


def setup_island_topology(
    num_islands: int, topology_type: str
) -> Tuple[Optional[List[PipeEdge]], Optional[List[PipeEdge]]]:
    """Configures communication topology between islands.

    Args:
        num_islands: Total number of islands.
        topology_type: Type of migration topology to create.

    Returns:
        Tuple of (incoming adjacency list, outgoing adjacency list) or (None, None) if no edges.

    Raises:
        SystemExit: If topology configuration fails.
    """
    try:
        edge_list: List[Tuple[int, int]] = get_edge_list(num_islands, topology_type)
    except Exception as err:
        print(f"Error creating migration topology: {err}")
        sys.exit(1)

    if not edge_list:
        return None, None

    in_adj: List[PipeEdge]
    out_adj: List[PipeEdge]
    in_adj, out_adj = get_pipe_graph(num_islands, edge_list)
    return in_adj, out_adj
