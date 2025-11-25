# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements a distributed islands algorithm.
#
# ===--------------------------------------------------------------------------------------===#

from typing import List, Tuple, Dict, Optional, DefaultDict

from collections import defaultdict
from dataclasses import dataclass
import threading
import multiprocessing as mp
import multiprocessing.sharedctypes as mpsct
import multiprocessing.synchronize as mps
import multiprocessing.connection as mpc
import logging

from codeevolve.database import Program


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
class IslandData:
    """Contains communication data for an island in a distributed system.

    This class stores the incoming and outgoing communication channels
    for an island in an island-based evolutionary algorithm.

    Attributes:
        id: Unique identifier for the island.
        in_neigh: List of incoming pipe edges from neighboring islands.
        out_neigh: List of outgoing pipe edges to neighboring islands.
    """

    id: int
    in_neigh: Optional[List[PipeEdge]]
    out_neigh: Optional[List[PipeEdge]]


@dataclass
class GlobalBestProg:
    """Tracks the globally best program across all islands using shared memory.

    This class maintains synchronized access to information about the best
    program found across all islands in a distributed evolutionary system.

    Attributes:
        fitness: Synchronized fitness value of the best program.
        iteration_found: Synchronized iteration number when best program was found.
        island_found: Synchronized ID of the island that found the best program.
    """

    fitness: mpsct.Synchronized
    iteration_found: mpsct.Synchronized
    island_found: mpsct.Synchronized

    def __repr__(self):
        """Returns a string representation of the global best program.

        Returns:
            A formatted string showing fitness, iteration found, and island found.
        """
        return (
            f"{self.__class__.__name__}"
            "("
            f"fitness={self.fitness.value:.8f},"
            f"iteration_found={self.iteration_found.value},"
            f"island_found={self.island_found.value}"
            ")"
        )


@dataclass
class GlobalData:
    """Contains shared data structures for coordinating distributed islands.

    This class encapsulates all shared memory objects and synchronization
    primitives needed for coordinating multiple islands in a distributed
    evolutionary algorithm.

    Attributes:
        best_sol: Information about the globally best program found.
        early_stop_counter: Counter for consecutive iterations without improvement.
        early_stop_aux: Auxiliary counter for early stopping coordination.
        lock: Mutex for protecting shared data access.
        barrier: Synchronization barrier for coordinating island phases.
        log_queue: Queue for collecting log messages from all islands.
    """

    best_sol: GlobalBestProg
    early_stop_counter: mpsct.Synchronized
    early_stop_aux: mpsct.Synchronized
    lock: mps.Lock
    barrier: mps.Barrier
    log_queue: mp.Queue


def early_stopping_check(
    island_id: int,
    num_islands: int,
    improved_local_fitness: bool,
    global_data: GlobalData,
    logger: logging.Logger,
) -> None:
    """Coordinates early stopping decision across all islands.

    This function implements a distributed early stopping mechanism where
    all islands must report no improvement before the early stopping counter
    is incremented. Uses barriers to ensure all islands participate in the decision.

    Args:
        island_id: ID of the current island.
        num_islands: Total number of islands in the system.
        improved_local_fitness: Whether this island improved its best fitness.
        global_data: Shared data structures for coordination.
        logger: Logger instance for this island.
    """
    if not improved_local_fitness:
        with global_data.lock:
            # indicates if an island didnt improve locally
            global_data.early_stop_aux.value += 1

    logger.info("Waiting for all islands to report progress...")
    global_data.barrier.wait()
    logger.info("All islands synced.")

    with global_data.lock:
        # first to arrive is the leader, makes the early stop check,
        # and then sets the aux to -1 so no other island can do the same
        if global_data.early_stop_aux.value != -1:
            if global_data.early_stop_aux.value == num_islands:
                global_data.early_stop_counter.value += 1
            else:
                global_data.early_stop_counter.value = 0

            global_data.early_stop_aux.value = (
                -1
            )  # flag for other islands to not repeat the above code

    logger.info("Waiting for other islands to finish early stopping check...")
    global_data.barrier.wait()
    logger.info("All islands synced.")

    global_data.early_stop_aux.value = 0  # reset to zero


# islands graph


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
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
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
    out_adj: Dict[int, List[int]] = {u: [] for u in range(num_nodes)}
    in_adj: Dict[int, List[int]] = {u: [] for u in range(num_nodes)}

    for u, v in edge_list:
        v_conn, u_conn = mp.Pipe(duplex=False)
        pedge = PipeEdge(u, v, u_conn, v_conn)

        out_adj[u].append(pedge)
        in_adj[v].append(pedge)

    return (in_adj, out_adj)


# migration
## TODO: async migration without barriers


def send_migrants(
    out_neigh: Optional[List[PipeEdge]],
    migrants: List[Program],
    logger: logging.Logger,
) -> None:
    """Sends migrant programs to neighboring islands.

    This function runs in a separate thread to send migrant programs
    to all outgoing neighbor islands through pipe connections.

    Args:
        out_neigh: List of outgoing pipe edges to neighbor islands.
        migrants: List of Program objects to send as migrants.
        logger: Logger instance for this thread.
    """
    if out_neigh:
        logger.info("[SEND THREAD] Sending migrants to neighbors...")

        for edge in out_neigh:
            for migrant in migrants:
                edge.u_conn.send(migrant)
                logger.info(f"[SEND THREAD] Sent {migrant} to {edge.v}.")
        logger.info("[SEND THREAD] Migrants sent.")


def recv_migrants(
    in_neigh: Optional[List[PipeEdge]],
    island2count: DefaultDict[int, int],
    in_migrants: List[Program],
    logger: logging.Logger,
) -> None:
    """Receives migrant programs from neighboring islands.

    This function runs in a separate thread to receive migrant programs
    from all incoming neighbor islands.

    Args:
        in_neigh: List of incoming pipe edges from neighbor islands.
        island2count: Mapping of island IDs to expected number of migrants.
        in_migrants: Empty list used to store incoming migrants.
        logger: Logger instance for this thread.

    Returns:
        List of received migrants
    """
    if in_neigh:
        logger.info("[RECV THREAD] Receiving migrants from neighbors...")
        for edge in in_neigh:
            for _ in range(island2count[edge.u]):
                try:
                    migrant: Program = edge.v_conn.recv()
                    in_migrants.append(migrant)
                    logger.info(f"[RECV THREAD] Received {migrant} from {edge.u}.")
                except:
                    logger.error(f"[RECV THREAD] Unable to receive migrant from {edge.u}.")
        logger.info("[RECV THREAD] Received migrants.")


def sync_migrate(
    out_migrants: List[Program],
    isl_data: IslandData,
    barrier: mps.Barrier,
    logger: logging.Logger,
) -> List[Program]:
    """Performs synchronized migration between islands.

    This function coordinates the migration of programs between islands
    using barriers to ensure all islands participate simultaneously.

    Args:
        out_migrants: List of programs to migrate.
        isl_data: Island communication data including neighbor connections.
        barrier: Synchronization barrier for coordinating migration phases.
        logger: Logger instance for this island.

    Returns:
        List of received programs.
    """
    in_migrants: List[Program] = []

    island2count: DefaultDict[int, int] = defaultdict(int)

    logger.info("Waiting for other islands to start migration...")
    barrier.wait()
    logger.info("Migration started.")

    if isl_data.out_neigh:
        logger.info(f"Informing other islands: {len(out_migrants)} migrants being sent.")
        for edge in isl_data.out_neigh:
            edge.u_conn.send(len(out_migrants))

    barrier.wait()

    if isl_data.in_neigh:
        logger.info("Receiving migrant counts from each neighbor.")
        for edge in isl_data.in_neigh:
            island2count[edge.u] = edge.v_conn.recv()
            logger.info(f"Island {edge.u} is sending {island2count[edge.u]} migrants.")

    barrier.wait()

    logger.info("Starting SEND and RECV threads...")
    send_thread = threading.Thread(
        target=send_migrants, args=(isl_data.out_neigh, out_migrants, logger)
    )
    recv_thread = threading.Thread(
        target=recv_migrants, args=(isl_data.in_neigh, island2count, in_migrants, logger)
    )

    send_thread.start()
    recv_thread.start()

    send_thread.join()
    recv_thread.join()
    logger.info("Threads finished.")

    logger.info("Waiting for other islands to finish migration...")
    barrier.wait()
    logger.info("Migration finished.")

    return in_migrants
