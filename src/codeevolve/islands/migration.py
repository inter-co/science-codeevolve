# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the migration phase of the distributed islands algorithm.
#
# ===--------------------------------------------------------------------------------------===#

import logging
import multiprocessing.synchronize as mps
import threading
from collections import defaultdict
from typing import DefaultDict, List, Optional

from codeevolve.database import Program
from codeevolve.islands.graph import IslandCommunicationData, PipeEdge


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
                except (EOFError, ConnectionError, OSError) as e:
                    logger.error(f"[RECV THREAD] Unable to receive migrant from {edge.u}: {e}")
        logger.info("[RECV THREAD] Received migrants.")


def sync_migrate(
    out_migrants: List[Program],
    isl_data: IslandCommunicationData,
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
