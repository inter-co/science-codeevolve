# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements unit tests for island communication graph structures.
#
# ===--------------------------------------------------------------------------------------===#

from typing import Dict, List, Optional, Tuple

import pytest

from codeevolve.islands.graph import (
    IslandCommunicationData,
    PipeEdge,
    get_edge_list,
    get_pipe_graph,
    setup_island_topology,
)

# ---------------------------------------------------------------------------
# get_edge_list
# ---------------------------------------------------------------------------


class TestGetEdgeList:
    """Test suite for the get_edge_list function."""

    def test_directed_ring(self):
        """Tests directed ring topology creates correct edges."""
        edges: List[Tuple[int, int]] = get_edge_list(3, "directed_ring")
        assert (0, 1) in edges
        assert (1, 2) in edges
        assert (2, 0) in edges
        assert len(edges) == 3

    def test_ring(self):
        """Tests bidirectional ring topology."""
        edges: List[Tuple[int, int]] = get_edge_list(3, "ring")
        assert (0, 1) in edges
        assert (1, 0) in edges
        assert len(edges) == 6

    def test_complete(self):
        """Tests complete graph topology."""
        edges: List[Tuple[int, int]] = get_edge_list(3, "complete")
        assert (0, 1) in edges
        assert (1, 0) in edges
        assert (0, 2) in edges
        assert (2, 0) in edges
        assert (1, 2) in edges
        assert (2, 1) in edges
        assert len(edges) == 6

    def test_inward_star(self):
        """Tests inward star topology (all islands send to island 0)."""
        edges: List[Tuple[int, int]] = get_edge_list(4, "inward_star")
        for i in range(1, 4):
            assert (i, 0) in edges
        assert len(edges) == 3

    def test_outward_star(self):
        """Tests outward star topology (island 0 sends to all)."""
        edges: List[Tuple[int, int]] = get_edge_list(4, "outward_star")
        for i in range(1, 4):
            assert (0, i) in edges
        assert len(edges) == 3

    def test_star(self):
        """Tests bidirectional star topology."""
        edges: List[Tuple[int, int]] = get_edge_list(4, "star")
        for i in range(1, 4):
            assert (0, i) in edges
            assert (i, 0) in edges
        assert len(edges) == 6

    def test_empty(self):
        """Tests empty topology returns no edges."""
        edges: List[Tuple[int, int]] = get_edge_list(3, "empty")
        assert len(edges) == 0

    def test_single_island(self):
        """Tests that single-island setup returns no edges."""
        edges: List[Tuple[int, int]] = get_edge_list(1, "directed_ring")
        assert len(edges) == 0

    def test_unsupported_topology(self):
        """Tests that unsupported topology raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            get_edge_list(3, "mesh")

    def test_no_duplicate_edges(self):
        """Tests that edge lists contain no duplicates."""
        edges: List[Tuple[int, int]] = get_edge_list(4, "complete")
        assert len(edges) == len(set(edges))


# ---------------------------------------------------------------------------
# get_pipe_graph
# ---------------------------------------------------------------------------


class TestGetPipeGraph:
    """Test suite for the get_pipe_graph function."""

    def test_pipe_graph_basic(self):
        """Tests pipe graph creation from simple edge list."""
        edge_list: List[Tuple[int, int]] = [(0, 1), (1, 0)]
        in_adj: Dict[int, List[PipeEdge]]
        out_adj: Dict[int, List[PipeEdge]]
        in_adj, out_adj = get_pipe_graph(2, edge_list)

        assert len(out_adj[0]) == 1
        assert len(in_adj[1]) == 1
        assert out_adj[0][0].v == 1
        assert in_adj[1][0].u == 0

    def test_pipe_graph_empty(self):
        """Tests pipe graph with no edges."""
        in_adj: Dict[int, List[PipeEdge]]
        out_adj: Dict[int, List[PipeEdge]]
        in_adj, out_adj = get_pipe_graph(3, [])
        for i in range(3):
            assert len(in_adj[i]) == 0
            assert len(out_adj[i]) == 0

    def test_pipe_graph_connections_work(self):
        """Tests that pipe connections can actually send and receive data."""
        edge_list: List[Tuple[int, int]] = [(0, 1)]
        in_adj: Dict[int, List[PipeEdge]]
        out_adj: Dict[int, List[PipeEdge]]
        in_adj, out_adj = get_pipe_graph(2, edge_list)

        test_data: str = "hello"
        out_adj[0][0].u_conn.send(test_data)
        received: str = in_adj[1][0].v_conn.recv()
        assert received == test_data


# ---------------------------------------------------------------------------
# setup_island_topology
# ---------------------------------------------------------------------------


class TestSetupIslandTopology:
    """Test suite for the setup_island_topology function."""

    def test_setup_directed_ring(self):
        """Tests setup returns valid adjacency lists for directed ring."""
        in_adj: Optional[List[PipeEdge]]
        out_adj: Optional[List[PipeEdge]]
        in_adj, out_adj = setup_island_topology(3, "directed_ring")
        assert in_adj is not None
        assert out_adj is not None

    def test_setup_empty_returns_none(self):
        """Tests that empty topology returns (None, None)."""
        in_adj: Optional[List[PipeEdge]]
        out_adj: Optional[List[PipeEdge]]
        in_adj, out_adj = setup_island_topology(3, "empty")
        assert in_adj is None
        assert out_adj is None

    def test_setup_single_island(self):
        """Tests that single island returns (None, None) for any topology."""
        in_adj: Optional[List[PipeEdge]]
        out_adj: Optional[List[PipeEdge]]
        in_adj, out_adj = setup_island_topology(1, "directed_ring")
        assert in_adj is None
        assert out_adj is None


# ---------------------------------------------------------------------------
# IslandCommunicationData
# ---------------------------------------------------------------------------


class TestIslandCommunicationData:
    """Test suite for the IslandCommunicationData dataclass."""

    def test_creation(self):
        """Tests that IslandCommunicationData can be created."""
        data: IslandCommunicationData = IslandCommunicationData(id=0, in_neigh=None, out_neigh=None)
        assert data.id == 0
        assert data.in_neigh is None
        assert data.out_neigh is None
