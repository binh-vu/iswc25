from __future__ import annotations

from copy import copy
from typing import Optional, cast

from gp.semanticmodeling.postprocessing.cgraph import (
    CGEdge,
    CGEdgeTriple,
    CGNode,
    CGraph,
)
from gp.semanticmodeling.postprocessing.common import (
    add_context,
    ensure_valid_statements,
)
from gp.semanticmodeling.postprocessing.config import PostprocessingConfig
from graph.retworkx.api import dag_longest_path
from sm.dataset import FullTable
from steiner_tree.bank.solver import PSEUDO_ROOT_ID, BankSolver
from steiner_tree.bank.struct import BankGraph, Solution


class SteinerTreeV100:
    def __init__(
        self,
        table: FullTable,
        cg: CGraph,
        edge_probs: dict[CGEdgeTriple, float],
        threshold: float,
        additional_terminal_nodes: Optional[list[str]] = None,
    ):
        self.table = table
        self.cg = cg
        self.edge_probs = edge_probs
        self.threshold = threshold

        self.top_k_st = 50
        self.top_k_path = 50

        # extra terminal nodes that the tree should have, usually used in
        # interactive modeling where users add some entity nodes in their model
        self.additional_terminal_nodes = additional_terminal_nodes

    def get_result(self) -> CGraph:
        """Select edges that forms a tree"""
        edge_probs = {e: p for e, p in self.edge_probs.items() if p >= self.threshold}

        # first step is to remove dangling statements and standalone nodes
        subcg = self.cg.subgraph_from_edge_triples(edge_probs.keys())
        subcg.remove_dangling_statement()
        subcg.remove_standalone_nodes()

        # terminal nodes are columns node, entity nodes are adding later
        terminal_nodes = {u.id for u in subcg.iter_nodes() if u.is_column_node}
        if self.additional_terminal_nodes is not None:
            terminal_nodes.update(self.additional_terminal_nodes)
        if len(terminal_nodes) == 0:
            # if the tree does not have any column, we essentially don't predict anything
            # so we return an empty graph
            return CGraph()

        # add pseudo root
        edge_weights = {e: 1.0 / p for e, p in edge_probs.items()}  # p > 0.5

        solver = BankSteinerTree(
            original_graph=subcg,
            terminal_nodes=terminal_nodes,
            top_k_st=self.top_k_st,
            top_k_path=self.top_k_path,
            weight_fn=lambda e: edge_weights[e.source, e.target, e.key],
            solution_cmp_fn=self.compare_solutions,
            invalid_roots={u.id for u in subcg.nodes() if u.is_statement_node},
            allow_shorten_graph=False,
        )
        trees, _solutions = solver.run()
        if len(trees) == 0:
            return CGraph()

        tree = cast(CGraph, trees[0])
        tree.remove_dangling_statement()

        if PostprocessingConfig.INCLUDE_CONTEXT:
            add_context(subcg, tree, edge_probs)

        # add back statement property if missing into to ensure a correct model
        # we cannot do this because we do not always have statement main property
        # ensure_valid_statements(subcg, tree, create_if_not_exists=False)

        # fmt: off
        # from graph.viz.graphviz import draw
        # draw(graph=tree, filename="/tmp/graphviz/st204.png", **CGGraph.graphviz_props())
        # draw(graph=tree, filename="/tmp/graphviz/g25.png", **CGGraph.graphviz_props())
        # fmt: on

        return tree

    def compare_solutions(self, a: Solution, b: Solution) -> int:
        """Comparing two solutions, -1 (smaller) means a better solution -- we are solving minimum steiner tree"""
        a_weight = a.weight / max(a.num_edges, 1)
        b_weight = b.weight / max(b.num_edges, 1)

        if a_weight < b_weight:
            return -1
        if a_weight > b_weight:
            return 1
        # equal weight, prefer the one with shorter depth
        # TODO: there is a case where a statement has a qualifier
        # it makes the tree without source node shorter than the one with source node
        # however, the one with the source node is better...
        # so the steiner tree should have an option to say like if we pick this path, we must
        # select the source node to avoid these issue...

        # for a dirty fix, perhaps we can say, the one with a dangling statement is infinitly bad
        if not hasattr(a, "has_no_source_statement"):
            has_no_source_statement = False
            for u in a.graph.iter_nodes():
                if u.id == PSEUDO_ROOT_ID:
                    continue
                cgu = self.cg.get_node(u.id)
                if cgu.is_statement_node and a.graph.in_degree(u.id) == 0:
                    has_no_source_statement = True
                    break
            setattr(a, "has_no_source_statement", has_no_source_statement)

        if not hasattr(b, "has_no_source_statement"):
            has_no_source_statement = False
            for u in b.graph.iter_nodes():
                if u.id == PSEUDO_ROOT_ID:
                    continue
                cgu = self.cg.get_node(u.id)
                if cgu.is_statement_node and b.graph.in_degree(u.id) == 0:
                    has_no_source_statement = True
                    break
            setattr(b, "has_no_source_statement", has_no_source_statement)

        a_has_no_source = getattr(a, "has_no_source_statement")
        b_has_no_source = getattr(b, "has_no_source_statement")
        if not a_has_no_source and b_has_no_source:
            return -1
        if a_has_no_source and not b_has_no_source:
            return 1

        if not hasattr(a, "depth"):
            setattr(a, "depth", len(dag_longest_path(a.graph)))
        if not hasattr(b, "depth"):
            setattr(b, "depth", len(dag_longest_path(b.graph)))
        return getattr(a, "depth") - getattr(b, "depth")


class BankSteinerTree(BankSolver[CGNode, CGEdge]):
    ADD_MISSING_STATEMENT_PROPS = True

    def add_missing_statement(self, g: BankGraph):
        if not self.ADD_MISSING_STATEMENT_PROPS:
            return False

        update_graph = False
        cg = cast(CGraph, self.original_graph)
        for uprime in g.iter_nodes():
            if uprime.id == PSEUDO_ROOT_ID:
                continue
            u = cg.get_node(uprime.id)
            if u.is_statement_node:
                (inedge,) = cg.in_edges(u.id)
                if all(outedge.key != inedge.key for outedge in g.out_edges(u.id)):
                    # no statement property, add it back
                    # sometimes the baseline don't have main statement
                    tmp = [
                        outedge
                        for outedge in self.graph.out_edges(u.id)
                        if outedge.key == inedge.key
                    ]
                    if len(tmp) == 0:
                        continue
                    # if there are multiple main prop, we select edge that has lower weight
                    main_prop = min(tmp, key=lambda e: e.weight)
                    if not g.has_node(main_prop.target):
                        target_id = g.add_node(
                            copy(self.graph.get_node(main_prop.target))
                        )
                        assert target_id == main_prop.target
                    g.add_edge(main_prop.clone())
                    update_graph = True
        return update_graph

    def _bank_graph_postconstruction(
        self, g: BankGraph, n_attrs: int, update_graph: bool = False
    ) -> bool:
        # this function does not handle the case where statement main prop is added to the graph
        # but the main prop is the same as one of the existing qualifier node
        # later, because of multiple paths between two nodes, either the main prop or qualifier
        # will be removed, and the main prop is removed, so we come back to the situation of
        # having a statement without main prop
        update_graph = self.add_missing_statement(g) or update_graph
        return super()._bank_graph_postconstruction(g, n_attrs, update_graph)
