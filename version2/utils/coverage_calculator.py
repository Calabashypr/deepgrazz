from version2.graph import Graph, Node
from version2.utils.common_utils import *
from version2.model_gen.OperatorSet import operator_set


class CoverageCalculator:
    def __init__(self):
        r""""""

    def op_type_cover(self, sub_g: Graph):
        sub_s = set()
        for node in sub_g.nodes.values():
            sub_s.add(node.str_op)
        return len(sub_s) / len(operator_set)

    def op_num_cover(self, sub_g: Graph, g: Graph):
        return len(sub_g) / len(g)

    def edge_cover(self, sub_g: Graph, g: Graph):
        sub_edges = 0
        edges = 0
        for node in sub_g.nodes.values():
            sub_edges += len(node.to_nodes)
        for node in g.nodes.values():
            edges += len(node.to_nodes)
        return sub_edges / edges

    def get_cover(self, sub_g: Graph, g: Graph):
        return (self.op_type_cover(sub_g=sub_g) + self.op_num_cover(sub_g=sub_g, g=g) + self.edge_cover(sub_g=sub_g,
                                                                                                        g=g)) / 3
