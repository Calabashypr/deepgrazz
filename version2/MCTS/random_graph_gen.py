from version2.graph import Node, Graph
from version2.utils.common_utils import *
import random


def random_graph(g: Graph, n):
    r"""
    select n nodes
    :param g:
    :param n:
    :return:    sub_g a sub graph from origin graph
    """
    ids = []
    for i in range(n):
        ids.append(get_random_num([id for id in g.nodes.keys()]))
    sub_g = Graph([id for id in range(n)])
    pre_id = -1
    for i in range(len(ids)):
        id_sub = i
        id_g = ids[i]
        node_g = g.nodes[id_g]
        sub_g.nodes[id_sub].id = id_sub
        sub_g.nodes[id_sub].origin_id = id_g
        sub_g.nodes[id_sub].str_op = node_g.str_op
        sub_g.nodes[id_sub].params = node_g.params
        sub_g.nodes[id_sub].in_degree = node_g.in_degree
        sub_g.nodes[id_sub].out_degree = node_g.out_degree
        sub_g.nodes[id_sub].from_nodes = []
        sub_g.nodes[id_sub].to_nodes = []
        if i == 0:
            sub_g.nodes[id_sub].state = 'src'
            sub_g.nodes[id_sub].is_src = True
            pre_id = id_sub
        elif i == len(ids) - 1:
            sub_g.nodes[id_sub].state = 'des'
            sub_g.nodes[id_sub].is_des = True
            # for _ in range(sub_g.nodes[id_sub].in_degree):
            sub_g.add_edge(pre_id, id_sub)
            pre_id = -1
        else:
            sub_g.nodes[id_sub].state = 'none'
            # for _ in range(sub_g.nodes[id_sub].in_degree):
            sub_g.add_edge(pre_id, id_sub)
            pre_id = id_sub
    return sub_g
