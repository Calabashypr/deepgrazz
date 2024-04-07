from version2.graph import Node, Graph
from version2.utils.common_utils import *
from version2.model_gen.OperatorSet import *
import json
from math import sqrt, log
import numpy as np


def json_to_graph(file_path='ResNet18.json'):
    with open(file_path, "r") as f:
        output_data = json.load(f)
        g = output_data['network']
    graph = Graph([i for i in range(len(g))])
    for node_ in g:
        node = Node(node_['id'])
        node.from_nodes = node_['from']
        node.to_nodes = node_['to']
        node.str_op = node_['name']
        node.params = node_['params']
        node.state = node_['state']
        node.in_degree = len(node.from_nodes)
        node.out_degree = len(node.to_nodes)
        if node.state == 'des':
            node.out_degree = 1
        graph.nodes[node.id] = node
    return graph


def child_potential(pre_node: Node, tmp_node: Node):
    N = pre_node.visit_count
    n = tmp_node.visit_count
    v = tmp_node.succ_count
    e = 1 / sqrt(2)
    return v / (n + 0.001) + e * sqrt(log(N + 1) / (n + 0.001))


def selection(g: Graph, tmp_id, terminal_condition, search_depth, path):
    if search_depth >= terminal_condition or \
            g.nodes[tmp_id].state == 'des' or \
            g.nodes[tmp_id].visit_count == 0:
        expend = False
        if g.nodes[tmp_id].state == 'des':
            expend = True
        g.nodes[tmp_id].visit_count += 1
        path.append(tmp_id)
        return tmp_id, path, expend

    g.nodes[tmp_id].visit_count += 1
    path.append(tmp_id)
    potential = []
    ids = []

    if len(g.nodes[tmp_id].to_nodes) == 0:
        print(f'{g.nodes[tmp_id]}')

    for i in g.nodes[tmp_id].to_nodes:
        child = g.nodes[i]
        potential.append(child_potential(g.nodes[tmp_id], child))
        ids.append(child.id)
    idx = np.argmax(potential)
    nxt_id = ids[idx]
    return selection(g, nxt_id, terminal_condition, search_depth + 1, path)


def expansion(g: Graph, path):
    new_id = len(g.nodes)
    new_node = Node(new_id)

    name_table = {*activation_operators, *seq_operators}
    p = randint(0, len(name_table) - 1)
    new_node.str_op = name_table[str(p)]
    new_node.set_state('des')
    return new_node


def get_nodes_from_path(g: Graph, path: list, new_node: Node):
    # path.append(new_node.id)
    sub_g = Graph(path)
    for i in range(len(path)):
        id = path[i]
        node_ = Node()
        node_.id = g.nodes[id].id
        node_.str_op = g.nodes[id].str_op
        node_.to_nodes = []
        node_.from_nodes = []
        node_.in_degree = 1
        if node_.str_op.lower() in ['add', 'max', 'min', 'maximum', 'minimum', 'concat', 'concatenate']:
            node_.in_degree = 2
        node_.out_degree = 1
        node_.params = g.nodes[id].params
        node_.input_shape = g.nodes[id].input_shape
        node_.output_shape = g.nodes[id].output_shape
        if i == 0:
            node_.state = 'src'
            node_.in_degree = 0
        sub_g.nodes[id] = node_
    for i in range(len(path) - 1):
        in_degree = g.nodes[path[i + 1]].in_degree
        for _ in range(in_degree):
            sub_g.add_edge(path[i], path[i + 1])

    if new_node.id != -1:
        new_node.from_nodes = []
        new_node.to_nodes = []
        new_node.state = 'des'
        new_node.visit_count = 0
        new_node.succ_count = 0
        new_node.in_degree = 1
        new_node.out_degree = 0
        sub_g.nodes[new_node.id] = new_node
        sub_g.add_edge(path[-1], new_node.id)
    else:
        sub_g.nodes[path[-1]].state = 'des'

    return sub_g, path


def block_chooser(g: Graph, terminal_condition):
    src = g.get_src().id
    tc1 = 10 if terminal_condition < 0 else terminal_condition
    path_ = []
    # path_.append(src)
    leaf_id, path, expend = selection(g, src, tc1, 0, path_)
    # print(path, expend)
    tc2 = 1
    new_node = Node()
    if expend:
        new_node = expansion(g, path)
    sub_g, path = get_nodes_from_path(g, path, new_node)
    return sub_g


def back_propagation(g: Graph, nodes_path):
    """"""
    for id in nodes_path:
        g.nodes[id].succ_count += 1
