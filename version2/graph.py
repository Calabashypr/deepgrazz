import copy
from version2.utils.common_utils import *
from version2.model_gen.OperatorSet import insert_operators
import random
import os
import json
import numpy as np


class Node:
    def __init__(self, index=-1):
        super().__init__()
        self.from_nodes = []
        self.to_nodes = []
        self.op = None
        self.id = index
        self.origin_id = -1
        self.state = 'none'
        self.is_des = False
        self.is_src = False
        self.visit_count = 0
        self.succ_count = 0
        self.str_op = 'empty'
        self.params = {}
        self.input_shape = []
        self.output_shape = []
        self.in_degree = len(self.from_nodes)
        self.out_degree = len(self.to_nodes)

    def add_from(self, node):
        self.from_nodes.append(node.id)

    def get_from(self, index=-1):
        if index == -1:
            return self.from_nodes
        return self.from_nodes[index]

    def del_from(self, index):
        if index in self.from_nodes:
            self.from_nodes.remove(index)

    def add_to(self, node):
        self.to_nodes.append(node.id)

    def get_to(self, index=-1):
        if index == -1:
            return self.to_nodes
        return self.to_nodes[index]

    def del_to(self, index):
        if index in self.to_nodes:
            self.to_nodes.remove(index)

    def set_op(self, operator, state='none', str_op='empty'):
        self.op = operator
        self.str_op = str_op
        self.state = state
        if self.op is not None and str_op == 'empty':
            op_name = str(type(self.op))
            op_name = op_name[:-2]
            dot_ip = op_name.rindex('.')
            op_name = op_name[dot_ip + 1:]
            self.str_op = op_name.lower()

    def get_op(self):
        return self.op

    def run(self, input_data):
        return self.op(input_data)

    def get_id(self):
        return self.id

    def set_id(self, index):
        self.id = index

    def set_state(self, state):
        self.state = state
        if state == 'des':
            self.is_des = True
        if state == 'src':
            self.is_src = True

    def set_input_shape(self, input_shape=None):
        if input_shape is None:
            input_shape = [None, None]
        self.input_shape = input_shape

    def get_input_shape(self):
        return self.input_shape

    def set_output_shape(self, output_shape):
        self.output_shape = output_shape

    def get_output_shape(self):
        return self.output_shape

    def __str__(self) -> str:
        return f'id:{self.id},op:{self.str_op},to:{self.to_nodes},from:{self.from_nodes},' \
               f'state:{self.state},in_degree:{self.in_degree},out_degree:{self.out_degree},' \
               f'input_shape:{self.input_shape},output_shape:{self.output_shape},' \
               f'params:{self.params},({self.origin_id})'

    def __hash__(self) -> int:
        return hash(self.str_op)

    def __eq__(self, o) -> bool:
        return hash(self.str_op) == hash(o.str_op)


class Graph:

    def __init__(self, nums: list):
        self.nodes = dict(zip([id for id in nums], [Node(id) for id in nums]))

    def get_node(self, index):
        return self.nodes[index]

    def set_op(self, index, operator, state='none'):
        self.nodes[index].set_op(operator, state)

    def set_ops(self, indexes, operator, state='none'):
        for index in indexes:
            self.set_op(index, operator, state)

    def add_edge(self, src: int, des: int):
        r"""add edge from node[src] to node[des] without fix the graph"""
        self.nodes[src].add_to(self.nodes[des])
        self.nodes[des].add_from(self.nodes[src])

    def insert_edge(self, src, des):
        r"""
        insert an edge from node[src] to node[des], and fix the graph
        :return:
        """
        op_add = Node()
        op_add.str_op = 'add'
        op_add.in_degree = 2
        op_add.out_degree = 1
        op_add.id = first_missing_positive([index for index in self.nodes.keys()])
        pre_id_ = random.randint(0, len(self.nodes[des].from_nodes) - 1)
        pre = self.nodes[des].from_nodes[pre_id_]
        op_add.output_shape = self.nodes[pre].output_shape
        op_add.input_shape.append(op_add.output_shape)
        self.insert_node_between(op_add, pre, des)
        self.add_edge(src, op_add.id)
        # self.del_edge(pre, des)
        self.fix_shape(op_add.id)

    def add_edges(self, src, des_nodes):
        for des in des_nodes:
            self.add_edge(src, des)

    def add_node(self, src: Node, des: Node):
        r"""add node des after node src, des doesn't need to link src.to_nodes"""
        self.nodes[des.id] = des
        self.add_edge(src.id, des.id)

    def append_node(self, src: Node, des: Node):
        r"""
        append node des after node src, src link des and des link src.to_nodes
        illustration:
        before append, src--> src_link1, src_link2
        after append, src --> des--> src_link1, src_link2
        """
        des.in_degree = 1
        des.out_degree = src.out_degree
        src.out_degree = 1

        for to_node in src.to_nodes:
            self.add_edge(des.id, to_node)
        for to_node in src.to_nodes:
            self.del_edge(src.id, to_node)
        self.add_node(src, des)

    def insert_node(self, src: int, node: Node):
        r"""
        insert node between node[src] and des(random selected from node[src].to_nodes
        :param src: insert node after src
        :param node: node to be inserted after src
        :return:
        """
        print(f'insert_node: insert {node} after {src}')

        self.nodes[node.id] = node

        to_nodes_id_list = self.nodes[src].to_nodes
        if len(to_nodes_id_list) == 0:
            # print(f'debug to_nodes_id_list,graph:')
            # self.display()
            # print(f'debug to_nodes_id_list,node:{self.nodes[src]}')
            id = random.randint(0, len(to_nodes_id_list) - 1)
        else:
            id = random.randint(0, len(to_nodes_id_list) - 1)
        des = self.nodes[src].to_nodes[id]
        self.add_edge(src, node.id)
        self.add_edge(node.id, des)
        self.del_edge(src, des)

    def insert_node_between(self, node: Node, id1, id2):
        r"""insert node between node[id1] and node[id2] without fix the graph"""
        self.nodes[node.id] = node
        self.del_edge(id1, id2)
        self.add_edge(id1, node.id)
        self.add_edge(node.id, id2)

    def del_edge(self, src: int, des: int):
        r"""delete edge from node[src] to node[des]"""
        self.nodes[src].del_to(des)
        self.nodes[des].del_from(src)

    def remove_edge(self, src: int, des: int):
        r"""
        remove edge from node[src] to node[des] in logically way
        :param src:
        :param des:
        :return:
        """
        id = first_missing_positive(list(self.nodes.keys()))
        op_remove_edge = Node(id)
        op_remove_edge.str_op = 'remove_edge_operator'
        op_remove_edge.in_degree = op_remove_edge.out_degree = 1
        op_remove_edge.input_shape.append(self.nodes[src].output_shape)
        op_remove_edge.output_shape = self.nodes[src].output_shape
        self.insert_node_between(op_remove_edge, src, des)

    def del_node(self, index):

        print(f'debug del_node, delete node {index}')

        if index not in self.nodes.keys():
            print(f'graph does not have node:{index}, delete node abolish ')
            return

        self.nodes[index].params = {}

        if self.nodes[index].in_degree > 1:
            self.nodes[index].str_op = 'empty_merge_operator'
        else:
            self.nodes[index].str_op = 'empty_seq_operator'
        self.nodes[index].output_shape = self.nodes[index].input_shape[0]
        for nxt in self.nodes[index].to_nodes:
            self.fix_shape(nxt)
        self.fix_shape(index)

    def fix_shape(self, id):
        r"""
        fix input_shape and output_shape for node[id] and it's pre_nodes
        take a-->b as example, if the output_shape of a doesn't match input_shape
        of b, it's likely to fix the graph as a-->flatten-->concat/slice-->b by
        fix_shape(b)
        """
        node = self.nodes[id]
        # output_shape = node.output_shape
        input_shapes = node.input_shape

        for i in range(node.in_degree):
            ''''''
            pre = node.from_nodes[i]
            pre_node = self.nodes[pre]
            # print(f'debug fix pre_node:{pre_node}')
            pre_output_shape = pre_node.output_shape
            batch_size = pre_output_shape[0]
            input_shape = input_shapes[0]  # a little problem here
            if pre_output_shape == input_shape:
                continue
            pre_output_nums = get_element_nums(pre_output_shape)
            input_nums = get_element_nums(input_shape)
            det = pre_output_nums - input_nums

            op_flatten = Node()
            op_flatten.str_op = 'flatten'
            op_flatten.in_degree = op_flatten.out_degree = 1
            op_flatten.input_shape.append(pre_node.output_shape)
            op_flatten.output_shape = [batch_size, pre_output_nums // batch_size]
            op_reshape = Node()
            op_reshape.str_op = 'reshape'
            op_reshape.in_degree = op_reshape.out_degree = 1
            op_reshape.output_shape = input_shape
            op_reshape.params['size'] = op_reshape.output_shape

            if det == 0:
                op_reshape.input_shape.append(pre_node.output_shape)
                op_reshape.id = first_missing_positive([index for index in self.nodes.keys()])
                self.insert_node_between(op_reshape, pre, id)
            elif det < 0:
                op_flatten.id = first_missing_positive([index for index in self.nodes.keys()])
                self.insert_node_between(op_flatten, pre, id)

                op_pad = Node()
                op_pad.str_op = 'pad'
                op_pad.in_degree = 1
                op_pad.out_degree = 1
                op_pad.input_shape.append(op_flatten.output_shape)
                op_pad.output_shape = [batch_size, input_nums // batch_size]
                print(f"op_pad output_shape[-1]:{op_pad.output_shape[-1]}")
                print(f"op_flatten output_shape[-1]:{op_flatten.output_shape[-1]}")
                zeros_len = op_pad.output_shape[-1] - op_flatten.output_shape[-1]
                op_pad.params['pad'] = (0, zeros_len)
                print(f"op_pad.params:{op_pad.params['pad']}")
                op_pad.id = first_missing_positive([index for index in self.nodes.keys()])
                self.insert_node_between(op_pad, op_flatten.id, id)

                op_reshape.input_shape.append(op_pad.output_shape)
                op_reshape.id = first_missing_positive([index for index in self.nodes.keys()])
                self.insert_node_between(op_reshape, op_pad.id, id)
            else:
                '''det>0'''
                op_flatten.id = first_missing_positive([index for index in self.nodes.keys()])
                self.insert_node_between(op_flatten, pre, id)

                op_slice = Node()
                op_slice.str_op = 'slice'
                op_slice.in_degree = op_slice.out_degree = 1
                op_slice.input_shape.append(op_flatten.output_shape)
                op_slice.output_shape = [batch_size, input_nums // batch_size]
                op_slice.params['size'] = [-1, op_slice.output_shape[-1]]
                op_slice.id = first_missing_positive([index for index in self.nodes.keys()])
                self.insert_node_between(op_slice, op_flatten.id, id)

                op_reshape.input_shape.append(op_slice.output_shape)
                op_reshape.id = first_missing_positive([index for index in self.nodes.keys()])
                self.insert_node_between(op_reshape, op_slice.id, id)

    def dup_node(self, index):
        r"""
        dup a node between node[index] and des (des will be random selected from node[index].to_nodes
        :param index:
        :return:
        """
        op_set = [*insert_operators]
        random_id = random.randint(0, len(op_set) - 1)
        random_op = op_set[random_id]
        node_ = Node()
        node_.in_degree = node_.out_degree = 1
        node_.str_op = random_op
        nums = [node.id for node in self.nodes.values()]
        id = first_missing_positive(nums)
        print(f'debug dub_node, id_nums:{nums}, get new id:{id}, name:{node_.str_op}')
        node_.id = id
        node_.input_shape.append(self.nodes[index].output_shape)

        first_input_shape = node_.input_shape[0]
        batch_size = first_input_shape[0]
        if node_.str_op.lower() == 'conv2d':
            node_.params['in_channels'] = first_input_shape[1]
            ch_set = [1, 3, 4, 8, 16, 32, 64, 128, 256]
            in_ch = get_random_num(ch_set)
            out_ch = get_random_num(ch_set)
            height_set = [28, 32, 112]
            height = get_random_num(height_set)

            node_.input_shape[0] = [batch_size, in_ch, height, height]
            node_.output_shape = [batch_size, out_ch, height, height]
            node_.params['in_channels'] = in_ch
            node_.params['out_channels'] = out_ch
        elif node_.str_op.lower() == 'softmax':
            node_.params['dim'] = -1
            node_.params['axis'] = -1
            node_.output_shape = self.nodes[index].output_shape
        elif node_.str_op.lower() == 'sum' or node_.str_op == 'mean':
            node_.output_shape = self.nodes[index].output_shape[:-1]
        elif node_.str_op.lower() == 'flatten':
            node_.output_shape = [batch_size, get_element_nums(first_input_shape) // batch_size]
        else:
            node_.output_shape = self.nodes[index].output_shape

        # node_.output_shape = self.nodes[index].output_shape
        to_nodes = self.nodes[index].to_nodes
        id_ = random.randint(0, len(to_nodes) - 1)
        to_id = to_nodes[id_]
        self.insert_node_between(node_, index, to_id)
        # self.display()
        self.fix_shape(node_.id)
        self.fix_shape(to_id)

    def mutate_shape(self, index):
        r"""
        exchange the last two dimension of input data,
        for example, when mutate_shape node a in a-->b,
        the graph becomes a-->transpose-->b
        :param index: the id of node
        :return:
        """
        print(f"debug mutate_shape, node id:{index}")
        output_shape = self.nodes[index].output_shape
        if len(output_shape) != 4:
            print('the length of output_shape < 4, mutate_shape abolish!')
            return
        if output_shape[-1] != output_shape[-2]:
            print(f'the last two dimensions are not equal, mutate_shape abolish!')
            return
        for nxt in self.nodes[index].to_nodes:
            new_id = first_missing_positive([node.id for node in self.nodes.values()])
            node_trans = Node(new_id)
            node_trans.in_degree = node_trans.out_degree = 1
            node_trans.output_shape = output_shape
            node_trans.input_shape.append(output_shape)
            node_trans.str_op = 'transpose'
            node_trans.params['dim0'] = 3
            node_trans.params['dim1'] = 2
            self.insert_node_between(node_trans, index, nxt)

    def mutate_params(self, index):
        r"""
        mutate the params of node index
        if the output shape of the node changes, fix the shape
        a-->b ==> a-->empty_node-->b, then fix shape of empty
        :param index:
        :return:
        """
        if len(self.nodes[index].params) == 0:
            print(f'there is no params in node')
            return
        tmp_output_shape = self.nodes[index].output_shape
        name = self.nodes[index].str_op.lower()
        if name == 'conv2d':
            self.nodes[index].params['kernel_size'] = 1
            self.nodes[index].params['stride'] = 1
            self.nodes[index].params['padding'] = 0
            self.nodes[index].output_shape = self.nodes[index].input_shape[0]

            for nxt in self.nodes[index].to_nodes:
                new_id = first_missing_positive([node.id for node in self.nodes.values()])
                node_empty = Node(new_id)
                node_empty.str_op = 'empty_seq_operator'
                node_empty.in_degree = node_empty.out_degree = 1
                node_empty.output_shape = tmp_output_shape
                node_empty.input_shape.append(node_empty.output_shape)
                self.insert_node_between(node_empty, index, nxt)
                self.fix_shape(nxt)

    def get_src(self):
        for i in self.nodes.keys():
            if self.nodes[i].state == 'src':
                return self.nodes[i]

    def get_des(self):
        for i in self.nodes.keys():
            if self.nodes[i].state == 'des':
                return self.nodes[i]

    def display(self):
        print('display graph:')
        for i in self.nodes.keys():
            print(self.nodes[i])

    def get_graph(self):
        g = []
        for i in self.nodes.keys():
            g.append(str(self.nodes[i]))
        return g

    def __len__(self):
        return len(self.nodes)
