from version2.graph import Graph
from version2.utils.common_utils import *
from random import randint
from math import ceil, floor


class MutationSelector:
    def __init__(self, mutation_sets, r=0.2):
        super().__init__()
        self.mutation_strategy = [self.graph_edges_addition, self.graph_edges_removal, self.block_nodes_addition,
                                  self.block_nodes_removal, self.tensor_shape_mutation, self.parameters_mutation]
        self.mutation_name = ['graph_edges_addition', 'graph_edges_removal', 'block_nodes_addition',
                              'block_nodes_removal', 'tensor_shape_mutation', 'parameters_mutation']
        self.r = r
        self.mutations = self.gen_mutation_strategy(mutation_sets)

    def gen_mutation_strategy(self, mutation_sets):
        r""""""
        mutations = []
        for i in mutation_sets:
            if 0 <= i <= 20:
                mutations.append(0)
            elif i <= 30:
                mutations.append(1)
            elif i <= 70:
                mutations.append(2)
            elif i <= 80:
                mutations.append(3)
            elif i <= 90:
                mutations.append(4)
            else:
                mutations.append(5)
        return mutations

    def mutate(self, network: Graph, input_data=None):
        if len(network) <= 1:
            print(f'the number of nodes less than 1, mutate abolish')
            return
        for mutation in self.mutations:
            print('mutate', self.mutation_name[mutation])
            self.mutation_strategy[mutation](network, input_data)
            # print('display:')
            # network.display()
        return True

    def graph_edges_addition(self, network: Graph, input_data=None):
        print("graph_edges_addition add edge")
        tot_edges = len(network.nodes)
        if tot_edges <= 1:
            print('not enough nodes, mutation add edge discard')
            return
        num_new_edges = ceil(tot_edges * self.r)
        num_new_edges = 1

        print('tot_edges', tot_edges, 'num_new_edges', num_new_edges)
        for i in range(num_new_edges):
            src = self.get_random_node(network)
            des = self.get_random_node(network)
            while src == des:
                des = self.get_random_node(network)
            network.add_edge(src, des)
            is_dag, topo_seq = topo_sort(network)
            network.del_edge(src, des)
            if not is_dag:
                print(f'swap src: {src}, des: {des}')
                src, des = des, src
            print('graph_edges_addition add edge:', src, des)
            network.insert_edge(src, des)

    def graph_edges_removal(self, network: Graph, input_data=None):
        r""""""
        if len(network) <= 1:
            print('not enough nodes, mutation remove edge discard')
            return
        src = self.get_random_node(network)
        while network.nodes[src].state == 'des':
            src = self.get_random_node(network)
        to_nodes = network.nodes[src].to_nodes
        des = get_random_num(to_nodes)
        network.remove_edge(src, des)

    def block_nodes_addition(self, network: Graph, input_data=None):
        if probability(self.r):
            id = self.get_random_node(network)
            while network.nodes[id].state == 'des':
                id = self.get_random_node(network)
            network.dup_node(id)

    def block_nodes_removal(self, network: Graph, input_data=None):
        if len(network) <= 2:
            print(f'the number of nodes less than 2, remove node abolish')
            return
        print(f'block_nodes_removal')
        if probability(self.r):
            id = self.get_random_node(network)
            while network.nodes[id].state == 'src' or network.nodes[id].state == 'des' or network.nodes[id] is None:
                id = self.get_random_node(network)
            network.del_node(id)

    def tensor_shape_mutation(self, network: Graph, input_data=None):
        if len(network) <= 2:
            print(f'the number of nodes less than 2, mutate shape abolish')
            return
        id = self.get_random_node(network)
        while network.nodes[id].state == 'des':
            id = self.get_random_node(network)
        network.mutate_shape(id)

    def parameters_mutation(self, network: Graph, input_data=None):
        if len(network) <= 2:
            print(f'the number of nodes less than 2, mutate shape abolish')
            return
        id = self.get_random_node(network)
        while network.nodes[id].state == 'des':
            id = self.get_random_node(network)
        network.mutate_params(id)

    @staticmethod
    def get_random_node(network: Graph):
        id_list = [node.id for node in network.nodes.values()]
        id = randint(0, len(id_list) - 1)
        return id_list[id]
