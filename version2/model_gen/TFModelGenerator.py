import queue

from version2.graph import Graph
import tensorflow as tf
import numpy as np
from version2.model_gen.OperatorGenerator import gen_operator


class TensorflowModel:
    def __init__(self, g: Graph):
        self.graph = g
        self.layers_ = []
        self.layers = {}
        self.frame_work = 'tensorflow'
        self.gen_dag()
        self.exception_track = {}

    def gen_sequential(self):
        """"""
        if len(self.graph) == 1:
            self.layers_.append(gen_operator(self.graph.nodes[0], self.frame_work))
            return
        id = self.graph.get_src().id
        while self.graph.nodes[id].state != 'des':
            nxt = self.graph.nodes[id].to_nodes[0]
            self.layers_.append(gen_operator(self.graph.nodes[id], self.frame_work))
            id = nxt
            if self.graph.nodes[id].state == 'des':
                self.layers_.append(gen_operator(self.graph.nodes[id], self.frame_work))
                break

    def compute(self, input_data):
        x = np.array(input_data, dtype=np.float32)
        layer_result = [{'input_data': x}]
        for layer in self.layers_:
            x = layer['operator'](x)
            layer_result.append({layer['name']: x})
        return np.array(x), layer_result

    def gen_dag(self):
        r""""""
        src_node = self.graph.get_src()
        if src_node is None:
            id = self.graph.get_des().id
        else:
            id = src_node.id
        if len(self.graph) == 1:
            self.layers[id] = gen_operator(self.graph.nodes[id], self.frame_work)
            return
        q = queue.Queue()
        q.put(id)
        while not q.empty():
            cur = q.get()
            self.layers[cur] = gen_operator(self.graph.nodes[cur], self.frame_work)
            for nxt in self.graph.nodes[cur].to_nodes:
                q.put(nxt)

    def compute_dag(self, input_data):
        print(f"debug TFModelGenerator compute_dag")
        self.exception_track = {}
        x = tf.convert_to_tensor(input_data, dtype=tf.float32)
        layer_result = {'input_data': x}
        x = tf.convert_to_tensor([input_data], dtype=tf.float32)
        res = self.dfs(self.graph.get_des().id, x, layer_result)
        return np.array(res), layer_result

    def dfs(self, id, input_datas, layer_result):
        if self.layers[id]['state'] == 'src' or len(self.graph) == 1:
            x, x_shape = self.get_output(id, input_datas)
            layer_result[id] = {'name': self.layers[id]['name'], 'output': np.array(x), 'output_shape': x_shape,
                                'from': self.layers[id]['from_nodes'], 'to': self.layers[id]['to_nodes']}
            return x

        inputs = []
        for pre in self.layers[id]['from_nodes']:
            inputs.append(self.dfs(pre, input_datas, layer_result))
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        x, x_shape = self.get_output(id, inputs)
        ''''''
        layer_result[id] = {'name': self.layers[id]['name'], 'output': np.array(x), 'output_shape': x_shape,
                            'from': self.layers[id]['from_nodes'], 'to': self.layers[id]['to_nodes']}
        return x

    def get_output(self, id, input_datas):
        r"""
        get the output of the operator
        :param id:          node id
        :param input_datas: the list of inputs ( the number of inputs might > 1 )
        :return: the result and its shape
        """
        name = self.layers[id]['name']
        name = name.lower()
        # print(f"debug TFModelGenerator get_output")
        # print(f"id:{id},name:{name}", end=' ' * 4)
        # for data in input_datas:
        #     print(f"input shape:{data.shape}", end=' ')
        # print()

        self.exception_track = {'id': id, 'name': name, 'framework': 'tensorflow', 'input_datas': input_datas}
        if name in ['argmax', 'argmin', 'reduce_sum', 'sum', 'reduce_mean', 'mean']:
            x = self.layers[id]['operator'](*input_datas, axis=self.layers[id]['params']['axis'])
        elif name == 'slice':
            begin = self.layers[id]['params']['begin']
            size = self.layers[id]['params']['size']
            x = self.layers[id]['operator'](*input_datas, begin=begin, size=size)
        elif name in ['cat', 'concat', 'concatenate']:
            x = self.layers[id]['operator'](input_datas)
        elif name == 'add':
            x = self.layers[id]['operator']([input_datas[0], input_datas[1]])
        elif name == 'pad':
            x = self.layers[id]['operator'](*input_datas, paddings=self.layers[id]['params']['paddings'])
        elif name == 'transpose':
            x = self.layers[id]['operator'](*input_datas, self.layers[id]['params']['perm'])
        elif name == 'remove_edge_operator':
            shape = self.layers[id]['params']['shape']
            dtype = self.layers[id]['params']['dtype']
            x = self.layers[id]['operator'](shape, dtype=dtype)
        else:
            x = self.layers[id]['operator'](*input_datas)

        self.exception_track = {}
        return x, x.shape
