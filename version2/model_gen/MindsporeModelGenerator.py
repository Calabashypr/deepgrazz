import queue

import mindspore as ms
from version2.graph import Graph
from version2.model_gen.OperatorGenerator import gen_operator


class MindSporeModel:
    def __init__(self, g: Graph):
        self.graph = g
        self.layers_ = []
        self.layers = {}
        self.frame_work = 'mindspore'
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
        """"""
        x = ms.Tensor(input_data, dtype=ms.float32)
        layer_result = [{'input_data': x}]
        for layer in self.layers_:
            x = layer['operator'](x)
            layer_result.append({layer['name']: x})
        return x.asnumpy(), layer_result

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
        print(f'debug MindsporeModelGenerator compute dag')
        self.exception_track = {}
        x = ms.Tensor(input_data, dtype=ms.float32)
        layer_result = {'input_data': x}
        x = ms.Tensor([input_data], dtype=ms.float32)
        res = self.dfs(self.graph.get_des().id, x, layer_result)
        return res.asnumpy(), layer_result

    def dfs(self, id, input_datas, layer_result):
        if self.layers[id]['state'] == 'src' or len(self.graph) == 1:
            x, x_shape = self.get_output(id, input_datas)
            layer_result[id] = {'name': self.layers[id]['name'], 'output': x.asnumpy(), 'output_shape': x_shape,
                                'from': self.layers[id]['from_nodes'], 'to': self.layers[id]['to_nodes']}
            return x

        inputs = []
        for pre in self.layers[id]['from_nodes']:
            inputs.append(self.dfs(pre, input_datas, layer_result))

        x, x_shape = self.get_output(id, inputs)
        ''''''
        layer_result[id] = {'name': self.layers[id]['name'], 'output': x.asnumpy(), 'output_shape': x_shape,
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
        # print('debug MindsporeModelGenerator get_output')
        # print(f"id:{id},name:{name}", end=' ' * 4)
        # for data in input_datas:
        #     print(f"input shape:{data.shape}", end=' ')
        # print()

        self.exception_track = {'id': id, 'name': name, 'framework': 'mindspore', 'input_datas': input_datas}
        if name in ['reduce_sum', 'sum', 'reduce_mean', 'mean']:
            x = self.layers[id]['operator'](*input_datas, axis=self.layers[id]['params']['axis'])
        elif name == 'conv2d':
            # print(f'mindspore conv2d compute')
            # print(f"weight type:{type(self.layers[id]['params']['ms_weight'])}")
            # print(f"weight type:{type(self.layers[id]['params']['weight'])}")
            x = self.layers[id]['operator'](*input_datas, self.layers[id]['params']['ms_weight'])
            # print(f'mindspore conv2d compute complete')
        elif name == 'slice':
            begin = self.layers[id]['params']['begin']
            size = self.layers[id]['params']['size']
            x = self.layers[id]['operator'](*input_datas, begin, size)
        elif name in ['cat', 'concat', 'concatenate']:
            x = self.layers[id]['operator'](input_datas)
        elif name == 'reshape':
            x = self.layers[id]['operator'](*input_datas, self.layers[id]['params']['shape'])
        elif name == 'transpose':
            x = self.layers[id]['operator'](*input_datas, self.layers[id]['params']['input_perm'])
        elif name == 'remove_edge_operator':
            shape = self.layers[id]['params']['shape']
            dtype = self.layers[id]['params']['dtype']
            x = self.layers[id]['operator'](shape, dtype)
        else:
            x = self.layers[id]['operator'](*input_datas)
        self.exception_track = {}
        return x, x.shape
