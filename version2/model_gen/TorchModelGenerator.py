from version2.graph import Graph
import torch
from version2.model_gen.OperatorGenerator import gen_operator
import queue


class TorchModel:
    def __init__(self, g: Graph):
        self.graph = g
        self.layers_ = []
        self.layers = {}
        self.frame_work = 'torch'
        self.gen_dag()
        self.exception_track = {}

    def gen_sequential(self):
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

    def compute_seq(self, input_data):
        x = torch.tensor(input_data, dtype=torch.float32)
        layer_result = [{'input_data': x}]
        for layer in self.layers_:
            name = layer['name']
            if layer['params'] is None or len(layer['params']) == 0:
                x = layer['operator'](x)
                layer_result.append({layer['name']: x})
            elif name == 'softmax':
                x = layer['operator'](x, layer['params']['dim'])
                layer_result.append({layer['name']: x})

        return x.numpy(), layer_result

    def gen_dag(self):
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
        print(f'debug TorchModelGenerator compute_dag')
        self.exception_track = {}
        x = torch.tensor(input_data, dtype=torch.float32)
        layer_result = {'input_data': x}
        x = torch.tensor([input_data], dtype=torch.float32)
        res = self.dfs(self.graph.get_des().id, x, layer_result)
        return res.detach().numpy(), layer_result

    def dfs(self, id, input_datas, layer_result):
        if self.layers[id]['state'] == 'src' or len(self.graph) == 1:
            x, x_shape = self.get_output(id, input_datas)
            layer_result[id] = {'name': self.layers[id]['name'], 'output': x.detach().numpy(), 'output_shape': x_shape,
                                'from': self.layers[id]['from_nodes'], 'to': self.layers[id]['to_nodes']}
            return x

        inputs = []
        for pre in self.layers[id]['from_nodes']:
            inputs.append(self.dfs(pre, input_datas, layer_result))

        x, x_shape = self.get_output(id, inputs)
        ''''''
        layer_result[id] = {'name': self.layers[id]['name'], 'output': x.detach().numpy(), 'output_shape': x_shape,
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

        # print("debug TorchModelGenerator get_output")
        # print(f"id:{id},name:{name}", end=' ' * 4)
        # for data in input_datas:
        #     print(f"input shape:{data.shape}", end=' ')
        # print()

        self.exception_track = {'id': id, 'name': name, 'frame_work': 'torch', 'input_datas': input_datas}
        if name in ['argmax', 'argmin', 'reduce_sum', 'sum', 'reduce_mean', 'mean']:
            x = self.layers[id]['operator'](*input_datas, dim=self.layers[id]['params']['dim'])
        elif name == 'conv2d':
            weight = self.layers[id]['params']['weight']
            stride = self.layers[id]['params']['stride']
            padding = self.layers[id]['params']['padding']
            # print(f'torch conv2d weight:{weight.shape}')
            x = self.layers[id]['operator'](*input_datas, weight=weight, stride=stride, padding=padding)
        elif name == 'slice':
            dim = self.layers[id]['params']['dim']
            index = self.layers[id]['params']['index']
            x = self.layers[id]['operator'](*input_datas, dim=dim, index=index)
        elif name in ['cat', 'concat', 'concatenate']:
            x = self.layers[id]['operator'](input_datas, dim=self.layers[id]['params']['dim'])
        elif name == 'pad':
            x = self.layers[id]['operator'](*input_datas, pad=self.layers[id]['params']['pad'])
        elif name == 'reshape':
            x = self.layers[id]['operator'](*input_datas, shape=self.layers[id]['params']['shape'])
        elif name == 'linear' or name == 'dense':
            weight = self.layers[id]['params']['weight']
            bias = self.layers[id]['params']['bias']
            x = self.layers[id]['operator'](*input_datas, weight=weight, bias=bias)
        elif name == 'transpose':
            dim0 = self.layers[id]['params']['dim0']
            dim1 = self.layers[id]['params']['dim1']
            x = self.layers[id]['operator'](*input_datas, dim0=dim0, dim1=dim1)
        elif name == 'remove_edge_operator':
            shape = self.layers[id]['params']['shape']
            dtype = self.layers[id]['params']['dtype']
            x = self.layers[id]['operator'](shape, dtype=dtype)
        else:
            x = self.layers[id]['operator'](*input_datas)

        self.exception_track = {}
        return x, x.shape
