from version2.graph import Graph, Node
from version2.utils.common_utils import *


class ShapeCalculator:
    def __init__(self):
        r""""""

    def set_shape(self, g: Graph, input_shape):
        r""""""
        if len(g) == 1:
            for id in g.nodes.keys():
                g.nodes[id].input_shape = [input_shape]
                g.nodes[id].output_shape = self.get_output_shape(g.nodes[id], g.nodes[id].input_shape)
            return

        src_id = g.get_src().id
        des_id = g.get_des().id
        g.nodes[src_id].input_shape = [input_shape]
        self.dfs(g, des_id)

    def get_output_shape(self, node: Node, input_shapes):
        r"""
        :param node:
        :param input_shapes: node's in_degree might > 1
        :return:
        """
        # print(f"debug get_output_shape:{node},{node.params}")

        output_shape = []
        name = node.str_op.lower()
        first_input_shape = input_shapes[0]
        # print(f'debug get_output_shape first_input_shape:{first_input_shape}')
        batch_size = first_input_shape[0]
        if name in ['argmin', 'argmax', 'reduce_sum', 'reduce_mean', 'sum', 'mean']:
            output_shape = first_input_shape[:-1]
        elif name == 'flatten':
            output_shape.append(batch_size)
            output_shape.append(get_element_nums(first_input_shape) // first_input_shape[0])
        elif name == 'conv2d' or name.endswith('pool2d'):
            r'''conv2d
                input_shape: [batch_size,in_channels,height,width]
            '''
            in_ch = first_input_shape[1]
            node.params['in_channels'] = in_ch
            out_ch = node.params.get('out_channels', node.params.get('out_channel'))
            if out_ch is None and name.endswith('pool2d'):
                out_ch = in_ch
            stride = node.params.get('stride', node.params.get('strides', 1))
            ker_size = node.params.get('kernel_size', 1)
            padding = node.params.get('padding', node.params.get('paddings', 0))
            dilation = node.params.get('dilation', 1)
            groups = node.params.get('groups', 1)
            bias = node.params.get('bias', True)
            padding_mode = node.params.get('padding_mode', 'zeros')
            pad_mode = node.params.get('pad_mode', node.params.get('padding_mode', 'valid'))
            if stride == 1 and ker_size == 2 * padding + 1:
                pad_mode = 'same'
            height = first_input_shape[2]
            width = first_input_shape[3]

            if pad_mode == 'valid':
                output_shape = [batch_size, out_ch,
                                (height + padding * 2 - ker_size - (ker_size - 1) * (dilation - 1)) // stride + 1,
                                (width + padding * 2 - ker_size - (ker_size - 1) * (dilation - 1)) // stride + 1, ]
            else:
                output_shape = [batch_size, out_ch, height // stride, width // stride]

        elif name == 'dense' or name == 'linear':
            in_features = first_input_shape[-1]
            node.params['in_features'] = in_features
            out_features = node.params.get('out_features', node.params.get('out_channels'))
            output_shape = first_input_shape[:-1]
            output_shape.append(out_features)
        elif name == 'pad' or name == 'padding':
            ''''''
            output_shape = node.output_shape
        else:
            output_shape = first_input_shape

        return output_shape

    def dfs(self, g: Graph, id):
        if g.nodes[id].state == 'src' or len(g) == 1:
            g.nodes[id].output_shape = self.get_output_shape(g.nodes[id], g.nodes[id].input_shape)
            return g.nodes[id].output_shape

        input_shapes = []
        for pre in g.nodes[id].from_nodes:
            input_shapes.append(self.dfs(g, pre))
        g.nodes[id].input_shape = input_shapes
        g.nodes[id].output_shape = self.get_output_shape(g.nodes[id], input_shapes)
        return g.nodes[id].output_shape
