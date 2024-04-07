import tensorflow as tf
import torch
import torch.nn.functional as F
import mindspore as ms
import mindspore.nn
from version2.utils.common_utils import get_random_num
from version2.graph import *
import os


def gen_operator(node: Node, frame_work):
    if frame_work in ['tensorflow', 'cntk', 'theano', 'mxnet']:
        os.environ['KERAS_BACKEND'] = frame_work
    str_op = node.str_op.lower()
    # params = {'mindspore': node.params, 'tensorflow': node.params, 'torch': node.params}
    params = {'mindspore': {}, 'tensorflow': {}, 'torch': {}}
    op = {}

    dim = axis = node.params.get('dim', node.params.get('axis', -1))
    kernel_size = node.params.get('kernel_size', 1)
    stride = node.params.get('stride', node.params.get('strides', 1))
    padding = node.params.get('padding', node.params.get('paddings', 0))
    data_format = node.params.get('data_format', 'channels_first')
    if data_format == 'channels_first' or data_format == 'NCHW':
        ms_data_format = 'NCHW'
        tf_data_format = 'channels_first'
    else:
        ms_data_format = 'NHWC'
        tf_data_format = 'channels_last'

    '''generate op'''
    if str_op == 'softmax':
        r'''
        ms.ops.Softmax(axis=axis)(ms_data)
        tf.keras.layers.Softmax(axis=axis)(tf_data)
        torch.nn.Softmax(dim=dim)(torch_data)
        '''
        op['mindspore'] = ms.ops.Softmax(axis=axis)
        op['tensorflow'] = tf.keras.layers.Softmax(axis=axis)
        op['torch'] = torch.nn.Softmax(dim=dim)
    elif str_op == 'relu':
        op['mindspore'] = ms.ops.ReLU()
        op['tensorflow'] = tf.keras.layers.ReLU()
        op['torch'] = torch.relu
    elif str_op == 'tanh':
        op['mindspore'] = ms.ops.Tanh()
        op['tensorflow'] = tf.keras.activations.tanh
        op['torch'] = torch.tanh
    elif str_op == 'sigmoid':
        op['mindspore'] = ms.ops.Sigmoid()
        op['tensorflow'] = tf.keras.activations.sigmoid
        op['torch'] = torch.sigmoid
    elif str_op == 'log':
        op['mindspore'] = ms.ops.Log()
        op['tensorflow'] = tf.keras.backend.log
        op['torch'] = torch.log
    elif str_op == 'exp':
        op['mindspore'] = ms.ops.Exp()
        op['tensorflow'] = tf.keras.backend.exp
        op['torch'] = torch.exp
    elif str_op == 'sin':
        op['mindspore'] = ms.ops.Sin()
        op['tensorflow'] = tf.keras.backend.sin
        op['torch'] = torch.sin
    elif str_op == 'cos':
        op['mindspore'] = ms.ops.Cos()
        op['tensorflow'] = tf.keras.backend.cos
        op['torch'] = torch.cos
    elif str_op == 'arctan':
        op['mindspore'] = ms.ops.Atan()
        op['tensorflow'] = tf.atan
        op['torch'] = torch.arctan
    elif str_op == 'argmax':
        r'''
        ms.ops.Argmax(axis=-1)(ms_data)
        tf.argmax(data, axis=-1)
        torch.argmax(torch_data, dim=-1)
        '''
        op['mindspore'] = ms.ops.Argmax(axis=axis)
        op['tensorflow'] = tf.keras.backend.argmax
        op['torch'] = torch.argmax
        params['tensorflow']['axis'] = axis
        params['torch']['dim'] = dim
    elif str_op == 'argmin':
        r'''
        ms.ops.Argmin(axis=-1)(ms_data)
        tf.argmin(data, axis=-1)
        torch.argmin(torch_data, dim=-1)
        '''
        op['mindspore'] = ms.ops.Argmin(axis=axis)
        op['tensorflow'] = tf.keras.backend.argmin
        op['torch'] = torch.argmin
        params['tensorflow']['axis'] = axis
        params['torch']['dim'] = dim
    elif str_op == 'minimum' or str_op == 'min':
        r'''
        ms.ops.Minimum()(ms_data, ms_data)
        tf.minimum(data,data)
        torch.minimum(torch_data, torch_data)
        '''
        op['mindspore'] = ms.ops.Minimum()
        op['tensorflow'] = tf.keras.layers.Minimum()
        op['torch'] = torch.minimum
    elif str_op == 'maximum' or str_op == 'max':
        r'''
        ms.ops.Maximum()(ms_data, ms_data)
        tf.maximum(data,data)
        torch.maximum(torch_data, torch_data)        
        '''
        op['mindspore'] = ms.ops.Maximum()
        op['tensorflow'] = tf.keras.layers.Minimum()
        op['torch'] = torch.maximum
    elif str_op == 'square':
        op['mindspore'] = ms.ops.Square()
        op['tensorflow'] = tf.keras.backend.square
        op['torch'] = torch.square
    elif str_op == 'reduce_sum' or str_op == 'sum':
        r'''
        ms.ops.reduce_sum(ms_data, axis=-1)
        tf.reduce_sum(data, axis=-1)
        torch.sum(torch_data, dim=-1)
        '''
        op['mindspore'] = ms.ops.reduce_sum
        op['tensorflow'] = tf.keras.backend.sum
        op['torch'] = torch.sum
        params['mindspore']['axis'] = axis
        params['tensorflow']['axis'] = axis
        params['torch']['dim'] = dim
    elif str_op == 'reduce_mean' or str_op == 'mean':
        r'''
        ms.ops.reduce_mean(ms_data, axis=-1)
        tf.reduce_mean(data, axis=-1)
        torch.mean(torch_data, dim=-1)
        '''
        op['tensorflow'] = tf.keras.backend.mean
        op['mindspore'] = ms.ops.reduce_mean
        op['torch'] = torch.mean
        params['mindspore']['axis'] = axis
        params['tensorflow']['axis'] = axis
        params['torch']['dim'] = dim
    elif str_op in ['max_pool2d', 'maxpool2d', 'max_pooling2d', 'maxpooling2d']:
        r'''
        ms.ops.MaxPool(kernel_size=kernel_size, strides=stride, pad_mode=pad_mode, data_format='NCHW')(ms_data)
        tf.keras.layers.MaxPool2D(pool_size=(kernel_size, kernel_size), strides=stride, padding=pad_mode,
                                data_format='channels_first')(data)
        torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)(torch_data)
        
        ps:  (kernel_size == 2 * padding + 1 and stride == 1)  equals to pad_mode = 'same'                        
        '''
        kernel_size = node.params.get('kernel_size', 1)
        strides = node.params.get('strides', node.params.get('stride', 1))
        padding = node.params.get('padding', node.params.get('paddings', 0))
        pad_mode = node.params.get('pad_mode', node.params.get('padding_mode', 'valid'))
        data_format = node.params.get('data_format', 'channels_first')
        if data_format == 'channels_first' or data_format == 'NCHW':
            ms_data_format = 'NCHW'
            tf_data_format = 'channels_first'
        else:
            ms_data_format = 'NHWC'
            tf_data_format = 'channels_last'

        if (stride == 1 and kernel_size == 2 * padding + 1) \
                or (kernel_size == 7 and padding == 3 and stride == 2) \
                or (kernel_size == 3 and stride == 2 and padding == 1):
            pad_mode = 'same'
        ms_pad_mode = pad_mode
        if padding > 0:
            ms_pad_mode = 'pad'

        op['mindspore'] = ms.ops.MaxPool(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode,
                                         data_format=ms_data_format)
        op['tensorflow'] = tf.keras.layers.MaxPool2D(pool_size=(kernel_size, kernel_size), strides=strides,
                                                     padding=pad_mode, data_format=tf_data_format)
        op['torch'] = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=strides, padding=padding)

    elif str_op == 'conv2d' or str_op == 'Conv2d':
        r"""
        nums = 1
        in_channels = 6
        out_channels = 2
        height_ = 6
        width_ = 6
        kernel_size = 3
        data = np.ones([nums, in_channels, height_, width_], float)
        torch_data = torch.tensor(data, dtype=torch.float32)
        ms_data = ms.Tensor(data, dtype=ms.float32)
        params = {}
        op = {}
        params['data_format'] = 'channels_first'
        params['kernel_initializer'] = 'ones'
        op['tensorflow'] = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=pad_mode,
                                                  data_format=params['data_format'],
                                                  kernel_initializer=params['kernel_initializer'])
        print(op['tensorflow'](tf_data))
        
        op['mindspore'] = ms.ops.Conv2D(out_channel=out_channels, kernel_size=kernel_size, pad=padding, stride=stride)
        params['ms_weight'] = ms.Tensor(np.ones([out_channels, in_channels, kernel_size, kernel_size]),dtype=ms.float32)
        print(op['mindspore'](ms_data, params['ms_weight']))
        
        op['torch'] = F.conv2d
        params['torch_weight'] = torch.nn.Parameter(torch.ones([out_channels, in_channels, kernel_size, kernel_size]))
        print(op['torch'](torch_data, weight=params['torch_weight'], stride=stride, padding=padding))
        """
        input_shape = node.input_shape[0]
        output_shape = node.output_shape
        in_ch = node.params.get('in_channels', node.params.get('in_channel', input_shape[1]))
        out_ch = node.params.get('out_channels', node.params.get('out_channel'))
        if out_ch is None:
            if len(output_shape) < 1:
                ch_set = [4, 8, 16, 32, 64, 128]
                out_ch = get_random_num(ch_set)
            else:
                out_ch = output_shape[1]
        kernel_size = node.params.get('kernel_size', 1)
        stride = node.params.get('stride', node.params.get('strides', 1))
        padding = node.params.get('padding', node.params.get('paddings', 0))
        dilation = node.params.get('dilation', 1)
        padding_mode = node.params.get('padding_mode', node.params.get('pad_mode', 'valid'))

        if (stride == 1 and kernel_size == 2 * padding + 1) \
                or (kernel_size == 7 and padding == 3 and stride == 2) \
                or (kernel_size == 3 and stride == 2 and padding == 1):
            padding_mode = 'same'
        ms_pad_mode = padding_mode
        if padding > 0:
            ms_pad_mode = 'pad'

        data_format = node.params.get('data_format', 'channels_first')
        if data_format == 'channels_first' or data_format == 'NCHW':
            ms_data_format = 'NCHW'
            tf_data_format = 'channels_first'
        else:
            ms_data_format = 'NHWC'
            tf_data_format = 'channels_last'
        kernel_initializer = node.params.get('kernel_initializer', 'ones')

        op['mindspore'] = ms.ops.Conv2D(out_channel=out_ch, kernel_size=kernel_size, pad=padding, pad_mode=ms_pad_mode,
                                        stride=stride, data_format=ms_data_format)
        op['tensorflow'] = tf.keras.layers.Conv2D(filters=out_ch, kernel_size=kernel_size, data_format=tf_data_format,
                                                  kernel_initializer=kernel_initializer, padding=padding_mode,
                                                  strides=stride)
        op['torch'] = F.conv2d

        params['mindspore']['ms_weight'] = ms.Tensor(
            np.ones([out_ch, in_ch, kernel_size, kernel_size]), dtype=ms.float32)
        params['torch']['weight'] = torch.nn.Parameter(
            torch.ones([out_ch, in_ch, kernel_size, kernel_size]))
        params['torch']['stride'] = stride
        params['torch']['padding'] = padding
    elif str_op == 'avgpool2d':
        r'''
        torch_avg = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0)
        tf_avg = tf.keras.layers.AvgPool2D(pool_size=kernel_size, strides=stride, padding='valid', 
                                            data_format='channels_first')
        ms_avg = ms.ops.AvgPool(kernel_size=kernel_size, strides=stride, pad_mode='valid', data_format='NCHW')
        
        torch_avg(torch_data)
        tf_avg(tf_data)
        ms_avg(ms_data)
        '''
        kernel_size = node.params.get('kernel_size', node.params.get('pool_size', 1))
        stride = node.params.get('stride', node.params.get('strides', 1))
        padding = node.params.get('padding', node.params.get('paddings', 0))
        pad_mode = node.params.get('pad_mode', node.params.get('padding_mode', 'valid'))
        tf_data_format = node.params.get('data_format', 'channels_first')
        ms_data_format = 'NCHW' if tf_data_format == 'channels_first' else 'NHWC'

        if stride == 1 and kernel_size == 2 * padding + 1:
            pad_mode = 'same'

        op['tensorflow'] = tf.keras.layers.AvgPool2D(pool_size=kernel_size, strides=stride, padding=pad_mode,
                                                     data_format=tf_data_format)
        op['torch'] = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        op['mindspore'] = ms.ops.AvgPool(kernel_size=kernel_size, strides=stride, pad_mode=pad_mode,
                                         data_format=ms_data_format)

    elif str_op == 'flatten':
        r'''
        tf_data = np.ones([3, 3, 3])
        torch_data = torch.tensor(tf_data, dtype=torch.float32)
        ms_data = ms.Tensor(tf_data, dtype=ms.float32)
        
        tf.keras.layers.Flatten(data_format='channels_first')(tf_data)
        torch.nn.Flatten(start_dim=1, end_dim=-1)(torch_data)
        ms.ops.Flatten()(ms_data)
        '''
        data_format = node.params.get('data_format', 'channels_last')
        start_dim = node.params.get('start_dim', 1)
        end_dim = node.params.get('end_dim', -1)

        first_input_shape = node.input_shape[0]
        if len(first_input_shape) == 1:
            start_dim = 0

        op['torch'] = torch.nn.Flatten(start_dim=start_dim, end_dim=end_dim)
        op['mindspore'] = ms.ops.Flatten()
        op['tensorflow'] = tf.keras.layers.Flatten(data_format=data_format)

    elif str_op == 'zeropad2d' or str_op == 'zeropadding2d':
        r'''
        torch.nn.ZeroPad2d(padding)(torch_data)
        ms.nn.ZeroPad2d(padding)(ms_data)
        tf.keras.layers.ZeroPadding2D(padding, 'channels_first')(data)
        '''
        op['torch'] = torch.nn.ZeroPad2d(padding=padding)
        op['mindspore'] = ms.nn.ZeroPad2d(padding=padding)
        op['tensorflow'] = tf.keras.layers.ZeroPadding2D(padding=padding, data_format=tf_data_format)
    elif str_op == 'slice':
        r'''
        a = np.array([i for i in range(2 * 6)], dtype=float)
        data = np.reshape(a, [2, 6])
        print(data)
        torch_data = torch.tensor(data, dtype=torch.float32)
        torch_slice = torch.index_select
        
        ms_data = ms.Tensor(data, dtype=ms.float32)
        ms_slice = ms.ops.Slice()
        
        tf_data = data
        tf_slice = tf.slice
        
        print(torch_slice(torch_data, dim=-1, index=torch.tensor([i for i in range(4)], dtype=torch.int64)))
        print(ms_slice(ms_data, begin=[0, 0], size=[-1, 4]))
        print(tf_slice(tf_data, begin=[0, 0], size=[-1, 4]))
        '''
        size = node.params.get('size', node.params.get('shape', node.output_shape))
        begin = node.params.get('begin', [0 for _ in range(len(node.output_shape))])

        op['mindspore'] = ms.ops.Slice()
        op['tensorflow'] = tf.slice
        op['torch'] = torch.index_select
        params['mindspore']['begin'] = begin
        params['mindspore']['size'] = size
        params['torch']['dim'] = dim
        params['torch']['index'] = torch.tensor([i for i in range(size[-1])], dtype=torch.int64)
        params['tensorflow']['begin'] = begin
        params['tensorflow']['size'] = size
    elif str_op == 'concat' or str_op == 'concatenate' or str_op == 'cat':
        r'''
        ms.ops.Concat(axis=-1)([ms_data1, ms_data2])
        tf.keras.layers.Concatenate(axis=-1)([tf_data1, tf_data2])
        torch.cat([torch_data1, torch_data2], dim=-1)
        
        
        
        padding_zeros = np.array([[0 for i in range(3)] for j in range(2)])

        torch_concat = torch.cat
        torch_zeros = torch.tensor(padding_zeros, dtype=torch.float32)
        print(torch_concat([torch_data, torch_zeros], dim=-1))
        tf_concat = tf.concat
        print(tf_concat([tf_data, padding_zeros], axis=-1))
        
        ms_concat = ms.ops.Concat(axis=-1)
        ms_zeros = ms.Tensor(padding_zeros, dtype=ms.float32)
        print(ms_concat([ms_data, ms_zeros]))
        '''
        # output_shape = node.params.get('size', node.params.get('shape', node.output_shape))
        # length = output_shape[1]
        # padding_zeros = np.array([[0 for _ in range(length)] for _ in range(2)])
        # params['mindspore']['zeros'] = ms.Tensor(padding_zeros, dtype=ms.float32)
        # params['tensorflow']['zeros'] = padding_zeros
        # params['torch']['zeros'] = torch.tensor(padding_zeros, dtype=torch.float32)

        op['mindspore'] = ms.ops.Concat(axis=axis)
        op['tensorflow'] = tf.keras.layers.Concatenate(axis=axis)
        op['torch'] = torch.cat

        params['torch']['dim'] = dim
    elif str_op == 'pad' or str_op == 'padding':
        r"""
        tf_data = np.ones([2, 3], dtype=float)
        torch_data = torch.tensor(tf_data, dtype=torch.float32)
        ms_data = ms.Tensor(tf_data, dtype=ms.float32)
        
        torch_pad = F.pad
        torch_pad(torch_data, pad=(0, 1))
        
        tf_pad = tf.pad
        tf_pad(tf_data, paddings=((0, 0), (0, 1)))
        
        ms_pad = ms.ops.Pad(paddings=((0, 0), (0, 1)))
        ms_pad(ms_data)
        """

        pad = node.params.get('pad', node.params.get('paddings', node.params.get('padding', (0, 0))))
        tf_paddings = ms_paddings = pad
        if len(pad) == 2:
            tf_paddings = ((0, 0), pad)
            ms_paddings = ((0, 0), pad)

        # print(f"OperatorGenerator pad:{pad},tf_paddings:{tf_paddings},ms_paddings:{ms_paddings}")

        op['mindspore'] = ms.ops.Pad(paddings=ms_paddings)
        op['tensorflow'] = tf.pad
        # tf.keras.layers.ZeroPadding2D()
        op['torch'] = F.pad

        params['tensorflow']['paddings'] = tf_paddings
        params['torch']['pad'] = pad

    elif str_op == 'add':
        r'''
        ms.ops.Add()(ms_data1, ms_data2)
        tf.keras.layers.Add()([tf_data1, tf_data2])     with [] !
        torch.add(torch_data1, torch_data2)
        '''
        op['mindspore'] = ms.ops.Add()
        op['tensorflow'] = tf.keras.layers.Add()
        # op['tensorflow'] = tf.add
        op['torch'] = torch.add
    elif str_op == 'reshape':
        r'''
        tf_data = np.ones([3, 3, 3])
        torch_data = torch.tensor(tf_data, dtype=torch.float32)
        ms_data = ms.Tensor(tf_data, dtype=ms.float32)
        
        tf.keras.layers.Reshape(target_shape=[9])(tf_data)  # without batch_size !
        torch.reshape(torch_data, shape=(3, 9))
        ms.ops.Reshape()(ms_data, (3, 9))        
        '''
        shape = node.params.get('target_shape',
                                node.params.get('size', node.params.get('shape', node.output_shape)))
        target_shape = shape[1:]

        op['mindspore'] = ms.ops.Reshape()
        op['tensorflow'] = tf.keras.layers.Reshape(target_shape=target_shape)
        op['torch'] = torch.reshape

        params['torch']['shape'] = tuple(shape)
        params['mindspore']['shape'] = tuple(shape)
    elif str_op == 'transpose':
        r"""
        data = np.reshape(np.linspace(1, 24, num=24), newshape=[1, 2, 3, 4])
        tf_data = tf.convert_to_tensor(data, dtype=tf.float32)
        torch_data = torch.tensor(data, dtype=torch.float32)
        ms_data = ms.Tensor(data, dtype=ms.float32)
        
        print(f'tf_res:\n{tf.transpose(tf_data, perm=(0, 1, 3, 2))}')
        print(f'torch_res:\n{torch.transpose(torch_data, dim0=3, dim1=2)}')
        print(f'ms_res:\n{ms.ops.Transpose()(ms_data, (0, 1, 3, 2))}')
        """

        input_shape = node.input_shape[0]
        if input_shape is None or len(input_shape) == 0:
            input_shape = node.output_shape
        dim0 = node.params.get('dim0', input_shape[-1])
        dim1 = node.params.get('dim1', input_shape[-2])

        op['tensorflow'] = tf.transpose
        # op['tensorflow'] = tf.keras.backend.transpose
        op['torch'] = torch.transpose
        op['mindspore'] = ms.ops.Transpose()

        params['tensorflow']['perm'] = (0, 1, dim0, dim1)
        params['torch']['dim0'] = dim0
        params['torch']['dim1'] = dim1
        params['mindspore']['input_perm'] = (0, 1, dim0, dim1)

    elif str_op == 'linear' or str == 'dense':
        r"""
        ms.nn.Dense(in_channels=in_channels, out_channels=out_channels, weight_init='ones', has_bias=False)(ms_data)
        tf.keras.layers.Dense(units=2, use_bias=False, kernel_initializer='ones')(tf_data)
        
        torch_weight=torch.nn.Parameter(torch.ones(out_features, in_features))
        F.linear(torch_data, weight=torch_weight, bias=None)
        """
        first_input_shape = node.input_shape[0]
        in_features = node.params.get('in_features', node.params.get('in_channels', first_input_shape[-1]))
        out_features = node.params.get('out_features', node.params.get('out_channels', node.output_shape[-1]))
        init = node.params.get('kernel_initializer', node.params.get('weight_init', 'ones'))
        bias = node.params.get('bias', node.params.get('use_bias', node.params.get('has_bias', False)))

        torch_weight = torch.nn.Parameter(torch.ones(out_features, in_features))

        op['mindspore'] = ms.nn.Dense(in_channels=in_features, out_channels=out_features, weight_init=init,
                                      has_bias=False)
        op['tensorflow'] = tf.keras.layers.Dense(units=out_features, kernel_initializer=init, use_bias=False)
        op['torch'] = F.linear

        params['torch']['weight'] = torch_weight
        if not bias:
            params['torch']['bias'] = None

    elif str_op == 'remove_edge_operator':
        op['mindspore'] = ms.ops.Zeros()
        op['tensorflow'] = tf.zeros
        op['torch'] = torch.zeros
        output_shape = node.output_shape
        if isinstance(output_shape, list):
            output_shape = tuple(output_shape)

        params['mindspore']['shape'] = output_shape
        params['mindspore']['dtype'] = ms.float32
        params['tensorflow']['shape'] = output_shape
        params['tensorflow']['dtype'] = tf.float32
        params['torch']['shape'] = output_shape
        params['torch']['dtype'] = torch.float32

    elif str_op == 'empty_merge_operator':
        op['mindspore'] = empty_merge_operator
        op['tensorflow'] = empty_merge_operator
        op['torch'] = empty_merge_operator
    else:
        op['mindspore'] = empty_seq_operator
        op['tensorflow'] = empty_seq_operator
        op['torch'] = empty_seq_operator

    return {'operator': op[frame_work], 'params': params[frame_work], 'name': str_op, 'in_degree': node.in_degree,
            'out_degree': node.out_degree, 'from_nodes': node.from_nodes, 'to_nodes': node.to_nodes,
            'state': node.state}


def fx(x):
    return x


def empty_seq_operator(x):
    return x


def empty_merge_operator(*x):
    return x[0]
