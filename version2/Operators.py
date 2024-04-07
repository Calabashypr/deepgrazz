import math
import os.path
import random

# name_table = {
#     '0': 'Conv1d',
#     '1': 'Conv2d',
#     '2': 'Conv3d',
#     '3': 'Conv2dTranspose',
#     '4': 'Conv3dTranspose',
#     '5': 'DepthwiseConv1d',
#     '6': 'DepthwiseConv2d',
#     '7': 'SeparableConv1d',
#     '8': 'SeparableConv2d',
#     '9': 'Embedding',
#     '10': 'BatchNorm1d',
#     '11': 'BatchNorm2d',
#     '12': 'BatchNorm3d',
#     '13': 'LayerNorm1d',
#     '14': 'LayerNorm2d',
#     '15': 'LayerNorm3d',
#     '16': 'MaxPool1d',
#     '17': 'AvgPool1d',
#     '18': 'MaxPool2d',
#     '19': 'AvgPool2d',
#     '20': 'MaxPool3d',
#     '21': 'AvgPool3d',
#     '22': 'GlobalMaxPool1d',
#     '23': 'GlobalAvgPool1d',
#     '24': 'GlobalMaxPool2d',
#     '25': 'GlobalAvgPool2d',
#     '26': 'GlobalMaxPool3d',
#     '27': 'GlobalAvgPool3d',
#     '28': 'ZeroPadding2d',
#     '29': 'Flatten',
#     '30': 'Reshape',
#     '31': 'Resize',
#     '32': 'CropAndResize',
#     '33': 'Upsample1d',
#     '34': 'Upsample2d',
#     '35': 'Upsample3d',
#     '36': 'Unsqueeze',
#     '37': 'Squeeze',
#     '38': 'Transpose',
#     '39': 'Dense',
#     '40': 'BiasAdd',
#     '41': 'Dropout',
#     '42': 'ReduceMean',
#     '43': 'ReduceMax',
#     '44': 'ReduceProd',
#     '45': 'ReduceSum',
#     '46': 'Argmax',
#     '47': 'Argmin',
#     '48': 'Tile',
#     '49': 'Cast',
#     '50': 'Shape',
#     '51': 'Gather',
#     '52': 'Slice',
#     '53': 'StridedSlice',
#     '54': 'Ceil',
#     '55': 'Floor',
#     '56': 'Exp',
#     '57': 'Sqrt',
#     '58': 'Rsqrt',
#     '59': 'Square',
#     '60': 'Compare',
#     '61': 'TopK',
#     '62': 'Lambda',
#     '63': 'GaussianNoise',
#     '64': 'Repeat',
#     '65': 'Threshold',
#     '66': 'ReLU',
#     '67': 'ReLU6',
#     '68': 'ELU',
#     '69': 'SeLU',
#     '70': 'LeakyReLU',
#     '71': 'PReLU',
#     '72': 'Sigmoid',
#     '73': 'Tanh',
#     '74': 'Softmax',
#     '75': 'RNN',
#     '76': 'GRU',
#     '77': 'LSTM',
#     '78': 'RNNCell',
#     '79': 'GRUCell',
#     '80': 'LSTMCell',
#     '81': 'BiRNN',
#     '82': 'BiGRU',
#     '83': 'BiLSTM'
# }

name_table = {
    '0': 'relu',
    '1': 'tanh',
    '2': 'sigmoid',
    '3': 'softmax',
    '4': 'log',
    '5': 'exp',
    '6': 'sin',
    '7': 'cos',
    '8': 'arctan',
}


def get_random_ops(name, input_shape, dim, **kwargs):
    params = {}
    if name == 'Conv1d':
        if len(input_shape) != 2 or dim != 1:
            return None
        params['in_channels'] = input_shape[-1]
        params['out_channels'] = random.randint(1, 128)
        params['kernel_size'] = random.randint(1, max(input_shape[0] // 4, 1))

        dice = random.uniform(0, 1.5)
        if dice <= 0.6:
            params['stride'] = random.randint(1, 4)
            params['dilation'] = 1
        elif dice <= 1.2:
            params['stride'] = 1
            if params['kernel_size'] == 1:
                params['dilation'] = 1
            else:
                params['dilation'] = random.randint(1, max((input_shape[0] - 1) // (params['kernel_size'] - 1) // 2, 1))
        else:
            params['stride'] = 1
            params['dilation'] = 1
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['padding'] = 'valid'
        else:
            params['padding'] = 'same'
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['bias'] = True
        else:
            params['bias'] = False
    elif name == 'Conv2d':
        if len(input_shape) != 3 or dim != 2:
            return None
        params['in_channels'] = input_shape[-1]
        params['out_channels'] = random.randint(1, 128)
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['kernel_size'] = [random.randint(1, max(input_shape[0] // 4, 1)),
                                     random.randint(1, max(input_shape[1] // 4, 1))]
        else:
            tmp = min(input_shape[0], input_shape[1])
            k = random.randint(1, max(tmp // 4, 1))
            params['kernel_size'] = [k, k]

        dice = random.uniform(0, 1.5)
        if dice <= 0.6:
            dice = random.uniform(0, 1)
            if dice <= 0.5:
                params['stride'] = [random.randint(1, 4),
                                    random.randint(1, 4)]
            else:
                params['stride'] = random.randint(1, 4)
            params['dilation'] = 1
        elif dice <= 1.2:
            params['stride'] = 1
            dice = random.uniform(0, 1)
            if dice <= 0.5:
                if params['kernel_size'][0] == 1:
                    l = 1
                else:
                    l = random.randint(1, max((input_shape[0] - 1) // (params['kernel_size'][0] - 1) // 2, 1))
                if params['kernel_size'][1] == 1:
                    r = 1
                else:
                    r = random.randint(1, max((input_shape[1] - 1) // (params['kernel_size'][1] - 1) // 2, 1))
                params['dilation'] = [l, r]
            else:
                if min(params['kernel_size'][0], params['kernel_size'][1]) == 1:
                    params['dilation'] = 1
                else:
                    k = min(params['kernel_size'][0], params['kernel_size'][1])
                    shape = min(input_shape[0], input_shape[1])
                    params['dilation'] = random.randint(1, max((shape - 1) // (k - 1) // 2, 1))
        else:
            params['stride'] = 1
            params['dilation'] = 1
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['padding'] = 'valid'
        else:
            params['padding'] = 'same'
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['bias'] = True
        else:
            params['bias'] = False
    elif name == 'Conv3d':
        if len(input_shape) != 4 or dim != 3:
            return None
        params['in_channels'] = input_shape[-1]
        params['out_channels'] = random.randint(1, 512)
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['kernel_size'] = [random.randint(1, max(input_shape[0] // 4, 1)),
                                     random.randint(1, max(input_shape[1] // 4, 1)),
                                     random.randint(1, max(input_shape[2] // 4, 1))]
        else:
            k = random.randint(1, max(min(input_shape[0], input_shape[1], input_shape[2]) // 4, 1))
            params['kernel_size'] = [k, k, k]

        dice = random.uniform(0, 1.5)
        if dice <= 0.6:
            dice = random.uniform(0, 1)
            if dice <= 0.5:
                params['stride'] = [random.randint(1, 4),
                                    random.randint(1, 4),
                                    random.randint(1, 4)]
            else:
                params['stride'] = random.randint(1, 4)
            params['dilation'] = 1
        elif dice <= 1.2:
            params['stride'] = 1
            dice = random.uniform(0, 1)
            if dice <= 0.5:
                if params['kernel_size'][0] == 1:
                    l = 1
                else:
                    l = random.randint(1, max((input_shape[0] - 1) // (params['kernel_size'][0] - 1) // 2, 1))
                if params['kernel_size'][1] == 1:
                    m = 1
                else:
                    m = random.randint(1, max((input_shape[1] - 1) // (params['kernel_size'][1] - 1) // 2, 1))
                if params['kernel_size'][2] == 1:
                    r = 1
                else:
                    r = random.randint(1, max((input_shape[2] - 1) // (params['kernel_size'][2] - 1) // 2, 1))
                params['dilation'] = [l, m, r]
            else:
                if min(params['kernel_size'][0], params['kernel_size'][1], params['kernel_size'][2]) == 1:
                    params['dilation'] = 1
                else:
                    k = min(params['kernel_size'][0], params['kernel_size'][1], params['kernel_size'][2])
                    shape = min(input_shape[0], input_shape[1], input_shape[2])
                    params['dilation'] = random.randint(1, max((shape - 1) // (k - 1) // 2, 1))
        else:
            params['stride'] = 1
            params['dilation'] = 1
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['padding'] = 'valid'
        else:
            params['padding'] = 'same'
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['bias'] = True
        else:
            params['bias'] = False
    elif name == 'Conv2dTranspose':
        if len(input_shape) != 3 or dim != 2:
            return None
        params['in_channels'] = input_shape[-1]
        params['out_channels'] = random.randint(1, 32)
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['kernel_size'] = [random.randint(1, max(input_shape[0] // 4, 1)),
                                     random.randint(1, max(input_shape[1] // 4, 1))]
        else:
            tmp = min(input_shape[0], input_shape[1])
            k = random.randint(1, max(tmp // 4, 1))
            params['kernel_size'] = [k, k]

        dice = random.uniform(0, 1.5)
        if dice <= 0.6:
            dice = random.uniform(0, 1)
            if dice <= 0.5:
                params['stride'] = [random.randint(1, 4),
                                    random.randint(1, 4)]
            else:
                params['stride'] = random.randint(1, 4)
            params['dilation'] = 1
        elif dice <= 1.2:
            params['stride'] = 1
            dice = random.uniform(0, 1)
            if dice <= 0.5:
                if params['kernel_size'][0] == 1:
                    l = 1
                else:
                    l = random.randint(1, max((input_shape[0] - 1) // (params['kernel_size'][0] - 1) // 2, 1))
                if params['kernel_size'][1] == 1:
                    r = 1
                else:
                    r = random.randint(1, max((input_shape[1] - 1) // (params['kernel_size'][1] - 1) // 2, 1))
                params['dilation'] = [l, r]
            else:
                if min(params['kernel_size'][0], params['kernel_size'][1]) == 1:
                    params['dilation'] = 1
                else:
                    k = min(params['kernel_size'][0], params['kernel_size'][1])
                    shape = min(input_shape[0], input_shape[1])
                    params['dilation'] = random.randint(1, max((shape - 1) // (k - 1) // 2, 1))
        else:
            params['stride'] = 1
            params['dilation'] = 1
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['padding'] = 'valid'
        else:
            params['padding'] = 'same'
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['bias'] = True
        else:
            params['bias'] = False
        if dice <= 0.6:
            if isinstance(params['stride'], int):
                stride_min = params['stride']
            else:
                stride_min = min(params['stride'][0], params['stride'][1])
            params['out_padding'] = random.randint(0, stride_min - 1)
        else:
            params['out_padding'] = 0
    elif name == 'Conv3dTranspose':
        if len(input_shape) != 4 or dim != 3:
            return None
        params['in_channels'] = input_shape[-1]
        params['out_channels'] = random.randint(1, 128)
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['kernel_size'] = [random.randint(1, max(input_shape[0] // 4, 1)),
                                     random.randint(1, max(input_shape[1] // 4, 1)),
                                     random.randint(1, max(input_shape[2] // 4, 1))]
        else:
            k = random.randint(1, max(min(input_shape[0], input_shape[1], input_shape[2]) // 4, 1))
            params['kernel_size'] = [k, k, k]

        dice = random.uniform(0, 1.5)
        if dice <= 0.6:
            dice = random.uniform(0, 1)
            if dice <= 0.5:
                params['stride'] = [random.randint(1, 4),
                                    random.randint(1, 4),
                                    random.randint(1, 4)]
            else:
                params['stride'] = random.randint(1, 4)
            params['dilation'] = 1
        elif dice <= 1.2:
            params['stride'] = 1
            dice = random.uniform(0, 1)
            if dice <= 0.5:
                params['dilation'] = [
                    random.randint(1, max((input_shape[0] - 1) // (params['kernel_size'][0] - 1) // 2, 1)),
                    random.randint(1, max((input_shape[1] - 1) // (params['kernel_size'][1] - 1) // 2, 1)),
                    random.randint(1, max((input_shape[2] - 1) // (params['kernel_size'][2] - 1) // 2, 1))]
            else:
                tmp = min(
                    (input_shape[0] - 1) // (params['kernel_size'][0] - 1),
                    (input_shape[1] - 1) // (params['kernel_size'][1] - 1),
                    (input_shape[2] - 1) // (params['kernel_size'][2] - 1)
                )
                params['dilation'] = random.randint(1, max(tmp // 2, 1))
        else:
            params['stride'] = 1
            params['dilation'] = 1
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['padding'] = 'valid'
        else:
            params['padding'] = 'same'
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['bias'] = True
        else:
            params['bias'] = False
        if dice <= 0.6:
            if isinstance(params['stride'], int):
                stride_min = params['stride']
            else:
                stride_min = min(params['stride'][0], params['stride'][1], params['stride'][2])
            params['out_padding'] = random.randint(0, stride_min - 1)
        else:
            params['out_padding'] = 0
    elif name == 'DepthwiseConv1d':
        return None
        if len(input_shape) != 2 or dim != 1:
            return None
        params['in_channels'] = input_shape[-1]
        params['depth_multiplier'] = random.randint(1, 64)
        params['kernel_size'] = random.randint(1, max(input_shape[0] // 4, 1))

        dice = random.uniform(0, 1.5)
        if dice <= 0.6:
            params['stride'] = random.randint(1, 4)
            params['dilation'] = 1
        elif dice <= 1.2:
            params['stride'] = 1
            if params['kernel_size'] == 1:
                params['dilation'] = 1
            else:
                params['dilation'] = random.randint(1, max((input_shape[0] - 1) // (params['kernel_size'] - 1) // 2, 1))
        else:
            params['stride'] = 1
            params['dilation'] = 1
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['padding'] = 'valid'
        else:
            params['padding'] = 'same'
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['bias'] = True
        else:
            params['bias'] = False
    elif name == 'DepthwiseConv2d':
        if len(input_shape) != 3 or dim != 2:
            return None
        params['in_channels'] = input_shape[-1]
        params['depth_multiplier'] = random.randint(1, 64)
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['kernel_size'] = [random.randint(1, max(input_shape[0] // 4, 1)),
                                     random.randint(1, max(input_shape[1] // 4, 1))]
        else:
            k = random.randint(1, max(min(input_shape[0], input_shape[1]) // 4, 1))
            params['kernel_size'] = [k, k]

        dice = random.uniform(0, 1.5)
        if dice <= 0.6:
            dice = random.uniform(0, 1)
            if dice <= 0.5:
                params['stride'] = [random.randint(1, 4),
                                    random.randint(1, 4)]
            else:
                params['stride'] = random.randint(1, 4)
            params['dilation'] = 1
        elif dice <= 1.2:
            params['stride'] = 1
            dice = random.uniform(0, 1)
            if dice <= 0.5:
                if params['kernel_size'][0] == 1:
                    l = 1
                else:
                    l = random.randint(1, max((input_shape[0] - 1) // (params['kernel_size'][0] - 1) // 2, 1))
                if params['kernel_size'][1] == 1:
                    r = 1
                else:
                    r = random.randint(1, max((input_shape[1] - 1) // (params['kernel_size'][1] - 1) // 2, 1))
                params['dilation'] = [l, r]
            else:
                if min(params['kernel_size'][0], params['kernel_size'][1]) == 1:
                    params['dilation'] = 1
                else:
                    k = min(params['kernel_size'][0], params['kernel_size'][1])
                    shape = min(input_shape[0], input_shape[1])
                    params['dilation'] = random.randint(1, max((shape - 1) // (k - 1) // 2, 1))
        else:
            params['stride'] = 1
            params['dilation'] = 1
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['padding'] = 'valid'
        else:
            params['padding'] = 'same'
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['bias'] = True
        else:
            params['bias'] = False
    elif name == 'SeparableConv1d':
        if len(input_shape) != 2 or dim != 1:
            return None
        params['in_channels'] = input_shape[-1]
        params['depth_multiplier'] = random.randint(1, 64)
        params['out_channels'] = random.randint(1, 128)
        params['kernel_size'] = random.randint(1, max(input_shape[0] // 4, 1))

        dice = random.uniform(0, 1.5)
        if dice <= 0.6:
            params['stride'] = random.randint(1, 4)
            params['dilation'] = 1
        elif dice <= 1.2:
            params['stride'] = 1
            if params['kernel_size'] == 1:
                params['dilation'] = 1
            else:
                params['dilation'] = random.randint(1, max((input_shape[0] - 1) // (params['kernel_size'] - 1) // 2, 1))
        else:
            params['stride'] = 1
            params['dilation'] = 1
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['padding'] = 'valid'
        else:
            params['padding'] = 'same'
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['bias'] = True
        else:
            params['bias'] = False
    elif name == 'SeparableConv2d':
        if len(input_shape) != 3 or dim != 2:
            return None
        params['in_channels'] = input_shape[-1]
        params['depth_multiplier'] = random.randint(1, 64)
        params['out_channels'] = random.randint(1, 128)
        dice = random.uniform(0, 1)
        if dice <= 0.3:
            params['kernel_size'] = [random.randint(1, max(input_shape[0] // 4, 1)),
                                     random.randint(1, max(input_shape[1] // 4, 1))]
        else:
            k = random.randint(1, max(min(input_shape[0], input_shape[1]) // 4, 1))
            params['kernel_size'] = [k, k]

        dice = random.uniform(0, 1.5)
        if dice <= 0.6:
            dice = random.uniform(0, 1)
            if dice <= 0.5:
                params['stride'] = [random.randint(1, 4),
                                    random.randint(1, 4)]
            else:
                params['stride'] = random.randint(1, 4)
            params['dilation'] = 1
        elif dice <= 1.2:
            params['stride'] = 1
            dice = random.uniform(0, 1)
            if dice <= 0.3:
                if params['kernel_size'][0] == 1:
                    l = 1
                else:
                    l = random.randint(1, max((input_shape[0] - 1) // (params['kernel_size'][0] - 1) // 2, 1))
                if params['kernel_size'][1] == 1:
                    r = 1
                else:
                    r = random.randint(1, max((input_shape[1] - 1) // (params['kernel_size'][1] - 1) // 2, 1))
                params['dilation'] = [l, r]
            else:
                if min(params['kernel_size'][0], params['kernel_size'][1]) == 1:
                    params['dilation'] = 1
                else:
                    k = min(params['kernel_size'][0], params['kernel_size'][1])
                    shape = min(input_shape[0], input_shape[1])
                    params['dilation'] = random.randint(1, max((shape - 1) // (k - 1) // 2, 1))
        else:
            params['stride'] = 1
            params['dilation'] = 1
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['padding'] = 'valid'
        else:
            params['padding'] = 'same'
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['bias'] = True
        else:
            params['bias'] = False
    # Embedding 必须在第一层 需要特殊解决
    elif name == 'Embedding':
        if dim != 1:
            return None
        if kwargs.__contains__('input_dim'):
            params['input_dim'] = kwargs['input_dim']
        else:
            raise Exception('Lack of Param named \"input_dim\".')
        params['output_dim'] = random.randint(1, 64)
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['mask_zero'] = True
        else:
            params['mask_zero'] = False
    elif name == 'BatchNorm1d' or name == 'BatchNorm2d' or name == 'BatchNorm3d':
        if len(input_shape) != int(name[-2]) + 1 or dim != int(name[-2]):
            return None
        params['num_features'] = input_shape[-1]
        params['eps'] = random.uniform(1e-7, 1e-5)
        params['momentum'] = random.uniform(0, 1)
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['affine'] = True
        else:
            params['affine'] = False
        params['track_running_stats'] = True
    elif name == 'LayerNorm1d' or name == 'LayerNorm2d' or name == 'LayerNorm3d':
        if len(input_shape) != int(name[-2]) + 1 or dim != int(name[-2]):
            return None
        params['eps'] = random.uniform(1e-7, 1e-5)
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['affine'] = True
        else:
            params['affine'] = False
    elif name == 'MaxPool1d' or name == 'AvgPool1d':
        if len(input_shape) != 2 or dim != 1:
            return None
        params['pool_size'] = random.randint(1, max(input_shape[0] // 4, 1))
        params['stride'] = random.randint(1, 4)
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['padding'] = 'valid'
        else:
            params['padding'] = 'same'
    elif name == 'MaxPool2d' or name == 'AvgPool2d':
        if len(input_shape) != 3 or dim != 2:
            return None
        dice = random.uniform(0, 1)
        if dice <= 0.3:
            params['pool_size'] = [random.randint(1, max(input_shape[0] // 4, 1)),
                                   random.randint(1, max(input_shape[1] // 4, 1))]
        else:
            k = random.randint(1, max(min(input_shape[0], input_shape[1]) // 4, 1))
            params['pool_size'] = [k, k]
        dice = random.uniform(0, 1)
        if dice <= 0.3:
            params['stride'] = [random.randint(1, 4),
                                random.randint(1, 4)]
        else:
            s = random.randint(1, 4)
            params['stride'] = [s, s]
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['padding'] = 'valid'
        else:
            params['padding'] = 'same'
    elif name == 'MaxPool3d' or name == 'AvgPool3d':
        if len(input_shape) != 3 or dim != 3:
            return None
        dice = random.uniform(0, 1)
        if dice <= 0.3:
            params['pool_size'] = [random.randint(1, max(input_shape[0] // 4, 1)),
                                   random.randint(1, max(input_shape[1] // 4, 1))]
        else:
            k = random.randint(1, max(min(input_shape[0], input_shape[1]) // 4, 1))
            params['pool_size'] = [k, k]
        dice = random.uniform(0, 1)
        if dice <= 0.3:
            params['stride'] = [random.randint(1, 4),
                                random.randint(1, 4)]
        else:
            s = random.randint(1, 4)
            params['stride'] = [s, s]
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['padding'] = 'valid'
        else:
            params['padding'] = 'same'
    elif name == 'GlobalAvgPool1d' or name == 'GlobalMaxPool1d':
        if len(input_shape) != 2 or dim != 1:
            return None
    elif name == 'GlobalAvgPool2d' or name == 'GlobalMaxPool2d':
        if len(input_shape) != 3 or dim != 2:
            return None
    elif name == 'GlobalAvgPool3d' or name == 'GlobalMaxPool3d':
        if len(input_shape) != 4 or dim != 3:
            return None
    elif name == 'ZeroPadding2d':
        if len(input_shape) != 3 or dim != 2:
            return None
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['pad'] = [random.randint(1, 8), random.randint(1, 8)]
        else:
            params['pad'] = [random.randint(1, 4), random.randint(1, 4),
                             random.randint(1, 4), random.randint(1, 4)]
    elif name == 'Flatten':
        pass
    elif name == 'Reshape':
        params['tensor_space'] = dim + 2
        sum = 1
        for x in input_shape:
            sum *= x
        params['output_shape'] = separate(sum, random.randint(1, len(input_shape) + 1))
    elif name == 'Resize':
        return None
        if len(input_shape) != 3 or dim != 2:
            return None
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['output_shape'] = random.randint(1, 128)
        else:
            params['output_shape'] = [random.randint(1, 128), random.randint(1, 128)]
        dice = random.uniform(0, 1.5)
        if dice <= 0.5:
            params['mode'] = 'bilinear'
        elif dice <= 1.0:
            params['mode'] = 'nearest'
        else:
            params['mode'] = 'bicubic'
    elif name == 'CropAndResize':
        return None
        if len(input_shape) != 3 or dim != 2:
            return None
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['output_shape'] = random.randint(1, 128)
        else:
            params['output_shape'] = [random.randint(1, 128), random.randint(1, 128)]
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['mode'] = 'bilinear'
        else:
            params['mode'] = 'nearest'
        params['top'] = random.randint(0, input_shape[0] - 1)
        params['left'] = random.randint(0, input_shape[1] - 1)
        params['height'] = random.randint(1, input_shape[0] - params['top'])
        params['width'] = random.randint(1, input_shape[1] - params['left'])
    elif name == 'Upsample1d':
        if len(input_shape) != 2 or dim != 1:
            return None
        params['scale_factor'] = random.randint(1, 4)
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['mode'] = 'bilinear'
        else:
            params['mode'] = 'nearest'
    elif name == 'Upsample2d':
        if len(input_shape) != 3 or dim != 2:
            return None
        dice = random.uniform(0, 1)
        if dice <= 0.7:
            params['scale_factor'] = random.randint(1, 4)
        else:
            params['scale_factor'] = [random.randint(1, 3), random.randint(1, 3)]
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['mode'] = 'bilinear'
        else:
            params['mode'] = 'nearest'
    elif name == 'Upsample3d':
        if len(input_shape) != 4 or dim != 3:
            return None
        dice = random.uniform(0, 1)
        if dice <= 0.7:
            params['scale_factor'] = random.randint(1, 4)
        else:
            params['scale_factor'] = [random.randint(1, 2), random.randint(1, 2), random.randint(1, 2)]
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['mode'] = 'bilinear'
        else:
            params['mode'] = 'nearest'
    elif name == 'Unsqueeze':
        params['tensor_space'] = dim + 2
        params['dim'] = random.randint(1, len(input_shape) + 1)
    elif name == 'Squeeze':
        if len(input_shape) <= 1:
            return None
        if 1 not in input_shape:
            return None
        axis = []
        for i in range(len(input_shape)):
            if input_shape[i] == 1:
                axis.append(i + 1)
        random.shuffle(axis)
        params['tensor_space'] = dim + 2
        params['dim'] = axis[0]
    elif name == 'Transpose':
        lst = [i + 1 for i in range(len(input_shape))]
        random.shuffle(lst)
        lst = [0] + lst
        params['output_shape'] = lst
        params['tensor_space'] = dim + 2
    elif name == 'Dense':
        if len(input_shape) != 1:
            return None
        params['in_features'] = input_shape[-1]
        params['out_features'] = random.randint(1, 1024)
        dice = 0.5
        if dice <= 0.5:
            params['bias'] = True
        else:
            params['bias'] = False
    elif name == 'BiasAdd':
        params['bias'] = random.uniform(0, 0.1)
    elif name == 'Dropout':
        params['p'] = random.uniform(0.1, 0.9)
    elif name == 'ReduceMean' or name == 'ReduceMax' or name == 'ReduceProd' or name == 'ReduceSum':
        if len(input_shape) <= 1:
            return None
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['keep_dims'] = True
        else:
            params['keep_dims'] = False
        params['dim'] = random.randint(1, len(input_shape))
        params['tensor_space'] = dim + 2
    elif name == 'Argmax' or name == 'Argmin':
        if len(input_shape) <= 1:
            return None
        params['dim'] = random.randint(1, len(input_shape))
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            if dim == 1 and len(input_shape) == 2:
                params['channel_stay'] = True
            elif dim == 2 and len(input_shape) == 3:
                params['channel_stay'] = True
            elif dim == 3 and len(input_shape) == 4:
                params['channel_stay'] = True
            else:
                params['channel_stay'] = False
        else:
            params['channel_stay'] = False
    elif name == 'Tile':
        params['multiples'] = []
        for i in range(len(input_shape)):
            params['multiples'].append(random.randint(1, 4))
        params['tensor_space'] = dim + 2
    elif name == 'Cast':
        lst = ["float32", "float64"]
        params['target_dtype'] = lst[random.randint(0, 1)]
    elif name == 'Shape':
        pass
    elif name == 'Gather':
        params['tensor_space'] = dim + 2
        params['dim'] = random.randint(1, len(input_shape))
        lst = []
        num = random.randint(1, input_shape[params['dim'] - 1] + 10)
        for i in range(num):
            lst.append(random.randint(0, input_shape[params['dim'] - 1] - 1))
        params['index'] = lst
    elif name == 'Slice':
        params['tensor_space'] = dim + 2
        params['begin'] = [0]
        params['size'] = [25]
        for x in input_shape:
            bg = random.randint(0, x - 1)
            size = random.randint(1, x - bg)
            params['begin'].append(bg)
            params['size'].append(size)
    elif name == 'StridedSlice':
        return None
        params['tensor_space'] = dim + 2
        params['begin'] = [0]
        params['end'] = [25]
        params['stride'] = [1]
        for x in input_shape:
            bg = random.randint(0, (x - 1) // 2)
            end = random.randint(bg + 1, x)
            params['begin'].append(bg)
            params['end'].append(end)
            params['stride'].append(random.randint(1, 3))
    elif name == 'Ceil' or name == 'Floor' or name == 'Exp' \
            or name == 'Sqrt' or name == 'Rsqrt' or name == 'Square':
        pass
    elif name == 'Compare':
        table = ['<', '<=', '==', '>=', '>']
        params['op'] = table[random.randint(0, 4)]
    elif name == 'TopK':
        if len(input_shape) != 1:
            return None
        params['output'] = 'values'
        params['k'] = random.randint(1, input_shape[0])
    elif name == 'Lambda':
        table = ['lambda x: x**2', 'lambda x: x + 1.0', 'lambda x: 1 / x**0.5 + 1.0',
                 'lambda x: x**0.5', 'lambda x: 1 / x**2 + 1.0']
        params['output_shape'] = input_shape
        params['function'] = table[random.randint(0, 4)]
    elif name == 'GaussianNoise':
        params['stddev'] = random.uniform(1e-5, 1e-2)
    elif name == 'Repeat':
        if len(input_shape) != 1:
            return None
        params['n'] = random.randint(1, 4)
    elif name == 'Threshold':
        params['threshold'] = random.uniform(0, 1)
    elif name == 'ReLU' or name == 'ReLU6' or name == 'SeLU' or name == 'Sigmoid' or name == 'Tanh':
        pass
    elif name == 'ELU':
        params['alpha'] = random.uniform(0.1, 1.0)
    elif name == 'LeakyReLU':
        params['alpha'] = random.uniform(1e-5, 1e-1)
    elif name == 'PReLU':
        params['init'] = random.uniform(0, 1)
        dice = random.uniform(0, 1)
        params['share'] = True
    elif name == 'Softmax':
        dice = random.uniform(0, 1)
        if dice <= 0.8:
            params['dim'] = -1
        else:
            params['dim'] = random.randint(1, len(input_shape))
    elif name == 'RNN' or name == 'GRU' or name == 'LSTM' \
            or name == 'BiRNN' or name == 'BiGRU' or name == 'BiLSTM':
        if dim != 1:
            return None
        params['input_size'] = input_shape[-1]
        params['hidden_size'] = random.randint(1, 1024)
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['bias'] = True
        else:
            params['bias'] = False
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['dropout'] = random.uniform(0.1, 0.9)
        else:
            params['dropout'] = 0
    elif name == 'RNNCell' or name == 'GRUCell' or name == 'LSTMCell':
        if dim != 1:
            return None
        params['input_size'] = input_shape[-1]
        params['hidden_size'] = random.randint(1, 1024)
        dice = random.uniform(0, 1)
        if dice <= 0.5:
            params['bias'] = True
        else:
            params['bias'] = False
    return {
        'name': name,
        'params': params
    }


def separate(x, num):
    if num == 1:
        return [x]
    lst = []
    while x != 1:
        for i in range(2, x + 1):
            if x % i == 0:
                lst.append(i)
                x = x // i
                break
    if len(lst) < num:
        return [1] * (num - len(lst)) + lst
    else:
        res = []
        len_per = len(lst) // num
        for i in range(num - 1):
            tmp = 1
            for k in lst[i * len_per: (i + 1) * len_per]:
                tmp *= k
            res.append(tmp)
        tmp = 1
        for k in lst[(num - 1) * len_per:]:
            tmp *= k
        res.append(tmp)
        return res


def get_random_divide(x):
    if x == 1:
        return 1
    else:
        lst = []
        for i in range(1, x + 1):
            if x % i == 0:
                lst.append(i)
        random.shuffle(x)
        return x[0]
