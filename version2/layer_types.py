seq_layer_types = [
    'dense',
    # 'masking',
    'embedding',

    'conv1D',
    'conv2D',
    'conv3D',
    'separable_conv1D',
    'separable_conv2D',
    'depthwise_conv2D',
    'conv2D_transpose',
    'conv3D_transpose',

    'max_pooling1D',
    'max_pooling2D',
    'max_pooling3D',
    'average_pooling1D',
    'average_pooling2D',
    'average_pooling3D',
    'global_max_pooling1D',
    'global_max_pooling2D',
    'global_max_pooling3D',
    'global_average_pooling1D',
    'global_average_pooling2D',
    'global_average_pooling3D',

    'time_distributed',
    'bidirectional',

    'batch_normalization',

    'reshape',
    'flatten',
    'repeat_vector',
    'permute',
    'cropping1D',
    'cropping2D',
    'cropping3D',
    'up_sampling1D',
    'up_sampling2D',
    'up_sampling3D',
    'zero_padding1D',
    'zero_padding2D',
    'zero_padding3D',

    'locally_connected1D',
    'locally_connected2D',
]

RNN_layer_types = [
    'LSTM',
    'GRU',
    'simpleRNN',
    'convLSTM2D',
]

activation_layer_types = [
    'activation',
    'ReLU',
    'softmax',
    'leakyReLU',
    'PReLU',
    'ELU',
    'thresholded_ReLU',
]

merging_layer_types = [
    'concatenate',
    'average',
    'maximum',
    'minimum',
    'add',
    'subtract',
    'multiply',
    'dot',
]

layer_types = seq_layer_types + RNN_layer_types + activation_layer_types + merging_layer_types
layer_types = [''.join(s.split('_')).lower() for s in layer_types]

print(layer_types)
