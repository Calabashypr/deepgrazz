activation_operators = {
    'relu',
    'tanh',
    'sigmoid',
    'softmax',
}

seq_math_operators = {
    'exp',
    'log',
    'sin',
    'cos',
    'square',
    'arctan',
}

merge_math_operators = {
    'add'
}

seq_operators = {
    *seq_math_operators,
    *activation_operators,
    'conv2d',
    'sum',
    'mean',
    'slice',
    'empty_single_operator',
    'remove_edge_operator',
    'zeropad2d',
    'maxpool2d',
    'avgpool2d',
    'flatten',
    'transpose'
}

merge_operators = {
    *merge_math_operators,
    'minimum',
    'maximum',
    'empty_merge_operator',
    'concat',
}

fix_operators = {
    'slice',
    'pad',
    'reshape'
}

insert_operators = {
    *activation_operators,
    *seq_math_operators,
    'conv2d',
    'sum',
    'mean',
    'flatten'
}

operator_set = {*activation_operators, *seq_math_operators, *seq_operators, *merge_operators, *fix_operators}

if __name__ == '__main__':
    print(len(operator_set))
    print(operator_set)
