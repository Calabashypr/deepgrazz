import tensorflow as tf
import torch
import mindspore as ms

if __name__ == '__main__':

    try:
        print(tf.tanh(tf.ones(2, 2), tf.ones(2, 2)))
    except Exception as e:
        print(e)

    try:
        print(torch.tanh(torch.ones(2, 2), torch.ones(2, 2)))
    except Exception as e:
        print(e)

    try:
        print(ms.ops.Tanh(ms.ops.Ones()((2, 2), ms.float32)), ms.ops.Ones()((2, 2), ms.float32))
    except Exception as e:
        print(e)
