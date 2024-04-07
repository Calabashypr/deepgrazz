import tensorflow as tf
import torch
import mindspore as ms

if __name__ == '__main__':
    print(f'tensorflow result:{tf.zeros(1)}')
    print(f'torch result:{torch.zeros(1)}')
    try:
        print(f'mindspore result:{ms.ops.Zeros()(1)}')
    except Exception as e:
        print(e)
