import tensorflow as tf
import torch
import torch.nn.functional as F
import mindspore as ms
import mindspore.nn
import numpy as np
from version2.utils.common_utils import get_random_num
from version2.graph import *

if __name__ == '__main__':
    add1 = tf.keras.layers.Add()
    add2 = tf.add
    data1 = np.ones([1, 64, 4, 4])
    data2 = np.ones([1, 64, 4, 4])
    data = []
    data.append(data1)
    data.append(data2)
    print(np.shape(data))
    print(add1(data))
    print(add2(*data))
