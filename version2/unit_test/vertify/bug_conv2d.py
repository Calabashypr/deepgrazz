import tensorflow as tf
import torch
import mindspore as ms
import numpy as np

if __name__ == '__main__':
    data = np.array([np.inf for _ in range(1 * 128 * 6 * 6)])
    data = np.reshape(data, newshape=[1, 128, 6, 6])
    conv = tf.keras.layers.Conv2D(filters=256, kernel_size=3)
    print(conv(data))
