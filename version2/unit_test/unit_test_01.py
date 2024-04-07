import tensorflow as tf
import torch
import torch.nn.functional
import mindspore as ms
import numpy as np
import os
import sys
from version2.MCTS.MonteCarloTree import *
from version2.Mutate.Mutate import MutationSelector
from version2.model_gen.TorchModelGenerator import TorchModel
from version2.model_gen.TFModelGenerator import TensorflowModel
from version2.model_gen.MindsporeModelGenerator import MindSporeModel
from version2.model_gen.OperatorSet import operator_set
from version2.utils.shape_calculator import ShapeCalculator

if __name__ == '__main__':
    print(sys.version)
    print(tf.__version__)
    print(torch.__version__)
    print(ms.__version__)


# data1 = np.array([[7, 8], [9, 10], [11, 12]], dtype=float)
# data2 = tf.fill([3, 3, 3], 1)
# torch_data1 = torch.tensor(data1, dtype=torch.float32)
# ms_data1 = ms.Tensor(data1, dtype=ms.float32)

# data0 = np.ones([32, 3, 224, 224], dtype=float)
# torch_data0 = torch.tensor(data0, dtype=torch.float32)
# ms_data0 = ms.Tensor(data0, dtype=ms.float32)
#
# json_dir = os.path.join('..', 'config', 'ResNet50', 'Residual.json')
# graph = json_to_graph(json_dir)
# shape_cal = ShapeCalculator()
# shape_cal.set_shape(graph, data0.shape)
# graph.display()
