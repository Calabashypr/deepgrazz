import os
from version2.MCTS.MonteCarloTree import *
from version2.Mutate.Mutate import MutationSelector
from version2.utils.shape_calculator import ShapeCalculator

json_dir = os.path.join('..', 'config', 'sub_graph03.json')
graph = json_to_graph(json_dir)
shape_cal = ShapeCalculator()
data = np.random.randn(2, 8)
shape_cal.set_shape(graph, input_shape=data.shape)
graph.display()
graph.del_node(index=3)
shape_cal.set_shape(graph, input_shape=data.shape)
graph.display()
