import os
from version2.MCTS.MonteCarloTree import *
from version2.MCTS.random_graph_gen import *
from version2.Mutate.Mutate import MutationSelector
from version2.utils.shape_calculator import ShapeCalculator

json_dir = os.path.join('..', 'config', 'vgg16', 'vgg16.json')
graph = json_to_graph(json_dir)
for op in graph.get_graph():
    print(op)
print('-' * 50)
input_shape = [4, 3, 28, 28]
# data = np.random.randn(2, 3, 8, 8)

for i in range(5):
    print('-' * 100)
    print(f'iter:{i}')
    sub_g = random_graph(graph, 10)
    sub_g.display()
    shape_cal = ShapeCalculator()
    shape_cal.set_shape(sub_g, input_shape=input_shape)
    sub_g.display()
    # mutation_sets = [randint(2, 3) for _ in range(min(i, 8))]
    # mutation_selector = MutationSelector(mutation_sets, r=1)
    # mutation_selector.mutate(sub_g)
    # sub_g.display()
    # shape_cal.set_shape(sub_g, input_shape=data.shape)
    # sub_g.display()
