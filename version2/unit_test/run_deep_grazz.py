import copy
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(project_root)
sys.path.insert(0, project_root)
from version2.MCTS.MonteCarloTree import *
from version2.MCTS.random_graph_gen import *
from version2.Mutate.Mutate import MutationSelector
from version2.model_gen.TorchModelGenerator import TorchModel
from version2.model_gen.TFModelGenerator import TensorflowModel
from version2.model_gen.MindsporeModelGenerator import MindSporeModel
from version2.utils.shape_calculator import ShapeCalculator
from version2.utils.coverage_calculator import CoverageCalculator
from version2.Results.result_analyser import ResultAnalyser
from version2.data.mnist_load import MnistDataLoader
from version2.data.rand_data_load import RandomDataLoader
from version2.data.cifar10_load import Cifar10DataLoader

import os
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import torch
import mindspore as ms


def model_run(model, frame_work, res_dict: dict, data):
    r"""

    :param model:
    :param frame_work:
    :param res_dict:
    :param data:
    :return:
    """
    res_dict[frame_work]['flag_run'] = True
    try:
        print('-' * 10, f'{frame_work}_res', '-' * 10)
        res = model.compute_dag(data)
        res_dict[frame_work]['final_res'] = res[0]
        res_dict[frame_work]['layer_res'] = res[1]
    except BaseException as e:
        res_dict[frame_work]['flag_run'] = False
        res_dict[frame_work]['exception_info'] = e
        print(f'{frame_work} compute failure:{e}')


if __name__ == '__main__':
    shape_calculator = ShapeCalculator()
    cover_cal = CoverageCalculator()
    result_analyser = ResultAnalyser()

    model_json = 'ResNet50'
    # model_json = 'vgg16'
    has_mutate = True
    has_MCTS = True
    terminate_condition = 100
    data_set = 'cifar10'
    if has_mutate:
        mutate_flag = 'mutation'
    else:
        mutate_flag = 'noMutation'
    if has_MCTS:
        search_flag = 'mcts'
    else:
        search_flag = 'rand'
    # log_file = f'test_{model_json.lower()}_{mutate_flag}_{data_set.lower()}_0{terminate_condition}.txt'
    log_file = 'result_0025.txt'
    json_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'config', model_json, f'{model_json}.json')
    res_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'Results', log_file)
    print('json_dir',json_dir)

    graph = json_to_graph(json_dir)
    graph.display()
    print('-' * 50)
    config_dict = {'log_dir': res_dir, 'model_cnt': 0}

    data_set = data_set.lower()
    batch_size = 1
    input_shape = [batch_size, 3, 32, 32]
    data_load = RandomDataLoader(input_shape)
    if data_set == 'mnist':
        data_load = MnistDataLoader()
        input_shape = [batch_size, 1, 28, 28]
    elif data_set == 'cifar10':
        data_load = Cifar10DataLoader()
        input_shape = [batch_size, 3, 32, 32]
    data_gen = data_load.data_gen()

    shape_calculator.set_shape(graph, input_shape=input_shape)

    cover_set = set()
    model_cnt = 0
    crash_model = 0

    for i in range(terminate_condition):
        data = next(data_gen)
        print("iter", i)
        if i == 10 or i == 50 or i == 100:
            result_analyser.statics_bugs(config_dict=config_dict)
        if has_MCTS:
            sub_g = block_chooser(graph, 10)
        else:
            sub_g = random_graph(graph, i % 10 + 1)
        try:
            shape_calculator.set_shape(sub_g, input_shape=input_shape)
        except BaseException as e:
            print(f'calculate shape failure')
            continue

        nodes_path = [node.id for node in sub_g.nodes.values() if node.id in graph.nodes.keys()]

        print('sub graph:')
        sub_g.display()
        try:
            if has_mutate:
                mutation_sets = [randint(0, 100) for _ in range(min(i, 8))]
                mutation_selector = MutationSelector(mutation_sets, r=1)
                mutation_selector.mutate(sub_g)
                print('after mutate')
                sub_g.display()
        except BaseException as e:
            print('mutation failure')
            continue

        print(f'generate model and compute')
        coverage = cover_cal.get_cover(sub_g, graph)

        # torch_model = TorchModel(sub_g)
        # tf_model = TensorflowModel(sub_g)
        # mindspore_model = MindSporeModel(sub_g)
        model_dict = {}
        try:
            model_dict = {'torch': TorchModel(sub_g), 'tensorflow': TensorflowModel(sub_g),
                          'mindspore': MindSporeModel(sub_g)}
        except BaseException as e:
            print(f'model generate failure in iter:{i}')
            continue

        print(f'model_cnt:{model_cnt}')
        cover_set.add(coverage)

        torch_res = {}
        tf_res = {}
        ms_res = {}
        res_dict = {'tensorflow': {}, 'torch': {}, 'mindspore': {}}
        config_dict['iter'] = i
        torch_exception = ''
        tf_exception = ''
        ms_exception = ''

        with ThreadPoolExecutor(max_workers=1) as pool:
            for frame in model_dict.keys():
                model_task = pool.submit(model_run,
                                         model=model_dict[frame], frame_work=frame, res_dict=res_dict,
                                         data=data)
                try:
                    model_task.result(timeout=3)
                except BaseException as e:
                    res_dict[frame]['exception_info'] = e
                    print(f'{frame} compute failure:{e}')

        print(f'frame work compute complete, starting compare')

        flag_tf = res_dict['tensorflow']['flag_run']
        flag_torch = res_dict['torch']['flag_run']
        flag_ms = res_dict['mindspore']['flag_run']

        if flag_tf == flag_ms and flag_ms == flag_torch and not flag_torch:
            '''all have crash'''
            crash_model += 1
            continue
        model_cnt += 1
        config_dict['model_cnt'] = model_cnt

        sub_g_des = sub_g.get_des()
        sub_g_src = sub_g.get_src()

        if not (flag_tf == flag_ms and flag_ms == flag_torch):
            tf_track = model_dict['tensorflow'].exception_track
            torch_track = model_dict['torch'].exception_track
            ms_track = model_dict['mindspore'].exception_track
            res_dict['tensorflow']['track'] = tf_track
            res_dict['torch']['track'] = torch_track
            res_dict['mindspore']['track'] = ms_track

            result_analyser.analyse_exception(config_dict=config_dict, res_dict=res_dict)
            back_propagation(graph, nodes_path)
            continue
            # break

        print(f'compare done, write into log')
        r'''res[0]:output
            res[1]:layer_result {id:name,output,output_size,from,to}
        '''

        if not (np.allclose(res_dict['torch']['final_res'], res_dict['tensorflow']['final_res'], 1e-3)
                and np.allclose(res_dict['torch']['final_res'], res_dict['mindspore']['final_res'], 1e-3)
                and np.allclose(res_dict['tensorflow']['final_res'], res_dict['mindspore']['final_res'], 1e-3)):
            if not (np.isnan(res_dict['torch']['final_res']).all() and np.isnan(
                    res_dict['tensorflow']['final_res']).all() and np.isnan(res_dict['mindspore']['final_res']).all()):
                has_bug = result_analyser.analyse_arrays(graph=sub_g, config_dict=config_dict, res_dict=res_dict)

                back_propagation(graph, nodes_path)

                # if has_bug:
                #     break

    result_analyser.statics_bugs(config_dict=config_dict)
    print('done')
