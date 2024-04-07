from version2.graph import Node, Graph
from version2.model_gen.OperatorSet import operator_set
import numpy as np
import os


class ResultAnalyser:
    def __init__(self, det=1e-3):
        """
        :param det:
        """
        self.det = det
        self.frame_works = ['tensorflow', 'torch', 'mindspore']
        self.bugs = {
            'tensorflow': {'nums': 0, 'distinct_bugs': 0, 'operators': {}},
            'mindspore': {'nums': 0, 'distinct_bugs': 0, 'operators': {}},
            'torch': {'nums': 0, 'distinct_bugs': 0, 'operators': {}},

        }

    def count_bugs(self, name, tf_torch, torch_ms, tf_ms):
        r"""
        :param name:        operator name
        :param tf_torch:    diff test result between tensorflow and torch
        :param torch_ms:    diff test result between torch and mindspore
        :param tf_ms:       diff test result between tensorflow and mindspore
        :return:
        """

        if tf_torch and torch_ms and tf_ms:
            print(f'no bugs found')
        elif tf_torch and torch_ms and not tf_ms:
            self.bugs['tensorflow']['nums'] += 1
            self.bugs['mindspore']['nums'] += 1
            self.bugs['tensorflow']['operators'][name] = self.bugs['tensorflow']['operators'].get(name, 0) + 1
            self.bugs['mindspore']['operators'][name] = self.bugs['mindspore']['operators'].get(name, 0) + 1
        elif tf_torch and not torch_ms and tf_ms:
            self.bugs['torch']['nums'] += 1
            self.bugs['mindspore']['nums'] += 1
            self.bugs['torch']['operators'][name] = self.bugs['torch']['operators'].get(name, 0) + 1
            self.bugs['mindspore']['operators'][name] = self.bugs['mindspore']['operators'].get(name, 0) + 1
        elif not tf_torch and torch_ms and tf_ms:
            self.bugs['tensorflow']['nums'] += 1
            self.bugs['torch']['nums'] += 1
            self.bugs['tensorflow']['operators'][name] = self.bugs['tensorflow']['operators'].get(name, 0) + 1
            self.bugs['torch']['operators'][name] = self.bugs['torch']['operators'].get(name, 0) + 1
        elif tf_torch and not torch_ms and not tf_ms:
            self.bugs['mindspore']['nums'] += 1
            self.bugs['mindspore']['operators'][name] = self.bugs['mindspore']['operators'].get(name, 0) + 1
        elif not tf_torch and not torch_ms and tf_ms:
            self.bugs['torch']['nums'] += 1
            self.bugs['torch']['operators'][name] = self.bugs['torch']['operators'].get(name, 0) + 1
        elif not tf_torch and torch_ms and not tf_ms:
            self.bugs['tensorflow']['nums'] += 1
            self.bugs['tensorflow']['operators'][name] = self.bugs['tensorflow']['operators'].get(name, 0) + 1
        else:
            self.bugs['tensorflow']['nums'] += 1
            self.bugs['mindspore']['nums'] += 1
            self.bugs['torch']['nums'] += 1
            self.bugs['mindspore']['operators'][name] = self.bugs['mindspore']['operators'].get(name, 0) + 1
            self.bugs['tensorflow']['operators'][name] = self.bugs['tensorflow']['operators'].get(name, 0) + 1
            self.bugs['torch']['operators'][name] = self.bugs['torch']['operators'].get(name, 0) + 1
            # self.bugs['all_frameworks']['nums'] += 1
        self.update_distinct_bugs()

    def judge_consistency(self, tf_res, torch_res, ms_res):
        # print(f'debug judge_consistency')
        # print(f'tf:{tf_res}')
        # print(f'torch:{torch_res}')
        # print(f'ms:{ms_res}')
        tf_torch = np.allclose(tf_res, torch_res, self.det)
        torch_ms = np.allclose(torch_res, ms_res, self.det)
        tf_ms = np.allclose(tf_res, ms_res, self.det)

        return {'tf_torch': tf_torch, 'torch_ms': torch_ms, 'tf_ms': tf_ms}

    def analyse_arrays(self, graph: Graph, config_dict, res_dict):
        r"""

        :param graph:
        :param config_dict: {log_dir, iter}
        :param res_dict: {frame_work:{
                                final_res,          final result
                                layer_res,          {id:{name, output, output_shape, from, to}}
                                exception_info,     exception_message
                                track               {id, name, framework, input_datas}
                                }}
        :return:
        """
        log_dir = config_dict['log_dir']
        tf_layer_res = res_dict['tensorflow']['layer_res']
        torch_layer_res = res_dict['torch']['layer_res']
        ms_layer_res = res_dict['mindspore']['layer_res']
        has_bug = False
        with open(log_dir, 'a', encoding='utf8') as f:
            f.write(f"\nanalyse output arrays in iter:{config_dict['iter']}\n")
            for cur in graph.nodes.keys():
                # print(f'cur={cur} tf_result[{cur}]:{tf_layer_res[cur]},torch_result[{cur}]:{torch_layer_res[cur]},'
                #       f'ms_result[{cur}]:{ms_layer_res[cur]}\n')
                name = tf_layer_res[cur]['name']
                consist_cur = self.judge_consistency(tf_res=tf_layer_res[cur]['output'],
                                                     torch_res=torch_layer_res[cur]['output'],
                                                     ms_res=ms_layer_res[cur]['output'])

                if self.has_bugs(consist_cur):
                    cur_is_bug = True
                    for pre in graph.nodes[cur].from_nodes:
                        consist_pre = self.judge_consistency(tf_res=tf_layer_res[pre]['output'],
                                                             torch_res=torch_layer_res[pre]['output'],
                                                             ms_res=ms_layer_res[pre]['output'])
                        if self.has_bugs(consist_pre):
                            cur_is_bug = False
                            break
                    if cur_is_bug:
                        torch_ms = consist_cur.get('torch_ms', consist_cur.get('ms_torch'))
                        tf_torch = consist_cur.get('tf_torch', consist_cur.get('torch_tf'))
                        tf_ms = consist_cur.get('tf_ms', consist_cur.get('ms_tf'))
                        self.count_bugs(name=name, tf_torch=tf_torch, torch_ms=torch_ms, tf_ms=tf_ms)
                        f.write(f'\npre layer res:\n')
                        for pre in graph.nodes[cur].from_nodes:
                            f.write(f"{pre}:{tf_layer_res[pre]['name']}\n")
                            f.write(f"{tf_layer_res[pre]}\n")

                        f.write(f"tf node:\n{tf_layer_res[cur]}\n")
                        f.write(f"ms node:\n{ms_layer_res[cur]}\n")
                        f.write(f"torch node:\n{torch_layer_res[cur]}\n")
                        has_bug = True
            f.write(f"\ngenerate models:{config_dict['model_cnt']}\n")
        print(self.bugs)
        return has_bug

    def analyse_exception(self, config_dict, res_dict):
        r"""

        :param config_dict: {log_dir, iter, model_cnt}
        :param res_dict: {frame_work:{
                                final_res,          final result
                                layer_res,          {id:{name, output, output_shape, from, to}}
                                exception_info,     exception_message
                                track               {id, name, framework, input_datas}
                                }}

        :return:
        """
        log_dir = config_dict['log_dir']
        flag_all_have_bugs = True
        with open(log_dir, 'a', encoding='utf8') as f:
            f.write(f"\nanalyse the exceptions in iter:{config_dict['iter']}\n")
            for framework, result in res_dict.items():
                if len(result['track']) != 0:
                    name = result['track']['name'].lower()
                    self.bugs[framework]['nums'] += 1
                    cnt = self.bugs[framework]['operators'].get(name, 0)
                    self.bugs[framework]['operators'][name] = cnt + 1
                    self.bugs[framework]['distinct_bugs'] = len(self.bugs[framework]['operators'])

                    f.write(f'{framework} exception:\n')
                    f.write(f"{result['track']}\n")
                    f.write(f"{result['exception_info']}\n")
                else:
                    flag_all_have_bugs = False
            # if flag_all_have_bugs:
            # self.bugs['all_frameworks']['nums'] += 1

            f.write(f"\ngenerate models:{config_dict['model_cnt']}\n")
        print(f'\n{self.bugs}\n')

    def statics_bugs(self, config_dict):
        print(f'debug ResultAnalyser statics_bugs')
        log_dir = config_dict['log_dir']
        self.update_distinct_bugs()

        print(self.bugs)
        with open(log_dir, 'a', encoding='utf8') as f:
            f.write('\nfinal statics:\n')
            f.write(f"total operators:{len(operator_set)}\n")
            for framework, result in self.bugs.items():
                f.write(f"{framework} --> nums:{result['nums']},distinct_bugs:{result['distinct_bugs']}\n")

            for framework, result in self.bugs.items():
                f.write(f"{framework} --> \n")
                for op, nums in result['operators'].items():
                    f.write(f"{op}:{nums}\n")
            f.write(f"\ngenerate models:{config_dict['model_cnt']}\n")
        for framework, result in self.bugs.items():
            print(f"{framework} --> nums:{result['nums']},distinct_bugs:{result['distinct_bugs']}\n")

    def update_distinct_bugs(self):
        self.bugs['tensorflow']['distinct_bugs'] = len(self.bugs['tensorflow']['operators'])
        self.bugs['torch']['distinct_bugs'] = len(self.bugs['torch']['operators'])
        self.bugs['mindspore']['distinct_bugs'] = len(self.bugs['mindspore']['operators'])
        bug_ops = {}
        for frame_work in self.frame_works:
            for name, cnt in self.bugs[frame_work]['operators'].items():
                bug_ops[name] = bug_ops.get(name, 0) + cnt
        # self.bugs['all_frameworks']['distinct_bugs'] = len(bug_ops)

    def has_bugs(self, consist_map: dict):
        for k, v in consist_map.items():
            if not v:
                return True
        return False
