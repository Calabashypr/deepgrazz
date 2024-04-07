from random import randint
from queue import Queue


def first_missing_positive(nums):
    n = len(nums)
    for i in range(n):
        while 0 <= nums[i] < n and nums[nums[i]] != nums[i]:
            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
    for i in range(n):
        if nums[i] != i:
            return i
    return n


def probability(rate):
    if rate == 0:
        return False
    if rate == 1:
        return True
    if rate == 0.2:
        return randint(0, 4) == 0
    if rate == 0.5:
        return randint(0, 1) == 0
    if rate == 0.3:
        return randint(0, 9) <= 2
    sample = int(1 / rate)
    p = randint(0, sample)
    if p == 1:
        return True
    return False


def cal_nodes_nums(g):
    all_op = set()
    for node in g.nodes.values():
        all_op.add(hash(node))
    return len(all_op)


def topo_sort(g):
    r"""
    judge whether graph g is dag, and return the topo seq of g
    :param g: graph
    :return: is_topo_sort, the seq of topo_sort
    """
    print(f'come into topo_sort')
    # print(f'current graph is:\n{g.display()}')
    seq = []
    in_degree = {k: 0 for k, v in g.nodes.items()}
    for node_ in g.nodes.values():
        for to_ in node_.to_nodes:
            in_degree[to_] += 1
    q = Queue()
    for i in in_degree:
        if in_degree[i] == 0:
            q.put(i)
    while not q.empty():
        x = q.get()
        seq.append(x)
        for t in g.nodes[x].to_nodes:
            in_degree[t] -= 1
            if in_degree[t] == 0:
                q.put(t)
    print('current topo queue is ')
    print(seq)
    return len(seq) == len(g.nodes), seq


def get_element_nums(shape):
    nums = 1
    for i in shape:
        nums *= i
    return nums


def get_element_det(shape1, shape2):
    r"""
    :param shape1:
    :param shape2:
    :return: the diff(det) between shape1 and shape2
    """
    return get_element_nums(shape1) - get_element_nums(shape2)


def get_random_node2(nums):
    r"""select two elements from nums:list"""
    a = nums[randint(0, len(nums) - 1)]
    b = nums[randint(0, len(nums) - 1)]
    while a == b:
        b = nums[randint(0, len(nums))]
    return a, b


def get_random_num(nums):
    r"""select one element from nums:list"""
    return nums[randint(0, len(nums) - 1)]
