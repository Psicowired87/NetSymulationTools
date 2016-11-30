
"""
test_netstructure
-----------------
Testing structure functions.

"""

import numpy as np
from ..NetStructure.scaling_functions import thresholding_scores,\
    normalization_scores_min_max, standarization_scores, min_diagonal,\
    scale_connectome_like, scaling_scores
from ..NetStructure.net_formation import grid_2d_graph_2order


def test():
    n = 100
    network = np.random.random((n, n))

    def testing_score_scaling(scores, network):
        n_el = len(network)
        assert(type(scores) == np.ndarray)
        assert(scores.shape == (n_el, n_el))

    ## Scores scaling for network
    scores = thresholding_scores(network, 0.2)
    testing_score_scaling(scores, network)
    scores = normalization_scores_min_max(network, (0, 1))
    testing_score_scaling(scores, network)
    scores = standarization_scores(network)
    testing_score_scaling(scores, network)
    scores = min_diagonal(network)
    testing_score_scaling(scores, network)
    scores = scale_connectome_like(network)
    testing_score_scaling(scores, network)

    methods_list = [thresholding_scores, 'gap_normalization', 'thresholding',
                    'standarization', 'scale_connectome']
    args_list = [{'thr': .2},  {'limits': (0, 1)}, {'thr': .2}, {}, {}]

    for i in range(len(methods_list)):
        scaling_scores(network, methods_list[i], args_list[i])
        testing_score_scaling(scores, network)
    scaling_scores(network, methods_list, args_list)
    testing_score_scaling(scores, network)

    #### Network structures
    n, m = 20, 20
    net = grid_2d_graph_2order(n, m)
    assert(np.all([len(net.neighbors(node)) == 8 for node in net.nodes()]))
    grid_2d_graph_2order(m, n, periodic=True, directed=True)
    assert(np.all([len(net.neighbors(node)) == 8 for node in net.nodes()]))
    grid_2d_graph_2order(m, n, periodic=True, directed=False)
    assert(np.all([len(net.neighbors(node)) == 8 for node in net.nodes()]))
    grid_2d_graph_2order(m, n, periodic=False, directed=True)
    assert(np.all([len(net.neighbors(node)) == 8 for node in net.nodes()]))
    grid_2d_graph_2order(m, n, periodic=False, directed=False)
    assert(np.all([len(net.neighbors(node)) == 8 for node in net.nodes()]))
