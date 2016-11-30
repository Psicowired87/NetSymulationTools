
"""


"""

import os
import numpy as np
import networkx as nx

from ..NetSimulation.dynamics_evolution import meta_evolution
from ..NetSimulation.requester_info import get_node_dynamic_info,\
    get_node_list_structure_info, get_node_dicts_structure_info,\
    get_node_structure_info, from_net2dicts

from ..NetSimulation.dynamic_operations import game_life_evolution_f,\
    next_step_node
from ..NetSimulation.aux_functions import steps_to_do_compute,\
    initialization_states, discretize_with_thresholds
from ..NetSimulation.dynamics_evolution import compute_next_state, evolution


def test():
    n_steps = 5
    net_ini = np.random.random((1, 40*40))
    net_edge = [(np.random.randint(0, 10), np.random.randint(0, 10))
                for i in range(20)]
    netx = nx.to_networkx_graph(np.random.random((10, 10))).to_directed()
    netnullx = nx.to_networkx_graph(net_edge).to_directed()
    netnullx.add_nodes_from(range(10))

    #### Requester info
    ###########################################################################
    ###########################################################################
    pre_states = np.random.random((1, 10))
    node = 1
    neighbours = [2, 3, 4]
    get_node_dynamic_info(node, pre_states, neighbours)

    ## TO DEPRECATE:
    ################
    node = 0
    node_list = range(10)
    get_node_structure_info(node, netx, node_list)
    ##############################################
    nodes_index = [0, 1, 2, 3]
    neighs_index = [[0, 1], [1, 2], [4, 2], [1, 2]]
    ws_e_neig = [[0.5, 1.2], [1.2, 2.4], [4.1, 2.1], [1., 2.]]
    ws_n_neig = [[0.5, 1.2], [1.2, 2.4], [4.1, 2.1], [1., 2.]]
    w_nodes = [.5, .6, .1, .3]
    net_info = nodes_index, neighs_index, ws_e_neig, ws_n_neig, w_nodes
    pre_states = np.random.random((1, 10))

    ## From net 2 dicts info
    from_net2dicts(net_info, pre_states)

    # Get node info
    nodes, node_list = [0, 2, 3], range(10)
    useweights = [True, True]
    get_node_list_structure_info(nodes, netx, node_list)
    get_node_list_structure_info(nodes, netx, node_list, useweights)
    get_node_list_structure_info(nodes, netnullx, node_list)
    get_node_list_structure_info(nodes, netnullx, node_list, useweights)

    get_node_dicts_structure_info(pre_states, nodes, netx, node_list)
    get_node_dicts_structure_info(pre_states, nodes, netx, node_list,
                                  useweights)
    get_node_dicts_structure_info(pre_states, nodes, netnullx, node_list)
    get_node_dicts_structure_info(pre_states, nodes, netnullx, node_list,
                                  useweights)

    ### dynamic operations
    ###########################################################################
    ###########################################################################
    pre_states_node = np.random.randint(0, 2, (1, 10))
    pre_states_neig = np.random.randint(0, 2, (10, 10))
    game_life_evolution_f(pre_states_node, pre_states_neig)

    variables = {'pre_states_node': pre_states_node,
                 'pre_states_neig': pre_states_neig}
    methods = [game_life_evolution_f, 'conways', '']
    for i in range(len(methods)):
        next_step_node(methods[i], variables)

    ### dynamic operations
    ###########################################################################
    ###########################################################################
    ## Managing tasks splitting
    steps_to_do_compute(100, 10, 20, 50)
    steps_to_do_compute(100, 10, 20, 500)
    steps_to_do_compute(100, 10, 15, 20)
    steps_to_do_compute(100, 10, 15, np.inf)

    ## Initialization of the state
    state = initialization_states(10, 2, .2)
    assert(len(state.shape) == 2)
    assert(state.shape[0] == 1)
    state = initialization_states(10, 2)
    assert(len(state.shape) == 2)
    assert(state.shape[0] == 1)
    state = initialization_states(10, 3)
    assert(len(state.shape) == 2)
    assert(state.shape[0] == 1)
    state = initialization_states(10, 3, [.3, .6])
    assert(len(state.shape) == 2)
    assert(state.shape[0] == 1)
    state = initialization_states(10, range(2, 5), [.3, .6])
    assert(len(state.shape) == 2)
    assert(state.shape[0] == 1)

    ## Discretization of array
    array = np.random.random(100)
    thr_array = discretize_with_thresholds(array, 0.2)
    assert(array.shape == thr_array.shape)
    thr_array = discretize_with_thresholds(array, [0.2, .5])
    assert(array.shape == thr_array.shape)
    thr_array = discretize_with_thresholds(array, [0.2, .5], [0, 1, 2])
    assert(array.shape == thr_array.shape)

    array = np.random.random((100, 2))
    thr_array = discretize_with_thresholds(array, 0.2)
    assert(array.shape == thr_array.shape)
    thr_array = discretize_with_thresholds(array, [0.2, .5])
    assert(array.shape == thr_array.shape)
    thr_array = discretize_with_thresholds(array, np.array([0.2, .5]))
    assert(array.shape == thr_array.shape)

    array = np.random.random((2, 100))
    thr_array = discretize_with_thresholds(array, np.array([0.2, .5]))
    assert(array.shape == thr_array.shape)

    thr = np.random.random((2, 100))
    thr_array = discretize_with_thresholds(array, thr)
    assert(array.shape == thr_array.shape)
    thr = np.random.random((2, 100))
    thr_array = discretize_with_thresholds(array, thr.T)
    assert(array.shape == thr_array.shape)
    thr = np.cumsum(np.random.random((2, 100, 3)), axis=2)
    thr_array = discretize_with_thresholds(array, thr)
    assert(array.shape == thr_array.shape)

    ######################### Evolution for each node #########################
    ###########################################################################
    pre_states = np.random.randint(0, 2, (1, len(netx.nodes())))
    node_list = netx.nodes()
    next_states = compute_next_state(pre_states, node_list, netx,
                                     syncronicity=1, parallelize=None,
                                     useweights=[False, False])
    assert(type(next_states) == np.ndarray)
    assert(len(next_states.shape) == 1)
    assert(len(next_states) == pre_states.shape[1])
    next_states = compute_next_state(pre_states, node_list, netx,
                                     syncronicity=.5, parallelize=None,
                                     useweights=[True, True])
    assert(type(next_states) == np.ndarray)
    assert(len(next_states.shape) == 1)
    assert(len(next_states) == pre_states.shape[1])
    next_states = compute_next_state(pre_states, node_list, netx,
                                     syncronicity=0, parallelize=None,
                                     useweights=[True, True])
    assert(type(next_states) == np.ndarray)
    assert(len(next_states.shape) == 1)
    assert(len(next_states) == pre_states.shape[1])

    ############################ Evolution for net ############################
    ###########################################################################
    node_list = range(10)
    n_steps = 1000
    sync, t_mem0, t_mem1 = 1, 0, 10
    useweights0 = [False, False]
    useweights1 = [True, True]
    pre_stts = np.random.randint(0, 2, (5, 10))

    dynamics = evolution(netx, node_list=node_list, n_steps=n_steps,
                         pre_states=[], syncronicity=sync, parallelize=None,
                         useweights=useweights0, is_static=True, t_mem=t_mem0)
    assert(type(dynamics) == np.ndarray)
    assert(len(dynamics) == (n_steps+1))
    assert(dynamics.shape[1] == len(node_list))

    dynamics = evolution(netx, node_list=range(10), n_steps=n_steps,
                         pre_states=[], syncronicity=1, parallelize=None,
                         useweights=useweights0, is_static=True, t_mem=t_mem1)
    assert(type(dynamics) == np.ndarray)
    assert(len(dynamics) == (n_steps+1))
    assert(dynamics.shape[1] == len(node_list))

    dynamics = evolution(netx, node_list=range(10), n_steps=n_steps,
                         pre_states=pre_stts, syncronicity=.5, parallelize=None,
                         useweights=useweights1, is_static=True, t_mem=t_mem1)
    assert(type(dynamics) == np.ndarray)
    assert(len(dynamics) == (n_steps+len(pre_stts)))
    assert(dynamics.shape[1] == len(node_list))

    ########################## Metaevolution for net ##########################
    ###########################################################################
    dynamics = meta_evolution(netx, n_steps, pre_states=[], syncronicity=1,
                              parallelize=None, useweights=useweights0,
                              is_static=True, t_mem=0, aux_file=None,
                              maxstore_fly=None)
    dynamics = meta_evolution(netx, n_steps, pre_states=pre_stts, syncronicity=1,
                              parallelize=None, useweights=[False, False],
                              is_static=True, t_mem=1, aux_file='',
                              maxstore_fly=None)
    dynamics = meta_evolution(netx, n_steps, pre_states=pre_stts, syncronicity=1,
                              parallelize=None, useweights=[False, False],
                              is_static=True, t_mem=1, aux_file='',
                              maxstore_fly=100)
    lisdirs = os.listdir(os.path.dirname(os.path.abspath(__file__)))
    lisdirs = os.listdir(os.getcwd())
    rmdirs = [e for e in lisdirs if 'store_file' in e]
    for e in rmdirs:
        os.remove(e)
