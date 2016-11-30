
"""
Evolution module which contains the main functions for simulate a system.
"""

import numpy as np
import pandas as pd
from aux_functions import steps_to_do_compute, initialization_states
from requester_info import get_node_structure_info,\
    get_node_list_structure_info
from dynamic_operations import next_step_node


def meta_evolution(net, n_steps=1000, pre_states=[], syncronicity=1,
                   parallelize=None, useweights=[False, False], is_static=True,
                   t_mem=0, aux_file=None, maxstore_fly=None):
    """Main function develop the evolution dealing with the memory problems.

    Parameters
    ----------
    net: np.ndarray
        the network support.
    n_steps: int (default=1000)
        the number of steps of the iteration.
    pre_states: array_like shape(Np,M) (default=[])
        the previous Np states of the system of each M elements. The memory.
    syncronicity: int (default=1)
        the number of updates at the same time.
    parallelize: str, default None
        how to parallelize the updating of the states.
        * If None, no parallelization.
        * If 'cpu', parallelization throught CPU
        * If 'gpu', parallelization throught GPU
    useweights: list, tuple 2-components
        specify if we use the weights:
        * If first True, we use edge weights.
        * If second True, we use node weights.
    is_static: boolean
        if the network do not change along the simulation.
    t_mem: int (default=0)
        number of previous times which the system uses to update the next
        state. It is called the memory of the system.
    aux_file: str (default=None)
        the auxiliar file in which is going to do ROM support.
    maxstore_fly: integer (default=None)
        max steps to compute in memory.

    Returns
    -------
    dynamics: array_like, shape (n_steps,len(node_list))
        the description of the evolution of the whole system.

    """
    ## 0. Checking inputs
    # Check nodes of the system
    node_list = net.nodes()
    n = len(node_list)

    ## 1. Obtain the initial state.
    n = len(net.nodes())
    if type(pre_states) != np.ndarray:
        # Generate first state
        pre_states = initialization_states(n)
        n_pre_states = 1
    else:
        n_pre_states = pre_states.shape[0]

    ## 2. Compute the number of steps for each part of the tasks
    maxstore_fly = np.inf if maxstore_fly is None else maxstore_fly
    steps_to_do = steps_to_do_compute(n_steps, n_pre_states,
                                      t_mem, maxstore_fly)
    # Set aux_file
    if aux_file is None:
        pass
    elif not aux_file:
        aux_file = 'store_file.csv'
    if type(aux_file) == str:
        numbs = ['00'+str(i) if i < 10 else ('0'+str(i) if i < 100 else str(i))
                 for i in range(len(steps_to_do))]
        aux_file = [aux_file+numbs[i] for i in range(len(steps_to_do))]

    ## 3. Evolution
    dynamics = evolution(net, node_list, n_steps, pre_states, syncronicity,
                         parallelize, useweights, is_static, t_mem)

    ## 4. Store if it is needed
    if aux_file is not None:
        # Write
        dynamics = pd.DataFrame(dynamics, columns=net.nodes())
        dynamics.to_csv(aux_file[0])
        # Next write info
        aux_file = aux_file[1:]
    steps_to_do = steps_to_do[1:]

    ## 5. Loop of the nexts phases
    while steps_to_do:
        dynamics = dynamics.as_matrix()[-t_mem:, :]
        dynamics = evolution(net, node_list, n_steps, dynamics, syncronicity,
                             parallelize, useweights, is_static, t_mem)
        if aux_file is not None:
            # Write
            dynamics = pd.DataFrame(dynamics, columns=net.nodes())
            dynamics.to_csv(aux_file[0])
            # Next write info
            aux_file = aux_file[1:]
        steps_to_do = steps_to_do[1:]

    ## 6. Preparation out
    dynamics = pd.DataFrame(dynamics, columns=node_list)

    return dynamics


def evolution(net, node_list=[], n_steps=1000, pre_states=[], syncronicity=1,
              parallelize=None, useweights=[False, False], is_static=True,
              t_mem=0):
    """The main function to run a simulation in which we could specify a lot
    of parameters and select or input the update rule.

    Parameters
    ----------
    net: networkx.DiGraph
        the network information needed to compute the next state.
    node_list: list
        the list of all the nodes of the system.
    n_steps: init
        number of time steps in the simulation.
    pre_states: array_like shape(Np,M)
        the previous Np states of the system of each M elements.
    syncronicity: float
        how syncronous would be the update.
    parallelize: str, default None
        how to parallelize the updating of the states.
        * If None, no parallelization.
        * If 'cpu', parallelization throught CPU
        * If 'gpu', parallelization throught GPU
    useweights: list, tuple 2-components
        specify if we use the weights:
        * If first True, we use edge weights.
        * If second True, we use node weights.
    is_static: boolean
        if the network do not change along the simulation.
    t_mem: int (default=0)
        number of previous times which the system uses to update the next
        state. It is called the memory of the system.

    Returns
    -------
    dynamics: array_like, shape (n_steps,len(node_list))
        the description of the evolution of the whole system.

    """

    ## 0. Checking inputs (to be substituted for the error module)
    # Check nodes of the system
    if not node_list:
        node_list = net.nodes()
    failnodes = [e for e in net.nodes() if e not in node_list]
    if failnodes:
        raise AssertionError("There are nodes in network not in the system.")

    ## 1. Obtain the initial state.
    n = len(net.nodes())
    if pre_states == []:
        # Generate first state
        pre_states = initialization_states(n)

    # Prepare memory to store the whole dynamics.
    n_pre = 1 if len(pre_states.shape) == 1 else len(pre_states)
    pre_idxs = 0 if len(pre_states.shape) == 1 else range(n_pre)
    dynamics = np.zeros([n_steps+n_pre, len(node_list)])
    dynamics[pre_idxs, :] = pre_states

    # Calculate the possible information of the net required
    if is_static and syncronicity == 1 and bool(parallelize):
        net = get_node_list_structure_info(node_list, net, node_list,
                                           useweights)

    # Sequential iteration for each times.
    for i in xrange(1, n_steps+1):
        dynamics[i, :] = compute_next_state(pre_states, node_list, net,
                                            syncronicity, parallelize,
                                            useweights)
        # Define t_ini in order to only pass the useful previous states
        if t_mem == 0 or i - t_mem <= 0:
            t_ini = 0
        else:
            t_ini = i - t_mem
        pre_states = dynamics[t_ini:i+1, :]

    return dynamics


def compute_next_state(pre_states, node_list, net, syncronicity=1,
                       parallelize=None, useweights=[False, False]):
    """This function make evolve the system to the next state. Could use the
    network structure and the previous states.

    Parameters
    ----------
    pre_states : array_like, shape (N, M)
        the values of the time history of the signal. N times and M elements.
    node_list : list
        the list of all the nodes of the systes.
    net : networkx.DiGraph or tuple
        the network information needed to compute the next state.
        * If networkx.DiGraph, information stored in networkx way.
        * If tuple, information stored in list to be iterated by.
    syncronicity: float
        how syncronous is the updating of the states in the simulation.
        * If 0, update only one element.
        * otherwise, proportion of elements to be updated each step.
    parallelize: str, default None
        how to parallelize the updating of the states.
        * If None, no parallelization.
        * If 'cpu', parallelization throught CPU
        * If 'gpu', parallelization throught GPU
    useweights: list, tuple 2-components
        specify if we use the weights:
        * If first True, we use edge weights.
        * If second True, we use node weights.

    Returns
    -------
    ys : ndarray, shape (1, pre_states.shape[1])
        the next state of the system.
        It has the state of each element of the system at one time.

    See also
    --------
    next_step_node, evolution

    Notes
    -----

    Examples
    --------
    >>> from net_formation import grid_2d_graph_2order
    >>> net = grid_2d_graph_2order(100,100)
    >>> len(net.nodes())
    10000
    >>> len(net.edges())
    80000
    >>> dyn = evolution(net, n_steps=1000)
    >>> dyn.shape
    (10000,1000)

    """
    ## 1. Syncronicity
    n = len(node_list)
    # Preparing the next state considering the syncronicity
    if syncronicity == 1:
        # Prepare all for the next step
        next_states = np.zeros(n)
        nodes_idx = list(node_list)
    elif syncronicity == 0:
        # Prepare only one for the next state
        next_states = pre_states[-1, :]
        nodes_idx = list(np.random.randint(0, n, 1))
    else:
        # Prepare some for the next state
        next_states = pre_states[-1, :]
        nodes_idx = list(np.random.permutation(n)[:np.around(n*syncronicity)])

    ## 2. Compute next state in the parralel way
    parallelize = False if parallelize not in ['cpu' 'gpu'] else parallelize
    if not parallelize:
        # For each node to be updated, search the new state
        for node in nodes_idx:
            # Previous magnitudes needed
            node_index, neigh_index = get_node_structure_info(node, net,
                                                              node_list)
            # Computation of the next state
            jnod = node_list.index(node)
            kwargs = {'pre_states_node': pre_states[:, [node_index]],
                      'pre_states_neig': pre_states[:, neigh_index]}
            kwargs = {'method': 'conways', 'variables': kwargs}
            next_states[jnod] = next_step_node(**kwargs)

    elif parallelize == 'cpu':
        pass
    elif parallelize == 'gpu':
        pass
    return next_states
