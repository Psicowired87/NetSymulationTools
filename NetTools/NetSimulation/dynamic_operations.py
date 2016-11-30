
"""
Module which groups the different possible dynamics to evolve a system.

TODO
----
- More options.

"""

import numpy as np


def next_step_node(method, variables):
    """This function acts as a switcher between the available methods on this
    module or apply the one given by method.

    Parameters
    ----------
    method: str or function
        method used to obtain the next state of the node from its neighbours.
    variables: dict
        the information needed to compute the next state of the node. It is
        usually composed by:
        - pre_states_node: the previous states of the node to evolve.
        - pre_states_neig: the previous states of the neighbours.
        ------------
        - the weights of nodes.
        - the weights of edges.

    Returns
    -------
    next_state_node: int or float
        the next state of the node to be updated.

    """

    method = 'conways'

    if type(method).__name__ == 'function':
        next_state_node = method(**variables)
    elif method == 'conways':
        # Filter the input variables to only the ones needed.
        needed = ['pre_states_node', 'pre_states_neig']
        variables = dict([(k, v) for k, v in variables.items()
                          if k in needed])
        # Computation of the next state.
        next_state_node = game_life_evolution_f(**variables)
    else:
        next_state_node = game_life_evolution_f(variables['pre_states_node'],
                                                variables['pre_states_neig'])

    return next_state_node


def game_life_evolution_f(pre_states_node, pre_states_neig):
    """This functions recreates the evolution step of the original Conways
    game of life.

    Parameters
    ----------
    pre_states_node: array_like, shape (Ntmem, 1)
        the previous states of the node considered to be updated.
    pre_states_neig: array_like, shape (Ntmem, M)
        the previous states of the M neighbours.

    Returns
    -------
    next_state: int or float
        the next state of the node to be updated.

    """
    assert(len(pre_states_node.shape) == 2)
    # From life state
    if pre_states_node[-1][0]:
        life_neig = np.sum(pre_states_neig[-1, :])
        next_state = life_neig == 2 or life_neig == 3
    # From dead state
    else:
        next_state = np.sum(pre_states_neig[-1, :]) == 3

    return next_state
