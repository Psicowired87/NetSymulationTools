
"""
Module oriented to group all the functions useful to give support for the
main functions of this package of dynamic simulations.
"""

import numpy as np


def steps_to_do_compute(n_steps, n_pre_states, t_mem, maxstore_fly):
    """Function to distribute the times in different tasks in order to save
    memory.

    Parameters
    ----------
    n_steps: integer
        number of steps we want to compute.
    n_pre_states: integer
        number of previous states introduced in order to compute the evolution
        of the system in the next times.
    t_mem: integer
        number of previous times which the system uses to update the next
        state. It is called the memory of the system.
    maxstore_fly: integer
        max steps to compute in memory.

    Returns
    -------
    steps_to_go: list
        number of steps in memory for each bunch.

    """

    # Prepare inputs
    #maxstore_fly = np.inf
    assert(n_pre_states <= maxstore_fly)
    assert(t_mem < maxstore_fly)
#    assert(t_mem >= n_pre_states)

    steps_to_go = []
    if n_steps+n_pre_states <= maxstore_fly:
        steps_to_go.append(n_steps)
        return steps_to_go
    else:
        n_next = maxstore_fly-t_mem
        steps_to_go.append(n_next)
        steps_to_go_i =\
            steps_to_do_compute(n_steps-n_next, t_mem, t_mem, maxstore_fly)
        steps_to_go += steps_to_go_i
    assert(sum(steps_to_go) == n_steps)
    return steps_to_go


def initialization_states(n, n_states=2, init_threshold=0):
    """This function creates an initial state randomly with the properties
    specified in the inputs.

    Parameters
    ----------
    n: int
        number of elements of the system
    n_states: int, list, tuple
        length of the vocabulary of the system.
        * If int it has discrete states (the number of states specified).
        * If list or tuple, it is considered the possible states all the
        numbers between these two numbers.
    init_threshold: float, list
        in case of finite number of states it divides give the probability of
        being from one state or another using the numbers given in this var as
        a threshold.

    Returns
    -------
    initial_state: array_like shape (1, n)
        the initial state selected randomly.

    Examples
    --------
    >>> initialization_states(10)
    array([[0, 0, 1, 1, 1, 1, 1, 1, 0, 1]])
    >>>
    >>> initialization_states(10,4)
    array([[1, 1, 2, 1, 2, 3, 2, 1, 1, 0]])

    """
    # check
    # TODO: input dictionary with probability distribution
    assert type(n_states) in [int, list, tuple]

    if init_threshold == 0:
        if n_states == 2:
            # HARDCODED
            init_threshold = 0.67
        else:
            init_threshold = list(np.arange(1./n_states, 1, 1./n_states))[:-1]

    if type(n_states) == int:
        if type(init_threshold) == list:
            assert len(init_threshold) == n_states-1  # TO Insert in errors.

        # initiate
        aux = np.random.random_sample(n)

        initial_state = discretize_with_thresholds(aux, init_threshold)
        initial_state = initial_state.astype(int)
        initial_state = initial_state.reshape(-1)

    elif type(n_states) == tuple or type(n_states) == list:
        # initiate reescaling
        initial_state = np.random.random(n)
        initial_state = initial_state * (n_states[1] - n_states[0])
        initial_state = initial_state + n_states[0]

    ## Shaping in the correct way
    initial_state = initial_state.reshape(1, initial_state.shape[0])

    return initial_state


def discretize_with_thresholds(array, thres, values=[]):
    """This function uses the given thresholds in order to discretize the array
    in different possible values, given in the variables with this name.

    Parameters
    ----------
    array: array_like
        the values of the signal
    thres: float, list of floats or np.ndarray
        the threshold values to discretize the array.
    values: list
        the values assigned for each discretized part of the array.

    Returns
    -------
    aux: array
        a array with discretized values

    """

    ## 1. Preparing thresholds and values
    nd_input = len(array.shape)
    # Parche para 1d
    if nd_input == 1:
        array = array.reshape((array.shape[0], 1))

    # From an input of a float
    if type(thres) == float:
        nd_thres = 1
        thres = thres*np.ones((array.shape[0], array.shape[1], 1))
    # From an input of a list
    elif type(thres) == list:
        nd_thres = len(thres)
        aux = np.ones((array.shape[0], array.shape[1], nd_thres))
        for i in range(nd_thres):
            aux[:, :, i] = thres[i]*aux[:, :, i]
        thres = aux
    # From an input of a array (4 possibilities)
    elif type(thres) == np.ndarray:
        # each threshold for each different elements (elements)
        if len(thres.shape) == 1 and thres.shape[0] == array.shape[1]:
            nd_thres = 1
            aux = np.ones((array.shape[0], array.shape[1], nd_thres))
            for i in range(array.shape[1]):
                aux[:, i, 0] = thres[i]*aux[:, i, 0]
            thres = aux
        # each threshold for each different times (times)
        elif len(thres.shape) == 1 and thres.shape[0] == array.shape[0]:
            nd_thres = 1
            aux = np.ones((array.shape[0], array.shape[1], nd_thres))
            for i in range(array.shape[0]):
                aux[i, :, 0] = thres[i]*aux[i, :, 0]
            thres = aux
        # some threshold for each different elements (elemnts-thres)
        elif len(thres.shape) == 2 and thres.shape[0] == array.shape[1]:
            nd_thres = thres.shape[1]
            aux = np.ones((array.shape[0], array.shape[1], nd_thres))
            for i in range(array.shape[1]):
                aux[:, i, :] = thres[i, :]*aux[:, i, :]
            thres = aux
        # some threshods for each time shared by all elements (times-thres)
        elif len(thres.shape) == 2 and thres.shape[0] == array.shape[0]:
            nd_thres = thres.shape[1]
            aux = np.ones((array.shape[0], array.shape[1], nd_thres))
            for i in range(array.shape[0]):
                aux[i, :, :] = thres[i, :]*aux[i, :, :]
            thres = aux
        # one threshold for each time and element (times-elements)
#        elif len(thres.shape) == 2 and thres.shape[:2] == array.shape[:2]:
#            nd_thres = 1
#            thres = thres.reshape((thres.shape[0], thres.shape[1], nd_thres))
        # some thresholds for each time and element
        elif len(thres.shape) == 3:
            nd_thres = thres.shape[2]

    # Setting values
    if values == []:
        values = range(nd_thres+1)
    elif type(thres) in [np.ndarray, list]:
        assert(nd_thres == len(values)-1)

    # Creation of the limit min and max thresholds
    mini = np.ones((array.shape[0], array.shape[1], 1))*np.min(array)
    maxi = np.ones((array.shape[0], array.shape[1], 1))*np.max(array)
    # Concatenation
    thres = np.concatenate([mini, thres, maxi], axis=2)

    ## 2. Fill the new vector discretized signal to the given values
    aux = np.zeros(array.shape)
    for i in range(len(values)):
        indices = np.logical_and(array >= thres[:, :, i],
                                 array <= thres[:, :, i+1])
        indices = np.nonzero(indices)
        aux[indices] = values[i]

    if nd_input == 1:
        aux = aux.squeeze()

    return aux
