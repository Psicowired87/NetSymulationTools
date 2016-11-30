
"""
This module groups the functions which performs a request of information useful
to compute the next states.

"""


def get_node_dynamic_info(node, pre_states, neighbours):
    """This function is an auxiliary function which performs the computation
    of the previous dynamics needed to compute the next state.

    Parameters
    ----------
    node: int
        the index of the given node.
    pre_states: array_like, shape (N, M)
        the previous states of the system.
    neighbours: list
        the indices of the neighbours.

    Returns
    -------
    pre_states_node: array_like, shape (N0,1)
        the previous states of the given node.
    pre_states_neig: array_like, shape (N0,M0)
        the previous states of the neigbours.

    """

    # previous states
    pre_states_neig = pre_states[:, neighbours]
    pre_states_node = pre_states[:, node]
    return pre_states_node, pre_states_neig


def get_node_list_structure_info(nodes, net, node_list,
                                 useweights=[False, False]):
    """This function is destined to compute network properties for a given
    list of nodes. Could be used each time of the computation of the new state
    or to compute the first time and parallelize the code.

    Parameters
    ----------
    nodes: list
        the list of the nodes we want to evaluate
    net: networkx.DiGraph
        the structure from which we want to extract the information.
    node_list:
        the complete list of nodes of the system in the correct order.
    useweights: list or tuple
        the boolean indicators of the use of weight variables.

    Returns
    -------
    nodes_index: list
        the index of each node in sequence with the nx order.
    neighs_index: list of list
        the list of the indices of the neighbours for each node.
    ws_e_neig: list of list
        the edge weight for each edge of each neighbours.
    ws_n_neig: list of list
        the weight of each edge of the node.
    w_nodes: list
        the weight of each node in the sequence.

    TODO:
    -----
    Only have a support for one type of property, weight
    Only could support weight.
    Default value if there is no weight is 1.

    """
    nodes_index, neighs_index = [], []
    ws_e_neig, ws_n_neig, w_nodes = [], [], []
    for i in range(len(nodes)):
        ## 1. Computation of nodes and neighbours
        node_index = node_list.index(nodes[i])
        neighbours = net.predecessors(nodes[i])
        neigh_index = [node_list.index(e) for e in neighbours]
        # appending
        nodes_index.append(node_index)
        neighs_index.append(neigh_index)

        ## 2. Computation of edges weights and node weights.
        if useweights[0]:
            # Extract from net
            we_ed_nei = [net.get_edge_data(neighbours[j], nodes[i])
                         for j in range(len(neighbours))]
            # Check if all have property and add default 1 for those without.
            for k in range(len(we_ed_nei)):
                if not we_ed_nei[k]:
                    we_ed_nei[k] = {'weight': 1}
            # Extract number of the weight
            we_ed_nei = [we_ed_nei[k]['weight'] for k in range(len(we_ed_nei))]
            # Append to list
            ws_e_neig.append(we_ed_nei)

        if useweights[1]:
            # Extract weight from net
            wei_node = net.node[nodes[i]]
            wei_neig = [net.node[neighbours[j]]
                        for j in range(len(neighbours))]
            # Set as default if they do not have property.
            if not wei_node:
                wei_node = {'weight': 1}
            for k in range(len(wei_neig)):
                if not wei_neig[k]:
                    wei_neig[k] = {'weight': 1}
            # Extract number of the weight
            wei_node = wei_node['weight']
            wei_neig = [wei_neig[k]['weight'] for k in range(len(wei_neig))]
            # Append to the list
            ws_n_neig.append(wei_neig)
            w_nodes.append(wei_node)

    return nodes_index, neighs_index, ws_e_neig, ws_n_neig, w_nodes


def get_node_dicts_structure_info(pre_states, nodes, net, node_list,
                                  useweights=[False, False]):
    """This function is destined to compute network properties for a given
    list of nodes. Could be used each time of the computation of the new state
    or to compute the first time and parallelize the code.

    Parameters
    ----------
    pre_states: array_like, shape (N,M)
        the previous states of the system.
    nodes: list
        the list of the nodes we want to evaluate
    net: networkx.DiGraph
        the structure from which we want to extract the information.
    node_list: list
        the complete list of nodes of the system in the correct order.
    useweights: list or tuple
        the boolean indicators of the use of weight variables.

    Returns
    -------
    lista: list
        contains a list of dictionaries of the variables needed.

    TODO:
    -----
    Only have a support for one type of property, weight
    Only could support weight.
    Default value if there is no weight is 1.

    """
    lista = []
    for i in range(len(nodes)):
        d = {}
        ## 1. Computation of nodes and neighbours
        node_index = node_list.index(nodes[i])
        neighbours = net.predecessors(nodes[i])
        neigh_index = [node_list.index(e) for e in neighbours]
        # Storing in the dictionary
        d['pre_states_node'] = pre_states[:, node_index]
        d['pre_states_neig'] = pre_states[:, neigh_index]

        ## 2. Computation of edges weights and node weights.
        if useweights[0]:
            # Extract from net
            we_ed_nei = [net.get_edge_data(neighbours[j], nodes[i])
                         for j in range(len(neighbours))]
            # Check if all have property and add default 1 for those without.
            for k in range(len(we_ed_nei)):
                if not we_ed_nei[k]:
                    we_ed_nei[k] = {'weight': 1}
            # Extract number of the weight
            we_ed_nei = [we_ed_nei[k]['weight'] for k in range(len(we_ed_nei))]
            # Storing in the dictionary
            d['we_ed_nei'] = we_ed_nei

        if useweights[1]:
            # Extract weight from net
            wei_node = net.node[nodes[i]]
            wei_neig = [net.node[neighbours[j]]
                        for j in range(len(neighbours))]
            # Set as default if they do not have property.
            if not wei_node:
                wei_node = {'weight': 1}
            for k in range(len(wei_neig)):
                if not wei_neig[k]:
                    wei_neig[k] = {'weight': 1}
            # Extract number of the weight
            wei_node = wei_node['weight']
            wei_neig = [wei_neig[k]['weight'] for k in range(len(wei_neig))]
            # Append to the list
            d['wei_neig'] = wei_neig
            d['wei_node'] = wei_node

        lista.append(d)

    return lista


def get_node_structure_info(node, net, node_list):
    """This function returns
    DEPRECATED
    ##TODO: weigths per node not only per edge
    """
    ## get neighbours
    neighbours = net.predecessors(node)
    neigh_index = [node_list.index(e) for e in neighbours]
    node_index = node_list.index(node)
    # get weights neighbours
    weights_neig = [net.get_edge_data(nei, node) for nei in neighbours]
    weights_neig = [1 if not weights_neig[i] else weights_neig[i]['weight']
                    for i in range(len(weights_neig))]
    return node_index, neigh_index


def from_net2dicts(net_info, pre_states):
    """This function transform the network information and the previous states
    into the information needed to run the update function.

    Parameters
    ----------
    net_info: tuple
        the information related with the net. It contains the variables
        `nodes_index`, `neighs_index`, `ws_e_neig`, `ws_n_neig`, `w_nodes`.
    pre_states: array_like, shape (N,M)
        the previous states of the system.

    Returns
    -------
    lista: list
        contains a list of dictionaries of the variables needed.

    """
    # Unpack variables
    nodes_index, neighs_index, ws_e_neig, ws_n_neig, w_nodes = net_info

    # Initialize needed variables
    useweights = [bool(ws_n_neig), bool(ws_n_neig)]
    lista = []

    # Loop for node
    for i in range(len(nodes_index)):
        # Initialize new dictionary
        d = {}
        # Input previous states
        d['pre_states_node'] = pre_states[:, nodes_index[i]]
        d['pre_states_neig'] = pre_states[:, neighs_index[i]]
        # Conditional input the weights
        if useweights[0]:
            d['wei_neig'] = ws_e_neig
        if useweights[1]:
            d['wei_neig'] = ws_n_neig
            d['wei_node'] = w_nodes

        # Append to the list
        lista.append(d)

    return lista
