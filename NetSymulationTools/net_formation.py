
"""
Module oriented to give support for simulations creating graph structure of the
system. Create a networkx graph. It is used to wrap and complement the task of
obtain networkx graph with a desired structure.
"""

import numpy as np
import networkx as nx


######################################################
#### Network creation
#######################
def grid_2d_graph_2order(m, n, periodic=True, directed=True):
    '''This function creates a 2d grid with first and second neighbours of the
    dimensions of the grid specified.

    Parameters
    ----------
    m: int
        the horizontal dimension of the grid.
    n: int
        the vertical dimension of the grid.
    periodic: bool
        make the network self connected as a toroidal topology.
    directed: bool
        give a direct network.

    Returns
    -------
    G: networkx.DiGraph or networkx.Graph
        the grid as a network.

    Examples
    --------
    >>> net = grid_2d_graph_2order(100,100)
    >>> # test regularity
    >>> np.all([len(net.neighbors(node))==8 for node in net.nodes()])
    True

    '''

    G = nx.grid_2d_graph(m, n, periodic)
    edgs = G.edges()

    # Definition of the rotation matrix
    Rot = np.array([[1, 1], [-1, 1]])

    edgs2 = []  # [[] for i in range(len(edgs))]
    for i in range(len(edgs)):
        central_node = np.array(edgs[i][0])
        rodal_node = np.array(edgs[i][1])

        inc_vect = (rodal_node - central_node) % np.array([m, n])
        inc_vect = np.dot(Rot, inc_vect).astype(int)  # round better

        edgs2.append((tuple(central_node),
                      tuple((central_node + inc_vect) % np.array([m, n]))))
        edgs2.append((tuple(rodal_node),
                      tuple((rodal_node - inc_vect) % np.array([m, n]))))

    G.add_edges_from(edgs2)
    if directed:
        G = G.to_directed()
    return G
