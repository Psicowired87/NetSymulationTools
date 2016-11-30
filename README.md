# NetTools
Package which groups different tools for network study. It is composed by network structure and network evolution.

# Version
__version__ = '0.0.0'


# Example
```python
from NetTools.NetSymulationTools.dynamics_evolution import meta_evolution
from NetTools.NetSymulationTools.net_formation import grid_2d_graph_2order
from NetTools.NetSymulationTools.aux_functions import initialization_states

net = grid_2d_graph_2order(n, m)
init = initialization_states(n*m)

dyn = meta_evolution(net, n_steps=steps, pre_states=init, t_mem=5)
```

# TODO
* possibility to change network along the dynamics
* general variables of the system
