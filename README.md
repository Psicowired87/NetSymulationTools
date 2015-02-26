# NetSymulationTools
Package which simulates the dynamics of a network system.

# Example
```python
from dynamics_evolution import meta_evolution
from net_formation import grid_2d_graph_2order
from aux_functions import initialization_states

net = grid_2d_graph_2order(n, m)
init = initialization_states(n*m)

dyn = meta_evolution(net, n_steps=steps, pre_states=init, t_mem=5)
```

# TODO
possibility to change network along the dynamics
general variables of the system
