

import time
from NetSymulationTools.dynamics_evolution import *
from NetSymulationTools.net_formation import *

ts = []
step_list = [1, 10, 100, 250, 500, 750, 1000]
for e in step_list:
    print 'new attemp with t=' + str(e)
    t0 = time.time()
    n, m = 10, 10
    steps = e
    net2 = grid_2d_graph_2order(n, m)
    init = initialization_states(n*m)
    t1 = time.time()
    print time.time()-t0
    dyn = meta_evolution(net2, n_steps=steps, pre_states=init, t_mem=5)
    #dyn = evolution(net2, n_steps=steps, pre_states=init, t_mem=5)
    t2 = time.time()
    print t2-t1
    ts.append(t2-t1)

import matplotlib.pyplot as plt
plt.plot(step_list, ts)
plt.show()
