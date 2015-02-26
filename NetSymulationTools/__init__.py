

__author__ = 'To\xc3\xb1o G. Quintela (tgq.spm@gmail.com)'
__version__ = '0.0.0'

"""
Module useful to simulate dynamics in a network system.

TODO
----
possibility to change network along the dynamics
general variables of the system


Variables structured
---------------------
## Structure information
# net
# node_list
## Dynamic information
# n_steps        -------------> 1
# pre_states     ------------> 1
## Simulation information
# syncronicity   -------------> 2
# parallelize    -------------> 2
# mem_use        -------------> 2 (how to compute net information)
## Model information
# useweights     -------------> 3 (linked with method)
# method         -------------> 3 (linked with useweights)
# is_static      -------------> multilabel
# w_temp         -------------> 3
# t_mem          -------------> 3

"""

