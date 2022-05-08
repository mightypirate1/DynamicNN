# from graph_tool.all import *

def to_float(x):
    return float(x.detach().numpy())

def draw_dag(dag):
    for node in [*dag.input_nodes, *dag.nodes]:
        print(f"{node.id} {node.role.ljust(7)} {node.sees}")

def draw_dag_weights(dag):
    for node in [*dag.input_nodes, *dag.nodes]:
        node_weights = node.input_weights.detach().numpy().round(decimals=3)
        node_bias = to_float(node.bias)
        print(f"{node.id} {node.role.ljust(7)} {node_weights}".ljust(80), node_bias)
