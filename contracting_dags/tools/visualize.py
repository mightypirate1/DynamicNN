import numpy as np
from utils.torch_utils import to_float, to_numpy

def draw_dag(dag):
    for node in [*dag.input_nodes, *dag.nodes]:
        print(f"{node.id} {node.role.ljust(7)} {node.sees}")

def draw_dag_weights(dag):
    np.set_printoptions(suppress=True)
    lines = []
    lengths = [0, 0, 0]
    for node in [*dag.input_nodes, *dag.nodes]:
        tag = f"{node.id} {node.role}"
        W = to_numpy(node.input_weights, decimals=3)[0]
        ids = [n.id for n in node.input_nodes]
        # print(ids)
        weights = f"{dict(zip(ids, W))}"
        # weights = f"{W}"
        bias = f"{to_float(node.bias, decimals=3)}"
        lines.append((tag, weights, bias))
    for i in range(len(lengths)):
        lengths[i] = max(*map(lambda x:len(x[i]), lines))
    for tag, weights, bias in lines:
        print(f"{tag.ljust(lengths[0])} {weights.ljust(lengths[1])} {bias.ljust(lengths[2])}")
