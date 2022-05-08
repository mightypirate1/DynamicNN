import torch
import numpy as np

from network.nodes import Node, InputNode

class DAG(torch.nn.Module):
    def __init__(self, input_size, output_size, n_hidden):
        super().__init__()
        self.node_count = 0
        self.input_size, self.output_size = input_size, output_size
        self.input_nodes = [InputNode(id=i) for i in range(input_size)]
        self.hidden_nodes = []
        self._output_nodes = []
        nodes = []
        traversed = [*self.input_nodes]
        for i in range(n_hidden + output_size):
            hidden = i < n_hidden
            role = 'hidden' if hidden else 'output'
            bucket = self.hidden_nodes if hidden else self._output_nodes
            node = Node(id=input_size+i, connections=traversed, role=role)
            traversed.append(node)
            bucket.append(node)
        nodes = [*self.hidden_nodes, *self._output_nodes]
        self.nodes = torch.nn.ModuleList(nodes)
        self.next_id = input_size + output_size + n_hidden

    def disable_weakest_node(self, threshold=0.3):
        weakest_val, weakest_node = np.inf, None
        for node in self.hidden_nodes:
            node_strength = 0
            for other in self.nodes:
                node_strength += other.compute_connection_strength_to(node)
            if node_strength < weakest_val:
                weakest_val  = node_strength
                weakest_node = node
        new_parameters = []
        if weakest_val < threshold:
            for node in self.all_nodes:
                new_parameters.append(node.disconnect_from(weakest_node))
            self.hidden_nodes = [n for n in self.hidden_nodes if n.id != weakest_node.id]
            nodes = [n for n in self.nodes if n.id != weakest_node.id]
            self.nodes = torch.nn.ModuleList(nodes)
        return new_parameters

    @property
    def all_nodes(self):
        return [*self.input_nodes, *self.nodes]

    @property
    def output_nodes(self):
        return [node for node in self.nodes if node.role == 'output']

    def ready_for_new_batch(self):
        for node in self.nodes:
            node.ready_for_new_batch()

    def forward(self, x, training=False):
        self.ready_for_new_batch()
        for i in range(x.shape[1]):
            self.input_nodes[i](x[:,i:i+1])
        y = torch.cat([o.output for o in self.output_nodes], axis=1)
        if not training:
            y = torch.nn.Sigmoid()(y)
        return y
