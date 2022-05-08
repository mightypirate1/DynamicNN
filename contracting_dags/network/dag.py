import torch
import numpy as np

from network.nodes import Node, InputNode

class DAG(torch.nn.Module):
    def __init__(self, input_size, output_size, n_hidden):
        super().__init__()
        self.node_count = 0
        self.input_size, self.output_size = input_size, output_size
        self.input_nodes = [InputNode(id=i) for i in range(input_size)]
        nodes = []
        traversed = [*self.input_nodes]
        for i in range(n_hidden + output_size):
            hidden = i < n_hidden
            role = 'hidden' if hidden else 'output'
            node = Node(id=input_size+i, connections=traversed, role=role)
            traversed.append(node)
            nodes.append(node)
        self.nodes = torch.nn.ModuleList(nodes)
        self.next_id = input_size + output_size + n_hidden

    def find_node_with_only_one_edge(self):
        for node in self.nodes:
            print("contract?", node.id, len(node))
            if len(node) == 1:
                return node

    def contract_node_with_only_one_edge(self, target_node):
        if len(target_node) != 1:
            raise ValueError(f"expected node with 1 edge, but {target_node} has {len(target_node)} edges")

        print(f"CONTRACTING: {target_node.id}")
        node_parent = target_node.input_nodes[0]
        new_parameters = []
        for other in self.nodes:
            new_parameters.append(
                other.replace_input_node(target_node, node_parent)
            )
        new_parameters.append(
            self.drop_node(target_node)
        )

        if target_node.role == 'output':
            self.change_hidden_node_to_be_output_node(node_parent)

        return new_parameters

    def change_hidden_node_to_be_output_node(self, node):
        if node.role != 'hidden':
            raise ValueError(f"{node} is not a hidden node")
        node.role = 'output'


    def contract_one_node_with_only_one_edge(self):
        if (node := self.find_node_with_only_one_edge()):
            return self.contract_node_with_only_one_edge(node)


    def disable_weakest_node(self, threshold=0.1):
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
            new_parameters = self.drop_node(weakest_node)
        return new_parameters

    def disable_weakest_edge(self, threshold=0.1):
        weakest_val, weakest_node, weakest_link = np.inf, None, None
        for node in self.nodes:
            node_strength = torch.min(torch.abs(node.input_weights))
            if node_strength < weakest_val:
                weakest_val = node_strength
                weakest_node = node
                weakest_link = node.input_nodes[torch.argmin(node.input_weights)]

        new_parameters = []
        if weakest_val < threshold:
            new_parameters.append(
                weakest_node.disconnect_from(weakest_link)
            )
            if len(weakest_node) == 0:
                new_parameters.append(
                    self.drop_node(weakest_node)
                )
        return new_parameters

    def drop_node(self, node):
        new_parameters = []
        for n in self.all_nodes:
            new_parameters.append(n.disconnect_from(node))
        nodes = [n for n in self.nodes if n != node]
        self.nodes = torch.nn.ModuleList(nodes)
        return new_parameters

    @property
    def all_nodes(self):
        return [*self.input_nodes, *self.nodes]

    @property
    def output_nodes(self):
        return [node for node in self.nodes if node.role == 'output']

    @property
    def hidden_nodes(self):
        return [node for node in self.nodes if node.role == 'hidden']

    def __len__(self):
        return len(self.nodes)

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
