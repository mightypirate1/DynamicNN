import torch

class Node(torch.nn.Module):
    inputs = []
    weights = []
    def __init__(self, id, connections=None, time_created=None, role='hidden'):
        super().__init__()
        self.id             = id
        self.time_created   = time_created
        self.input_nodes    = []
        self._input_weights = None
        self.precomputed    = None
        self._disabled      = False
        self.role           = role

        bias = torch.ones(1)
        if connections is not None:
            self.connect([*connections])
            self.input_weights = torch.nn.Parameter(self._input_weights)
        self.bias = torch.nn.Parameter(bias)



    def connect(self, new_nodes):
        new_weights = torch.nn.Parameter(0.01 * torch.randn(1, len(new_nodes)))
        self.input_nodes.extend(new_nodes)
        if self._input_weights is None:
            self._input_weights = new_weights
        else:
            self._input_weights = torch.cat(self._input_weights, new_weights, axis=1)


    def compute_output(self):
        if self.precomputed is None:
            wx = 0
            if len(self) > 0:
                _x = [n.output for n in self.input_nodes]
                x = torch.cat(_x, axis=1)
                wx = torch.sum(self.input_weights * x, axis=1, keepdim=True)
            y = wx - self.bias
            if self.role == 'hidden':
                # y = torch.nn.Tanh()(y)
                y = torch.nn.Sigmoid()(y)
                # y = torch.maximum(torch.nn.Sigmoid()(y), torch.nn.ReLU()(y))
                # y = torch.nn.ReLU()(y)
                # y = torch.nn.ELU()(y)
            self.precomputed = y
        return self.precomputed


    def disconnect_from(self, other):
        print(f"node {self.id} disconnects from {other.id}")
        idx = [i for i, n in enumerate(self.input_nodes) if n.id == other.id]
        if len(idx) == 0:
            return
        if len(idx) > 1:
            raise Exception(f"expected exactly one occurance, got {len(idx)}")
        cut = idx[0]
        if cut == 0:
            new_weights = self.input_weights[:,1:]
        elif cut == len(self.input_nodes) - 1:
            new_weights = self.input_weights[:,:-1]
        else:
            part1 = self.input_weights[:,:cut]
            part2 = self.input_weights[:,cut+1:]
            new_weights = torch.cat(
                [part1, part2],
                dim=1,
            )
        nodes = [n for n in self.input_nodes if n != other]
        self.input_nodes = nodes #torch.nn.ModuleList(nodes)
        self.input_weights = torch.nn.Parameter(new_weights)
        return new_weights

    def replace_input_node(self, target_node, replacement_node):
        print(f"REPLACING: {target_node.id} -> {replacement_node.id}")
        if replacement_node in self.input_nodes:
            self.disconnect_from(target_node)
        else:
            nodes = [
                (replacement_node if node == target_node else node)
                for node in self.input_nodes
            ]
            self.input_nodes = nodes #torch.nn.ModuleList(nodes)


    def compute_connection_strength_to(self, other):
        mask = torch.Tensor([int(node == other) for node in self.input_nodes])
        return torch.sum(mask * torch.abs(self.input_weights))


    def ready_for_new_batch(self):
        self.precomputed = None


    def __eq__(self, other):
        return self.id == other.id


    def __len__(self):
        return len(self.input_nodes)


    def __hash__(self):
        return self.id


    @property
    def sees(self):
        return [i.id for i in self.input_nodes]


    @property
    def output(self):
        try:
            return self.compute_output()
        except Exception as e:
            print(self)
            print(self.input_nodes)
            print(self.input_weights)
            print(self.bias)
            raise e



class InputNode(Node):
    def __init__(self, id, time_created=None):
        super().__init__(id, time_created=time_created, role='input')
        self.input_weights = torch.nn.Parameter(torch.zeros(1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1))
    def compute_output(self):
        return self.x
    def __call__(self, x):
        self.x = x
