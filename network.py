import torch


class Actor(torch.nn.Module):
    """Class for policy (or actor) network"""

    def __init__(self, state_dim=8, hidden_size=16, action_dim=2,
                 action_gate=None,  # torch.nn.Softmax(),
                 action_scale=1.0,
                 non_linear=torch.nn.ReLU(),
                 batch_norm=False,
                 use_gpu=True):
        super(Actor, self).__init__()
        self.non_linear = non_linear
        self.batch_norm = batch_norm
        self.use_gpu = use_gpu
        self.action_gate = action_gate
        self.action_scale = action_scale

        # Build the actual network
        self.layer1 = torch.nn.Linear(state_dim, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, action_dim)
        if self.batch_norm:
            self.bn1 = torch.nn.BatchNorm1d(hidden_size)
            self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.init_weights()

    def init_weights(self):
        # Initialize with bounded uniform distribution for last layer (constant 0 bias)
        bound = 3e-3
        torch.nn.init.uniform(self.layer3.weight.data, -bound, bound)
        torch.nn.init.constant(self.layer3.bias.data, 0)
        # Initialize the other layers with xavier (still constant 0 bias)
        torch.nn.init.xavier_uniform(self.layer1.weight.data)
        torch.nn.init.constant(self.layer1.bias.data, 0)
        torch.nn.init.xavier_uniform(self.layer2.weight.data)
        torch.nn.init.constant(self.layer2.bias.data, 0)

    def forward(self, x):
        x = torch.autograd.Variable(torch.Tensor(x))
        x = self.non_linear(self.layer1(x))
        if self.batch_norm:
            x = self.bn1(x)
        x = self.non_linear(self.layer2(x))
        if self.batch_norm:
            x = self.bn2(x)
        x = self.layer3(x)
        # Use optional gate and scaling for action
        if self.action_gate:
            x = self.action_gate(x)
        x = self.action_scale * x
        return x


class Critic(torch.nn.Module):
    """Class for Q-function (or critic) network"""

    def __init__(self):
        pass
