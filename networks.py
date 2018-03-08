import torch


class IntentionActor(torch.nn.Module):
    """Class for a single Intention head within the policy (or actor) network"""

    def __init__(self,
                 input_size=16,
                 hidden_size=8,
                 output_size=2,
                 non_linear=torch.nn.ELU(),
                 use_gpu=True):
        super(IntentionActor, self).__init__()
        self.non_linear = non_linear
        self.use_gpu = use_gpu

        # Build the network
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, output_size)

        # Initialize the weights
        self.init_weights()

    def init_weights(self):
        # Initialize the other layers with xavier (still constant 0 bias)
        torch.nn.init.xavier_uniform(self.layer1.weight.data)
        torch.nn.init.constant(self.layer1.bias.data, 0)
        torch.nn.init.xavier_uniform(self.layer2.weight.data)
        torch.nn.init.constant(self.layer2.bias.data, 0)

    def forward(self, x):
        x = self.non_linear(self.layer1(x))
        x = self.non_linear(self.layer2(x))
        # Feed through tanh and scale
        x = torch.nn.Tanh()(x)
        x = scale * x  # TODO: Figure out proper scaling from paper
        return x


class Actor(torch.nn.Module):
    """Class for policy (or actor) network"""

    def __init__(self,
                 state_dim=8,
                 base_hidden_size=32,
                 head_input_size=16,
                 num_intentions=1,
                 head_hidden_size=8,
                 head_output_size=2,
                 output_dim=2,
                 action_gate=torch.nn.Tanh,
                 action_scale=1.0,
                 non_linear=torch.nn.ELU(),
                 batch_norm=True,
                 use_gpu=True):
        super(Actor, self).__init__()
        self.action_gate = action_gate
        self.action_scale = action_scale
        self.non_linear = non_linear
        self.batch_norm = batch_norm
        self.use_gpu = use_gpu

        # Build the base of the network
        self.layer_1 = torch.nn.Linear(state_dim, base_hidden_size)
        self.layer_2 = torch.nn.Linear(base_hidden_size, head_input_size)
        if self.batch_norm:
            self.bn1 = torch.nn.BatchNorm1d(base_hidden_size)

        # Create the many intention nets heads
        self.intention_nets = []
        for _ in range(num_intentions):
            self.intention_nets.append(IntentionActor(input_size=head_input_size,
                                                      hidden_size=head_hidden_size,
                                                      output_size=head_output_size,
                                                      use_gpu=use_gpu))

        # Initialize the weights
        self.init_weights()

    def init_weights(self):
        # Initialize the other layers with xavier (still constant 0 bias)
        torch.nn.init.xavier_uniform(self.layer1.weight.data)
        torch.nn.init.constant(self.layer1.bias.data, 0)
        torch.nn.init.xavier_uniform(self.layer2.weight.data)
        torch.nn.init.constant(self.layer2.bias.data, 0)

    def forward(self, x, intention):
        x = torch.autograd.Variable(torch.Tensor(x))
        x = self.non_linear(self.layer1(x))
        if self.batch_norm:
            x = self.bn1(x)
        x = self.non_linear(self.layer2(x))
        x = self.intention_nets[intention].forward(x)

        # TODO: Selection component based on taskID
        # TODO: Output defines normal distribution

        # Use optional gate and scaling for action
        if self.action_gate:
            x = self.action_gate(x)
        x = self.action_scale * x
        return x


class IntentionCritic(torch.nn.Module):
    """Class for a single Intention head within the Q-function (or critic) network"""

    def __init__(self,
                 input_size=64,
                 hidden_size=32,
                 output_size=1,
                 non_linear=torch.nn.ELU(),
                 use_gpu=True):
        super(IntentionCritic, self).__init__()
        self.non_linear = non_linear
        self.use_gpu = use_gpu

        # Build the network
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, output_size)

        # Initialize the weights
        self.init_weights()

    def init_weights(self):
        # Initialize the other layers with xavier (still constant 0 bias)
        torch.nn.init.xavier_uniform(self.layer1.weight.data)
        torch.nn.init.constant(self.layer1.bias.data, 0)
        torch.nn.init.xavier_uniform(self.layer2.weight.data)
        torch.nn.init.constant(self.layer2.bias.data, 0)

    def forward(self, x):
        x = self.non_linear(self.layer1(x))
        x = self.non_linear(self.layer2(x))
        return x


class Critic(torch.nn.Module):
    """Class for Q-function (or critic) network"""

    def __init__(self,
                 state_dim=10,
                 base_hidden_size=64,
                 num_intentions=1,
                 head_input_size=64,
                 head_hidden_size=32,
                 head_output_size=1,
                 output_dim=2,
                 action_gate=torch.nn.Tanh,
                 action_scale=1.0,
                 non_linear=torch.nn.ELU(),
                 batch_norm=True,
                 use_gpu=True):
        super(Actor, self).__init__()
        self.action_gate = action_gate
        self.action_scale = action_scale
        self.non_linear = non_linear
        self.batch_norm = batch_norm
        self.use_gpu = use_gpu

        # Build the base of the network
        self.layer_1 = torch.nn.Linear(state_dim, base_hidden_size)
        self.layer_2 = torch.nn.Linear(base_hidden_size, head_input_size)
        if self.batch_norm:
            self.bn1 = torch.nn.BatchNorm1d(base_hidden_size)

        # Create the many intention nets heads
        self.intention_nets = []
        for _ in range(num_intentions):
            self.intention_nets.append(IntentionCritic(input_size=head_input_size,
                                                       hidden_size=head_hidden_size,
                                                       output_size=head_output_size,
                                                       use_gpu=use_gpu))

        # Initialize the weights
        self.init_weights()

    def init_weights(self):
        # Initialize the other layers with xavier (still constant 0 bias)
        torch.nn.init.xavier_uniform(self.layer1.weight.data)
        torch.nn.init.constant(self.layer1.bias.data, 0)
        torch.nn.init.xavier_uniform(self.layer2.weight.data)
        torch.nn.init.constant(self.layer2.bias.data, 0)

    def forward(self, x, intention):
        x = torch.autograd.Variable(torch.Tensor(x))
        x = self.non_linear(self.layer1(x))
        if self.batch_norm:
            x = self.bn1(x)
        x = self.non_linear(self.layer2(x))
        x = self.intention_nets[intention].forward(x)

        # TODO: Selection component based on taskID
        # TODO: Output defines normal distribution

        # Use optional gate and scaling for action
        if self.action_gate:
            x = self.action_gate(x)
        x = self.action_scale * x
        return x
