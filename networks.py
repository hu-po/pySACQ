import torch


class IntentionNet(torch.nn.Module):
    """Generic class for a single intention head (used within actor/critic networks)"""

    def __init__(self, input_size, hidden_size, output_size, non_linear, use_gpu=True):
        super(IntentionNet, self).__init__()
        self.non_linear = non_linear
        self.use_gpu = use_gpu

        # Build the network
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.init_weights()

    def init_weights(self):
        # Initialize the other layers with xavier (still constant 0 bias)
        torch.nn.init.xavier_uniform(self.layer1.weight.data)
        torch.nn.init.constant(self.layer1.bias.data, 0)

    def forward(self, x):
        x = self.non_linear(self.layer1(x))
        return x


class IntentionCritic(IntentionNet):
    """Class for a single Intention head within the Q-function (or critic) network"""

    def __init__(self, input_size, hidden_size, output_size, non_linear, use_gpu=True):
        super(IntentionCritic, self).__init__(input_size, hidden_size, output_size, non_linear, use_gpu)

        # Critic specific final layer
        self.layer2 = torch.nn.Linear(hidden_size, output_size)
        torch.nn.init.xavier_uniform(self.layer2.weight.data)
        torch.nn.init.constant(self.layer2.bias.data, 0)

    def forward(self, x):
        x = super().forward(x)
        x = self.non_linear(self.layer2(x))
        return x


class IntentionActor(IntentionNet):
    """Class for a single Intention head within the policy (or actor) network"""

    def __init__(self, input_size, hidden_size, output_size, non_linear, use_gpu=True):
        super(IntentionActor, self).__init__(input_size, hidden_size, output_size, non_linear, use_gpu)

        # Actor specific final layers (separate layers for mean and std)
        self.mean_activation_func = torch.nn.Tanh()
        self.mean_layer = torch.nn.Linear(hidden_size, output_size)
        self.std_layer = torch.nn.Linear(hidden_size, output_size)
        torch.nn.init.xavier_uniform(self.mean_layer.weight.data)
        torch.nn.init.constant(self.mean_layer.bias.data, 0)
        torch.nn.init.xavier_uniform(self.std_layer.weight.data)
        torch.nn.init.constant(self.std_layer.bias.data, 0)

    def forward(self, x):
        x = super().forward(x)
        # TODO: This isn't exactly what is in paper, check this
        mean = self.mean_activation_func(self.mean_layer(x))
        std = self.non_linear(self.std_layer(x))
        return mean, std


class BaseNet(torch.nn.Module):

    def __init__(self, state_dim, base_hidden_size, head_input_size, non_linear, batch_norm=True, use_gpu=True):
        super(BaseNet, self).__init__()
        self.non_linear = non_linear
        self.batch_norm = batch_norm
        self.use_gpu = use_gpu

        # Build the base of the network
        self.layer1 = torch.nn.Linear(state_dim, base_hidden_size)
        self.layer2 = torch.nn.Linear(base_hidden_size, head_input_size)
        if self.batch_norm:
            self.bn1 = torch.nn.BatchNorm1d(base_hidden_size)
        self.init_weights()

    def init_weights(self):
        # Initialize the other layers with xavier (still constant 0 bias)
        torch.nn.init.xavier_uniform(self.layer1.weight.data)
        torch.nn.init.constant(self.layer1.bias.data, 0)
        torch.nn.init.xavier_uniform(self.layer2.weight.data)
        torch.nn.init.constant(self.layer2.bias.data, 0)

    def forward_base(self, x):
        x = torch.autograd.Variable(torch.Tensor(x))
        x = self.non_linear(self.layer1(x))
        if self.batch_norm:
            x = self.bn1(x)
        x = self.non_linear(self.layer2(x))
        return x

    def predict(self, x, intention, to_numpy=True):
        y = self.forward(x, intention)
        if to_numpy:
            y = y.cpu().data.numpy()
        return y


class Actor(BaseNet):
    """Class for policy (or actor) network"""

    def __init__(self,
                 num_intentions=6,
                 state_dim=8,
                 base_hidden_size=32,
                 head_input_size=16,
                 head_hidden_size=8,
                 head_output_size=2,
                 non_linear=torch.nn.ELU(),
                 batch_norm=False,
                 use_gpu=True):
        super(Actor, self).__init__(state_dim, base_hidden_size, head_input_size, non_linear, batch_norm, use_gpu)

        # Create the many intention nets heads
        self.intention_nets = []
        for _ in range(num_intentions):
            self.intention_nets.append(IntentionActor(input_size=head_input_size,
                                                      hidden_size=head_hidden_size,
                                                      output_size=head_output_size,
                                                      use_gpu=use_gpu,
                                                      non_linear=non_linear))

        # Initialize the weights
        self.init_weights()

    def forward(self, x, intention):
        x = super().forward_base(x)
        # Feed forward through the relevant intention head
        mean, std = self.intention_nets[intention].forward(x)
        # Intention head determines parameters of Normal distribution
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action


class Critic(BaseNet):
    """Class for Q-function (or critic) network"""

    def __init__(self,
                 num_intentions=6,
                 state_dim=10,
                 base_hidden_size=64,
                 head_input_size=64,
                 head_hidden_size=32,
                 head_output_size=1,
                 non_linear=torch.nn.ELU(),
                 batch_norm=False,
                 use_gpu=True):
        super(Critic, self).__init__(state_dim, base_hidden_size, head_input_size, non_linear, batch_norm, use_gpu)

        # Create the many intention nets heads
        self.intention_nets = []
        for _ in range(num_intentions):
            self.intention_nets.append(IntentionNet(input_size=head_input_size,
                                                    hidden_size=head_hidden_size,
                                                    output_size=head_output_size,
                                                    use_gpu=use_gpu,
                                                    non_linear=non_linear))

        # Initialize the weights
        self.init_weights()

    def forward(self, x, intention):
        x = super().forward_base(x)
        # Feed forward through the relevant intention head
        x = self.intention_nets[intention].forward(x)
        return x


if __name__ == '__main__':
    print('Run this file directly to debug')

    actor = Actor()
    critic = Critic()

    # Carry out a step on the environment to test out forward functions
    import gym
    import random

    env = gym.make('LunarLanderContinuous-v2')
    obs = env.reset()
    task_idx = random.randint(0, 5)

    # Get the action from current actor policy
    action = actor.predict(obs, task_idx)
    _, _, _, _ = env.step(action)

    print('Got to end sucessfully! (Though this only means there are no major bugs..)')
