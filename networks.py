import torch
import numpy as np


class IntentionBase(torch.nn.Module):
    """Generic class for a single intention head (used within actor/critic networks)"""

    def __init__(self, input_size, hidden_size, output_size, non_linear, final_non_linear, use_gpu=True):
        super(IntentionBase, self).__init__()
        self.non_linear = non_linear
        self.final_non_linear = final_non_linear
        self.use_gpu = use_gpu

        # Build the network
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.final_layer = torch.nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        # Initialize the other layers with xavier (still constant 0 bias)
        torch.nn.init.xavier_uniform(self.layer1.weight.data)
        torch.nn.init.constant(self.layer1.bias.data, 0)
        torch.nn.init.xavier_uniform(self.final_layer.weight.data)
        torch.nn.init.constant(self.final_layer.bias.data, 0)

    def forward(self, x):
        x = self.non_linear(self.layer1(x))
        x = self.final_non_linear(self.final_layer(x))
        return x


class IntentionCritic(IntentionBase):
    """Class for a single Intention head within the Q-function (or critic) network"""

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 non_linear=torch.nn.ELU(),
                 final_non_linear=torch.nn.ELU(),
                 use_gpu=True):
        super(IntentionCritic, self).__init__(input_size, hidden_size, output_size, non_linear, final_non_linear,
                                              use_gpu)


class IntentionActor(IntentionBase):
    """Class for a single Intention head within the policy (or actor) network"""

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 non_linear=torch.nn.ELU(),
                 final_non_linear=torch.nn.Softmax(),
                 use_gpu=True):
        super(IntentionActor, self).__init__(input_size, hidden_size, output_size, non_linear, final_non_linear,
                                             use_gpu)


class SQXNet(torch.nn.Module):
    """Generic class for actor and critic networks. The arch is very similar."""

    def __init__(self,
                 state_dim,
                 base_hidden_size,
                 num_intentions,
                 head_input_size,
                 head_hidden_size,
                 head_output_size,
                 non_linear,
                 net_type,
                 batch_norm=False,
                 use_gpu=True):
        super(SQXNet, self).__init__()
        self.non_linear = non_linear
        self.batch_norm = batch_norm
        self.use_gpu = use_gpu

        # Build the base of the network
        self.layer1 = torch.nn.Linear(state_dim, base_hidden_size)
        self.layer2 = torch.nn.Linear(base_hidden_size, head_input_size)
        if self.batch_norm:
            self.bn1 = torch.nn.BatchNorm1d(base_hidden_size)
        self.init_weights()

        # Create the many intention nets heads
        if net_type == 'actor':
            intention_net_type = IntentionActor
        elif net_type == 'critic':
            intention_net_type = IntentionCritic
        else:
            raise Exception('Invalid net type for SQXNet')
        self.intention_nets = []
        for _ in range(num_intentions):
            self.intention_nets.append(intention_net_type(input_size=head_input_size,
                                                          hidden_size=head_hidden_size,
                                                          output_size=head_output_size,
                                                          use_gpu=use_gpu,
                                                          non_linear=non_linear))

    def init_weights(self):
        # Initialize the other layers with xavier (still constant 0 bias)
        torch.nn.init.xavier_uniform(self.layer1.weight.data)
        torch.nn.init.constant(self.layer1.bias.data, 0)
        torch.nn.init.xavier_uniform(self.layer2.weight.data)
        torch.nn.init.constant(self.layer2.bias.data, 0)

    def forward(self, x, intention):
        # Feed the input through the base layers of the model
        x = torch.autograd.Variable(torch.Tensor(x))
        x = self.non_linear(self.layer1(x))
        if self.batch_norm:
            x = self.bn1(x)
        x = self.non_linear(self.layer2(x))
        if isinstance(intention, int):  # single intention head
            x = self.intention_nets[intention].forward(x)
        else:
            # Create intention mask
            one_hot_mask = np.zeros((x.shape[0], len(self.intention_nets)))
            one_hot_mask[np.arange(x.shape[0]), intention.numpy()] = 1
            mask_tensor = torch.autograd.Variable(torch.FloatTensor(one_hot_mask).unsqueeze(1), requires_grad=False)
            # Feed forward through all the intention heads and concatenate on new dimension
            intention_out = torch.cat(list(head.forward(x).unsqueeze(2) for head in self.intention_nets), dim=2)
            # Multiply by the intention mask and sum in the final dimension to get the right output shape
            x = (intention_out * mask_tensor).sum(dim=2)
        return x

    def predict(self, x, intention):
        y = self.forward(x, intention).cpu().data
        return y


class Actor(SQXNet):
    """Class for policy (or actor) network"""

    def __init__(self,
                 state_dim=8,
                 base_hidden_size=32,
                 num_intentions=6,
                 head_input_size=16,
                 head_hidden_size=8,
                 head_output_size=4,
                 non_linear=torch.nn.ELU(),
                 net_type='actor',
                 batch_norm=False,
                 use_gpu=True):
        super(Actor, self).__init__(state_dim,
                                    base_hidden_size,
                                    num_intentions,
                                    head_input_size,
                                    head_hidden_size,
                                    head_output_size,
                                    non_linear,
                                    net_type,
                                    batch_norm,
                                    use_gpu)

    def forward(self, x, intention, log_prob=False):
        x = super().forward(x, intention)
        # Intention head determines parameters of Categorical distribution
        dist = torch.distributions.Categorical(x)
        action = dist.sample()
        if log_prob:
            log_prob = dist.log_prob(action)
            return action, log_prob
        return action

    def predict(self, x, intention, log_prob=False):
        if log_prob:
            action, log_prob = self.forward(x, intention, log_prob=True)
            return action.cpu().data, log_prob.cpu().data
        else:
            action = self.forward(x, intention).cpu().data
            return action
        return None


class Critic(SQXNet):
    """Class for Q-function (or critic) network"""

    def __init__(self,
                 num_intentions=6,
                 state_dim=9,
                 base_hidden_size=64,
                 head_input_size=64,
                 head_hidden_size=32,
                 head_output_size=1,
                 non_linear=torch.nn.ELU(),
                 net_type='critic',
                 batch_norm=False,
                 use_gpu=True):
        super(Critic, self).__init__(state_dim,
                                     base_hidden_size,
                                     num_intentions,
                                     head_input_size,
                                     head_hidden_size,
                                     head_output_size,
                                     non_linear,
                                     net_type,
                                     batch_norm,
                                     use_gpu)


if __name__ == '__main__':
    print('Run this file directly to debug')

    actor = Actor()
    critic = Critic()

    # Carry out a step on the environment to test out forward functions
    import gym
    import random

    env = gym.make('LunarLander-v2')
    obs = env.reset()
    task_idx = random.randint(0, 6)

    # Get the action from current actor policy
    action = actor.predict(obs, task_idx)
    _, _, _, _ = env.step(action)

    print('Got to end sucessfully! (Though this only means there are no major bugs..)')
