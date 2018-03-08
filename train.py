from pathlib import Path
import sys
from collections import namedtuple
import random
import gym
import torch
import numpy as np
import pandas as pd

# Add local files to path
ROOT_DIR = Path.cwd()
sys.path.append(str(ROOT_DIR))
from networks import Actor, Critic
from tasks import TaskScheduler

# Named tuple for a single step within a trajectory
Step = namedtuple('Step', ['state', 'action', 'reward', 'task_idx', 'actor', 'critic'])


def cumulative_main_task_reward(trajectory, task_idx, task_period, gamma=0.95):
    """
    Follows Equation 6 in [1]
    :return:
    """
    total_reward = 0
    # first summation in paper
    h_bounds_from = 0
    h_bounds_to = 2
    for h in range(h_bounds_from, h_bounds_to):
        # second summation in paper
        time_bounds_from = h * task_period
        time_bounds_to = (h + 1) * task_period
        for t in range(time_bounds_from, time_bounds_to):
            # The reward for the main task is at the end of the rewards vector
            main_task_reward = trajectory[t].reward[-1]
            total_reward += (gamma ** t) * main_task_reward
    return total_reward

def actor_loss(trajectory, actor, critic, entropy_reg_param):
    """
    Follows Equation 9 in [1]
    :return:
    """
    loss = 0
    for step in trajectory:
        task_q = critic.forward(step.state, step.task_idx)
        task_policy = actor.forward(step.state, step.task_idx)
        loss += task_q + entropy_reg_param * torch.log(task_policy)
    return loss


def act(num_trajectories=100, task_period=30):
    """
    Performs actions in the environment, populating the Q-table and collecting reward/experience.
    This follows Algorithm 3 in [1]
    :param num_trajectories: (int) number of trajectories to collect at a time
    :param task_period: (int) number of steps in a single task period
    :return:
    """

    for trajectory_idx in range(num_trajectories):

        # Reset environment and trajectory specific parameters
        trajectory = []  # collection of state, action, task pairs
        task_idx = 0  # h in paper
        obs = env.reset()
        done = False
        num_steps = 0

        # Roll out
        while not done:

            if num_steps % task_period == 0:
                task.sample()  # Sample a new task using the scheduler
                task_idx = task_idx + 1

            # Get the action from current actor policy
            action = actor.forward(obs, task_idx)

            # Execute action and collect rewards for each task
            new_obs, gym_reward, done, _ = env.step(action)

            # reward is a vector of the reward for each task
            reward = task.reward(new_obs, gym_reward)

            # group information into a step and add to current trajectory
            new_step = Step(obs, action, reward, task_idx, actor, critic)
            trajectory.append(new_step)

            num_steps += 1  # increment step counter

        # send trajectory and schedule decisions (tasks) to learner

        for _ in range(task.num_tasks):
            M = M + 1  # increment monte carlo counter

            # Calculate cumulative discounted rewards

            # Update q-table using monte carlo rewards


# Pseudo-code follows Algorithm 2 in [1]
def learn(num_learning_iterations,
          learning_rate=0.001):

    for i in range(num_learning_iterations):

        for k in range(1000):
            # Sample a random trajectory from the replay buffer
            trajectory = random.choice(B)

            for task in trajectory_tasks:

                # Use actor and critic from that specific trajectory

                # optimizers for critic and actor
                actor_opt = torch.optim.RMSprop(actor.parameters(), learning_rate)
                critic_opt = torch.optim.RMSprop(critic.parameters(), learning_rate)
                actor_opt.zero_grad()
                critic_opt.zero_grad()

                # compute gradient for critic

                # compute gradient for actor

                # train networks
                actor_opt.step()
                critic_opt.step()

if __name__ == 'main()':

    # Initialize Q table as a dataframe
    q_table_columns = ['state', 'action', 'reward', 'task_id', 'policy']
    Q = pd.DataFrame(columns=q_table_columns)

    # Monte carlo counter for simulated trajectories
    M = 0

    # Replay buffer stores collected trajectories
    B = []

    # actor and critic policies are defined in networks.py
    actor = Actor(state_dim=8,
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
                  use_gpu=True)
    critic = Critic(state_dim=10,
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
                    use_gpu=True)

    # Environment is the lunar lander from OpenAI gym
    env = gym.make('LunarLanderContinuous-v2')

    # task scheduler is defined in tasks.py
    task = TaskScheduler()
