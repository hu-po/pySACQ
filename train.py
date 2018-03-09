from pathlib import Path
import sys
from collections import namedtuple
import random
import time
import gym
import torch
import numpy as np

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


def act(num_trajectories=10, task_period=30):
    """
    Performs actions in the environment collecting reward/experience.
    This follows Algorithm 3 in [1]
    :param num_trajectories: (int) number of trajectories to collect at a time
    :param task_period: (int) number of steps in a single task period
    :return: None
    """

    for trajectory_idx in range(num_trajectories):
        print('Acting: trajectory %s of %s' % (trajectory_idx + 1, num_trajectories))
        # Reset environment and trajectory specific parameters
        trajectory = []  # collection of state, action, task pairs
        task.reset()  # h in paper
        obs = env.reset()
        done = False
        num_steps = 0

        # Roll out
        while not done:

            # Sample a new task using the scheduler
            if num_steps % task_period == 0:
                task.sample()

            # Get the action from current actor policy
            action = actor.predict(obs, task.current_task)

            # Execute action and collect rewards for each task
            new_obs, gym_reward, done, _ = env.step(action)

            # Reward is a vector of the reward for each task (with the main task reward appended)
            reward = task.reward(new_obs) + [gym_reward]

            # group information into a step and add to current trajectory
            new_step = Step(obs, action, reward, task.current_task, actor, critic)
            trajectory.append(new_step)

            num_steps += 1  # increment step counter

        # Add trajectory to replay buffer
        B.append(trajectory)


def learn(num_learning_iterations=10, lr=0.0002, erp=0.0001):
    """
    Pushes back gradients from the replay buffer, updating the actor and critic.
    This follows Algorithm 2 in [1]
    :param num_learning_iterations: (int) number of learning iterations per function call
    :param lr: (float) learning rate
    :param erp: (float) Entropy regularization parameter
    :return: None
    """
    for learn_idx in range(num_learning_iterations):
        print('Learning: trajectory %s of %s' % (learn_idx + 1, num_learning_iterations))

        # Sample a random trajectory from the replay buffer
        trajectory = random.choice(B)

        actor_loss = 0
        critic_loss = 0

        # optimizers for critic and actor
        actor_opt = torch.optim.Adam(actor.parameters(), lr)
        critic_opt = torch.optim.Adam(critic.parameters(), lr)
        actor_opt.zero_grad()
        critic_opt.zero_grad()

        for step in trajectory:
            # Input to critic is state and action combined
            critic_input = np.concatenate((step.state, step.action), axis=0)
            task_critic = step.critic.forward(critic_input, step.task_idx)
            task_actor = step.actor.forward(step.state, step.task_idx)
            # Actor loss follows Equation 9 in [1]
            actor_loss += task_critic + erp * torch.log(task_actor)
            # Critic loss
            critic_loss += 0

        # compute gradients
        actor_loss.backwards()
        critic_loss.backwards()

        # train networks
        actor_opt.step()
        critic_opt.step()


def run(min_rate=None):
    """
    Runs the actor policy on the environment, rendering it. This does not store anything
    and is only used for visualization.
    :param min_rate: (float) minimum framerate
    :return: None
    """
    obs = env.reset()
    done = False
    # Counter variables for number of steps and total episode time
    epoch_tic = time.clock()
    num_steps = 0
    while not done:
        step_tic = time.clock()
        env.render()
        # Use the previous observation to get an action from policy
        action = actor.predict(obs, -1)  # Last intention is main task
        # Step the environment and push outputs to policy
        obs, reward, done, _ = env.step(action)
        step_toc = time.clock()
        step_time = step_toc - step_tic
        if min_rate and step_time < min_rate:  # Sleep to ensure minimum rate
            time.sleep(min_rate - step_time)
        num_steps += 1
    # Total elapsed time in epoch
    epoch_toc = time.clock()
    epoch_time = epoch_toc - epoch_tic
    print('Episode complete (%s steps in %.2fsec), final reward %s ' % (num_steps,
                                                                        epoch_time,
                                                                        reward))


if __name__ == '__main__':
    # # Initialize Q table as a dataframe
    # q_table_columns = ['state', 'action', 'reward', 'task_id', 'policy']
    # Q = pd.DataFrame(columns=q_table_columns)

    # Replay buffer stores collected trajectories
    B = []

    # actor and critic policies are defined in networks.py
    actor = Actor()
    critic = Critic()

    # Environment is the lunar lander from OpenAI gym
    env = gym.make('LunarLanderContinuous-v2')

    # task scheduler is defined in tasks.py
    task = TaskScheduler()

    act()
    learn()
    run(min_rate=0.01)
