from collections import namedtuple
import random
import torch
import numpy as np


# Named tuple for a single step within a trajectory
Step = namedtuple('Step', ['state', 'action', 'reward', 'task_idx', 'actor', 'critic'])


def cumulative_discounted_reward(trajectory, task_idx, task_period, gamma=0.95):
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


def act(actor, critic, env, task, B, num_trajectories=10, task_period=30):
    """
    Performs actions in the environment collecting reward/experience.
    This follows Algorithm 3 in [1]
    :param actor: (Actor) actor network object
    :param critic: (Critic) critic network object
    :param env: (Environment) OpenAI Gym Environment object
    :param task: (Task) task object
    :param B: (list) replay buffer containing trajectories
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

            # Clip the gym reward to be between -1 and 1 (the huge -100 and 100 values cause instability)
            np.clip(-1.0, 1.0, gym_reward, out=gym_reward)

            # Reward is a vector of the reward for each task (with the main task reward appended)
            reward = task.reward(new_obs) + [gym_reward]

            # group information into a step and add to current trajectory
            new_step = Step(obs, action, reward, task.current_task, actor, critic)
            trajectory.append(new_step)

            num_steps += 1  # increment step counter

        # Add trajectory to replay buffer
        B.append(trajectory)


def learn(actor, critic, B, num_learning_iterations=10, lr=0.0002, erp=0.0001):
    """
    Pushes back gradients from the replay buffer, updating the actor and critic.
    This follows Algorithm 2 in [1]
    :param actor: (Actor) actor network object
    :param critic: (Critic) critic network object
    :param B: (list) replay buffer containing trajectories
    :param num_learning_iterations: (int) number of learning iterations per function call
    :param lr: (float) learning rate
    :param erp: (float) Entropy regularization parameter
    :return: None
    """
    for learn_idx in range(num_learning_iterations):
        print('Learning: trajectory %s of %s' % (learn_idx + 1, num_learning_iterations))

        # Sample a random trajectory from the replay buffer
        trajectory = random.choice(B)

        # optimizers for critic and actor
        actor_opt = torch.optim.Adam(actor.parameters(), lr)
        critic_opt = torch.optim.Adam(critic.parameters(), lr)
        actor_loss = 0
        critic_loss = 0

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
        actor_opt.zero_grad()
        critic_opt.zero_grad()
        actor_loss.backwards()
        critic_loss.backwards()

        # train networks
        actor_opt.step()
        critic_opt.step()