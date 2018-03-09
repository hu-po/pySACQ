from collections import namedtuple
import random
import torch
import numpy as np

# Named tuple for a single step within a trajectory
Step = namedtuple('Step', ['state', 'action', 'reward', 'task_id', 'actor', 'critic'])


def cumulative_discounted_reward(trajectory, task_id, task_period, gamma=0.95):
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
            gym_reward = np.clip(-1.0, 1.0, gym_reward)
            # Reward is a vector of the reward for each task (with the main task reward appended)
            reward = task.reward(new_obs) + [gym_reward]
            # group information into a step and add to current trajectory
            new_step = Step(obs, action, reward, task.current_task, actor, critic)
            trajectory.append(new_step)
            num_steps += 1  # increment step counter
        # Add trajectory to replay buffer
        B.append(trajectory)


def _compute_actor_loss(actor, critic, task, trajectory):
    """
    Computes the actor loss for a given trajectory. Uses the Q value from the critic.
    This follows equations 5 and 9 in [1]
    :param actor: (Actor) actor network object
    :param critic: (Critic) critic network object
    :param task: (Task) task object
    :param trajectory: (list of Steps) trajectory or episode
    :return: actor loss
    """
    actor_loss = 0
    # Convert trajectory states into a Tensor
    states = torch.autograd.Variable(torch.FloatTensor(np.array([step.state for step in trajectory])))
    for task_id in range(task.num_tasks):
        # Vector of actions this particular intention policy would have taken at each state in the trajectory
        # as well as the log probability of that action having been taken
        log_prob, actions = actor.forward(states, task_id, log_prob=True)
        # Combine action and state vectors to feed into critic
        critic_input = torch.cat((actions, states), dim=0)
        # critic outputs the q value at each of the state action pairs
        q = critic.predict(critic_input, task_id)
        # Weight the q value by the log probability of that particular action
        actor_loss += log_prob * q
    return actor_loss


def _compute_critic_loss(actor, critic, task, trajectory, gamma=0.95):
    """
    Computes the critic loss for a given trajectory. Based on the Retrace method.
    This follows equation 13 in [1]
    :param actor: (Actor) actor network object
    :param critic: (Critic) critic network object
    :param task: (Task) task object
    :param trajectory: (list of Steps) trajectory or episode
    :param gamma: (float) discount factor
    :return: critic loss
    """
    critic_loss = 0
    num_steps = len(trajectory)  # Number of steps in trajectory
    # Convert trajectory states into a Tensor
    states = torch.autograd.Variable(torch.FloatTensor(np.array([step.state for step in trajectory])))
    actions = torch.autograd.Variable(torch.FloatTensor(np.array([step.action for step in trajectory])))
    for task_id in range(task.num_tasks):

        q_ret = []

        for i in range(num_steps):

            q_ret_i = 0

            action = actor.predict(trajectory[i].state, task_id)
            critic_input = torch.cat((action, trajectory[i].state), dim=0)
            qi = critic.predict(critic_input, task_id)

            for j in range(i, num_steps):

                critic_input = torch.cat((trajectory[j].action, trajectory[j].state), dim=0)
                qj = critic.predict(critic_input, task_id)

                reward = trajectory[j][task_id]

                del_q = qi - qj

                discount = gamma ** (j - i)

                cj = 0
                for k in range(i, j):
                    log_prob1, _ = actor.forward(trajectory[k].state, task_id, log_prob=True)
                    log_prob2, _ = actor.forward(trajectory[k].state, trajectory[k].task_id, log_prob=True)
                    ck = min(1, (log_prob1 / log_prob2))
                    cj *= ck

                q_ret_i += discount * cj * (reward + del_q)

            q_ret[i] = q_ret_i

        critic_input = torch.cat((actions, states), dim=0)
        q = critic.forward(critic_input, task_id)

        critic_loss += (q - q_ret) ** 2

    return critic_loss


def learn(actor, critic, task, B, num_learning_iterations=10, episode_batch_size=10, lr=0.0002):
    """
    Pushes back gradients from the replay buffer, updating the actor and critic.
    This follows Algorithm 2 in [1]
    :param actor: (Actor) actor network object
    :param critic: (Critic) critic network object
    :param task: (Task) task object
    :param B: (list) replay buffer containing trajectories
    :param num_learning_iterations: (int) number of learning iterations per function call
    :param episode_batch_size: (int) number of trajectories in a batch (one gradient push)
    :param lr: (float) learning rate
    :return: None
    """
    for learn_idx in range(num_learning_iterations):
        print('Learning: trajectory %s of %s' % (learn_idx + 1, num_learning_iterations))
        # optimizers for critic and actor
        actor_opt = torch.optim.Adam(actor.parameters(), lr)
        critic_opt = torch.optim.Adam(critic.parameters(), lr)
        # Zero the gradients and set the losses to zero
        actor_opt.zero_grad()
        critic_opt.zero_grad()

        for batch_idx in range(episode_batch_size):
            # Sample a random trajectory from the replay buffer
            trajectory = random.choice(B)
            # Compute losses for critic and actor
            actor_loss = _compute_actor_loss(actor, critic, task, trajectory)
            critic_loss = _compute_critic_loss(actor, critic, task, trajectory)
            # TODO: Make sure to average gradients based on number of steps (batch size) per intention
            # compute gradients
            actor_loss.backwards()
            critic_loss.backwards()

        # Push back the accumulated gradients and update the networks
        actor_opt.step()
        critic_opt.step()
