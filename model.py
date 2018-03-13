from collections import namedtuple
import random
import torch
import numpy as np

# Named tuple for a single step within a trajectory
Step = namedtuple('Step', ['state', 'action', 'reward', 'task_id', 'log_prob'])

# Global step counters
ACT_STEP = 0
LEARN_STEP = 0


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


def act(actor, critic, env, task, B, num_trajectories=10, task_period=30, writer=None):
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
    :param writer: (SummaryWriter) writer object for logging
    :return: None
    """
    global ACT_STEP
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
            action, log_prob = actor.predict(obs, task.current_task, log_prob=True)
            # Execute action and collect rewards for each task
            new_obs, gym_reward, done, _ = env.step(np.asscalar(action))
            # # Clip the gym reward to be between -1 and 1 (the huge -100 and 100 values cause instability)
            # gym_reward = np.clip(-1.0, 1.0, gym_reward / 100.0)
            # Reward is a vector of the reward for each task (with the main task reward appended)
            reward = task.reward(new_obs) + [gym_reward]
            if writer:
                for i in range(task.num_tasks):
                    writer.add_scalar('reward_t%s' % i, reward[i], ACT_STEP)
            # group information into a step and add to current trajectory
            new_step = Step(obs, action, reward, task.current_task, log_prob)
            trajectory.append(new_step)
            num_steps += 1  # increment step counter
            ACT_STEP += 1
        # Add trajectory to replay buffer
        B.append(trajectory)


def _actor_loss(actor, critic, task, trajectory):
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
    states = torch.FloatTensor(np.array([step.state for step in trajectory]))
    for task_id in range(task.num_tasks):
        # Vector of actions this particular intention policy would have taken at each state in the trajectory
        # as well as the log probability of that action having been taken
        actions, log_prob = actor.forward(states, task_id, log_prob=True)
        # Combine action and state vectors to feed into critic
        critic_input = torch.cat([actions.data.float().unsqueeze(1), states], dim=1)
        # critic outputs the q value at each of the state action pairs
        q = critic.predict(critic_input, task_id, to_numpy=False)
        # TODO: Why does this tensor need to be converted to a Variable for .mul() to work?
        q = torch.autograd.Variable(q, requires_grad=False).squeeze(1)
        # Loss is the log probability of that particular action weighted by the q value
        actor_loss += - torch.sum(q * log_prob)
    # Divide by the number of runs to prevent trajectory length from mattering
    actor_loss /= len(trajectory)
    return actor_loss


def _critic_loss(actor, critic, task, trajectory, gamma=0.95):
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
    states = torch.FloatTensor(np.array([step.state for step in trajectory]))
    actions = torch.FloatTensor(np.array([step.action for step in trajectory]))
    rewards = torch.FloatTensor(np.array([step.reward for step in trajectory]))
    for task_id in range(task.num_tasks):
        # Get the task-specific q value for the task-specific action at every state action pair
        task_action = actor.predict(states, task_id, to_numpy=False)
        critic_input = torch.cat((task_action.float().unsqueeze(1), states), dim=1)
        qi = critic.predict(critic_input, task_id, to_numpy=False)
        # Get the task-specific q value for the trajectory action at every state action pair
        critic_input = torch.cat((actions, states), dim=1)
        q = critic.forward(critic_input, task_id)
        qj = q.data
        # Get the task-specific logprob values for the trajectory action at every state action pair
        _, task_log_prob = actor.forward(states, task_id, log_prob=True)
        # Calculation of retrace Q
        q_ret = torch.zeros(num_steps, 1)
        for i in range(num_steps):
            q_ret_i = 0
            for j in range(i, num_steps):
                # Discount factor
                discount = gamma ** (j - i)
                # Importance weights
                cj = 1.0
                for k in range(i, j):
                    ck = min((task_log_prob[k].data[0] / float(trajectory[k].log_prob)), 1.0)
                    cj *= ck
                # Difference between the two q values
                del_q = qi[i] - qj[j]
                # Retrace Q value is sum of discounted weighted rewards
                q_ret_i += discount * cj * (rewards[j, task_id] + del_q)
            # Append retrace Q value to float tensor using index_fill
            q_ret.index_fill_(0, torch.LongTensor([i]), q_ret_i[0])
        critic_loss += torch.sum((q - torch.autograd.Variable(q_ret, requires_grad=False)) ** 2)
    return critic_loss


def learn(actor, critic, task, B, num_learning_iterations=10, episode_batch_size=10, lr=0.0002, writer=None):
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
    :param writer: (SummaryWriter) writer object for logging
    :return: None
    """
    global LEARN_STEP
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
            actor_loss = _actor_loss(actor, critic, task, trajectory)
            critic_loss = _critic_loss(actor, critic, task, trajectory)
            if writer:
                writer.add_scalar('actor_loss', actor_loss, LEARN_STEP)
                writer.add_scalar('critic_loss', critic_loss, LEARN_STEP)
            # TODO: Make sure to average gradients based on number of steps (batch size) per intention
            # compute gradients
            actor_loss.backward()
            critic_loss.backward()
            LEARN_STEP += 1

        # Push back the accumulated gradients and update the networks
        actor_opt.step()
        # critic_opt.step()
