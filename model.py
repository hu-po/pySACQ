from collections import namedtuple
import random
import torch
import numpy as np

# Named tuple for a single step within a trajectory
Step = namedtuple('Step', ['state', 'action', 'log_prob', 'reward'])

# Global step counters
ACT_STEP = 0
LEARN_STEP = 0


def act(actor, env, task, B, num_trajectories=10, task_period=30, writer=None):
    """
    Performs actions in the environment collecting reward/experience.
    This follows Algorithm 3 in [1]
    :param actor: (Actor) actor network object
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
            actor.eval()
            action, log_prob = actor.predict(np.expand_dims(obs, axis=0), task.current_task, log_prob=True)
            # Execute action and collect rewards for each task
            obs, gym_reward, done, _ = env.step(action[0])
            # # Modify the main task reward (the huge -100 and 100 values cause instability)
            # gym_reward /= 100.0
            # Reward is a vector of the reward for each task
            reward = task.reward(obs, gym_reward)
            if writer:
                for i, r in enumerate(reward):
                    writer.add_scalar('train/reward/%s'%i, r, ACT_STEP)
            # group information into a step and add to current trajectory
            trajectory.append(Step(obs, action[0], log_prob[0], reward))
            num_steps += 1  # increment step counter
            ACT_STEP += 1
        # Add trajectory to replay buffer
        B.append(trajectory)


def _calculate_losses_simpler(trajectory, task, actor, critic, gamma=0.95):
    """
    Calculates actor and critic losses for a given trajectory. Simpler version
    :param trajectory: (list of Steps) trajectory or episode
    :param task: (Task) task object
    :param actor: (Actor) actor network object
    :param critic: (Critic) critic network object
    :param gamma: (float) discount factor
    :return: (float, float) actor and critic loss
    """
    # Extract information out of trajectory
    num_steps = len(trajectory)
    states = torch.FloatTensor([step.state for step in trajectory])
    rewards = torch.FloatTensor([step.reward for step in trajectory])
    actions = torch.FloatTensor([step.action for step in trajectory]).unsqueeze(1)
    log_probs = torch.FloatTensor([step.log_prob for step in trajectory]).unsqueeze(1)
    # Create an intention (task) mask for all possible intentions
    task_mask = np.repeat(np.arange(0, task.num_tasks), num_steps)
    imask_task = torch.LongTensor(task_mask)
    states = states.repeat(task.num_tasks, 1)
    actions = actions.repeat(task.num_tasks, 1)
    # actions (for each task) for every state action pair in trajectory
    task_actions, task_log_prob = actor.forward(states, imask_task, log_prob=True)
    # Q-values (for each task) for every state and task-action pair in trajectory
    critic_input = torch.cat([task_actions.data.float().unsqueeze(1), states], dim=1)
    task_q = critic.forward(critic_input, imask_task)
    task_q_t = task_q.cpu().data  # Available as tensor
    # Q-values (for each task) for every state and action pair in trajectory
    critic_input = torch.cat([actions, states], dim=1)
    traj_q = critic.predict(critic_input, imask_task)
    # Actor loss is log-prob weighted sum of Q values (for each task) given states from trajectory
    actor_loss = - torch.sum(torch.autograd.Variable(task_q.data, requires_grad=False).squeeze(1) * task_log_prob)
    actor_loss /= len(trajectory)  # Divide by the number of runs to prevent trajectory length from mattering
    # Calculation of retrace Q
    q_ret = torch.zeros_like(task_q.data)
    for task_id in range(task.num_tasks):
        start = task_id * num_steps
        for i in range(num_steps):
            q_ret_i = 0
            for j in range(i, num_steps):
                # Discount factor
                discount = gamma ** (j - i)
                # Difference between the two q values
                del_q = task_q_t[start + i] - traj_q[start + j]
                # Retrace Q value is sum of discounted weighted rewards
                q_ret_i += discount * (rewards[j, task_id] + del_q)
            # Append retrace Q value to float tensor using index_fill
            q_ret.index_fill_(0, torch.LongTensor([start + i]), q_ret_i[0])
    # Critic loss uses retrace Q
    # critic_loss = torch.sum((task_q - torch.autograd.Variable(q_ret, requires_grad=False)) ** 2)
    # critic_loss /= len(trajectory)  # Divide by the number of runs to prevent trajectory length from mattering
    # Use Huber Loss for critic
    critic_loss = torch.nn.SmoothL1Loss()(task_q, torch.autograd.Variable(q_ret, requires_grad=False))
    return actor_loss, critic_loss

def _calculate_losses(trajectory, task, actor, critic, gamma=0.95):
    """
    Calculates actor and critic losses for a given trajectory. Following equations in [1]
    :param trajectory: (list of Steps) trajectory or episode
    :param task: (Task) task object
    :param actor: (Actor) actor network object
    :param critic: (Critic) critic network object
    :param gamma: (float) discount factor
    :return: (float, float) actor and critic loss
    """
    # Extract information out of trajectory
    num_steps = len(trajectory)
    states = torch.FloatTensor([step.state for step in trajectory])
    rewards = torch.FloatTensor([step.reward for step in trajectory])
    actions = torch.FloatTensor([step.action for step in trajectory]).unsqueeze(1)
    log_probs = torch.FloatTensor([step.log_prob for step in trajectory]).unsqueeze(1)
    # Create an intention (task) mask for all possible intentions
    task_mask = np.repeat(np.arange(0, task.num_tasks), num_steps)
    imask_task = torch.LongTensor(task_mask)
    states = states.repeat(task.num_tasks, 1)
    actions = actions.repeat(task.num_tasks, 1)
    # actions (for each task) for every state action pair in trajectory
    task_actions, task_log_prob = actor.forward(states, imask_task, log_prob=True)
    # Q-values (for each task) for every state and task-action pair in trajectory
    critic_input = torch.cat([task_actions.data.float().unsqueeze(1), states], dim=1)
    task_q = critic.forward(critic_input, imask_task)
    # Q-values (for each task) for every state and action pair in trajectory
    critic_input = torch.cat([actions, states], dim=1)
    traj_q = critic.predict(critic_input, imask_task)
    # Actor loss is log-prob weighted sum of Q values (for each task) given states from trajectory
    actor_loss = - torch.sum(torch.autograd.Variable(task_q.data, requires_grad=False).squeeze(1) * task_log_prob)
    actor_loss /= len(trajectory)  # Divide by the number of runs to prevent trajectory length from mattering
    # Calculation of retrace Q
    q_ret = torch.zeros_like(task_q.data)
    for task_id in range(task.num_tasks):
        start = task_id * num_steps
        for i in range(num_steps):
            q_ret_i = 0
            for j in range(i, num_steps):
                # Discount factor
                discount = gamma ** (j - i)
                # Importance weights
                cj = 1.0
                for k in range(i, j):
                    ck = min(abs(task_log_prob.data[start + k] / float(log_probs[k])), 1.0)
                    cj *= ck
                # Difference between the two q values
                del_q = task_q.data[start + i] - traj_q[start + j]
                # Retrace Q value is sum of discounted weighted rewards
                q_ret_i += discount * cj * (rewards[j, task_id] + del_q)
            # Append retrace Q value to float tensor using index_fill
            q_ret.index_fill_(0, torch.LongTensor([start + i]), q_ret_i[0])
    # Critic loss uses retrace Q
    # critic_loss = torch.sum((task_q - torch.autograd.Variable(q_ret, requires_grad=False)) ** 2)
    # critic_loss /= len(trajectory)  # Divide by the number of runs to prevent trajectory length from mattering
    # Use Huber Loss for critic
    critic_loss = torch.nn.SmoothL1Loss()(task_q, torch.autograd.Variable(q_ret, requires_grad=False))
    return actor_loss, critic_loss


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
        # Put the nets in training mode
        actor.train()
        critic.train()
        for batch_idx in range(episode_batch_size):
            # Sample a random trajectory from the replay buffer
            trajectory = random.choice(B)
            # Compute losses for critic and actor
            # actor_loss = _actor_loss(actor, critic, task, trajectory)
            # critic_loss = _critic_loss(actor, critic, task, trajectory)
            actor_loss, critic_loss = _calculate_losses_simpler(trajectory, task, actor, critic)
            if writer:
                writer.add_scalar('train/loss/actor', actor_loss.data[0], LEARN_STEP)
                writer.add_scalar('train/loss/critic', critic_loss.data[0], LEARN_STEP)
            # compute gradients
            actor_loss.backward()
            critic_loss.backward()
            LEARN_STEP += 1
        # Push back the accumulated gradients and update the networks
        actor_opt.step()
        critic_opt.step()
