from pathlib import Path
import sys

# Add local files to path
ROOT_DIR = Path.cwd()
sys.path.append(str(ROOT_DIR))
from networks import Actor, Critic, IntentionActor, IntentionCritic
from rewards import rewards

# Pseudo-code follows Algorithm 3 in [1]
def actor(num_trajectories, steps_per_episode, task_period=10):

    # Initialize Q table

    M = 0 # Monte carlo simulated trajectories

    for trajectory_idx in num_trajectories:

        # get actor policy

        trajectory = []# collection of state, action, task pairs
        task_idx = 0 # h in paper

        # Go through each timestep in the trajectory
        for i in range(steps_per_episode):

            if  i % task_period == 0:
                # Switch task
                task = sample_task(tasks)
                task_idx = task_idx + 1

            # Get the action from current actor policy
            action = actor_policy(state, task)

            # Execute action and collect rewards for each task
            new_state = env.step(action)
            rewards = (state, new_state, tasks)

            # Append the new datapoint to the trajectory
            trajectory.append(state, action, reward, policy)

        # send trajectory and schedule decisions (tasks) to learner

        for i in enumerate(tasks):
            M = M + 1

            # Update q-table using monte carlo rewards


# Pseudo-code follows Algorithm 2 in [1]
def learner(num_learning_iterations, entropy_reg_param,
            initial_weights_actor, initial_weights_critic):


    for i in range(num_learning_iterations):

        # Update replay buffer with received trajectories

        for k in range(1000):

            trajectory = sample(replay_buffer)

            # compute gradients for actor and critic

            # train networks