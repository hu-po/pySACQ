from pathlib import Path
import sys
import time
import gym

# Add local files to path
ROOT_DIR = Path.cwd()
sys.path.append(str(ROOT_DIR))
from networks import Actor, Critic
from tasks import TaskScheduler
from model import act, learn


def run(actor, env, min_rate=None):
    """
    Runs the actor policy on the environment, rendering it. This does not store anything
    and is only used for visualization.
    :param actor: (Actor) actor network object
    :param env: (Environment) OpenAI Gym Environment object
    :param min_rate: (float) minimum framerate
    :return: None
    """
    obs = env.reset()
    done = False
    # Counter variables for number of steps and total episode time
    epoch_tic = time.clock()
    num_steps = 0
    reward = 0
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
    print('Episode complete (%s steps in %.2fsec), final reward %s ' % (num_steps, epoch_time, reward))


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

    act(actor, critic, env, task, B, num_trajectories=10, task_period=30)
    learn(actor, critic, task, B, num_learning_iterations=10, lr=0.0002, erp=0.0001)
    run(min_rate=0.01)
