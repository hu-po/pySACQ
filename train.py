import argparse
from pathlib import Path
import sys
import time
import gym
import torch
from collections import deque
from tensorboardX import SummaryWriter

# Add local files to path
root_dir = Path.cwd()
sys.path.append(str(root_dir))
from networks import Actor, Critic
from tasks import TaskScheduler
from model import act, learn

parser = argparse.ArgumentParser(description='Train Arguments')
parser.add_argument('--log', type=str, default=None, help='Write tensorboard style logs to this folder [default: None]')
parser.add_argument('--num_train_cycles', type=int, default=1000, help='Number of training cycles [default: 1]')
parser.add_argument('--saveas', type=str, default=None, help='savename for trained model [default: None]')
parser.add_argument('--buffer_size', type=int, default=200, help='Number of trajectories in replay buffer [default: 200]')


parser.add_argument('--num_trajectories', type=int, default=5,
                    help='Number of trajectories collected per acting cycle [default: 5]')
parser.add_argument('--num_learning_iterations', type=int, default=1,
                    help='Number of learning iterations per learn cycle [default: 1]')
parser.add_argument('--episode_batch_size', type=int, default=2,
                    help='Number of trajectories per batch (gradient push) [default: 2]')

# Global step counters
TEST_STEP = 0


def run(actor, env, min_rate=None, writer=None, render=False):
    """
    Runs the actor policy on the environment, rendering it. This does not store anything
    and is only used for visualization.
    :param actor: (Actor) actor network object
    :param env: (Environment) OpenAI Gym Environment object
    :param min_rate: (float) minimum framerate
    :param writer: (SummaryWriter) writer object for logging
    :param render: (Bool) toggle for rendering to window
    :return: None
    """
    global TEST_STEP
    obs = env.reset()
    done = False
    # Counter variables for number of steps and total episode time
    epoch_tic = time.clock()
    num_steps = 0
    reward = 0
    while not done:
        step_tic = time.clock()
        if render:
            env.render()
        # Use the previous observation to get an action from policy
        action = actor.predict(obs, -1)  # Last intention is main task
        # Step the environment and push outputs to policy
        obs, reward, done, _ = env.step(action[0])
        if writer:
            writer.add_scalar('test/reward', reward, TEST_STEP)
        step_toc = time.clock()
        step_time = step_toc - step_tic
        if min_rate and step_time < min_rate:  # Sleep to ensure minimum rate
            time.sleep(min_rate - step_time)
        num_steps += 1
        TEST_STEP += 1
    # Total elapsed time in epoch
    epoch_toc = time.clock()
    epoch_time = epoch_toc - epoch_tic
    print('Episode complete (%s steps in %.2fsec), final reward %s ' % (num_steps, epoch_time, reward))


if __name__ == '__main__':

    # Parse and print out parameters
    args = parser.parse_args()
    print('Running Trainer. Parameters:')
    for attr, value in args.__dict__.items():
        print('%s : %s' % (attr.upper(), value))

    # Make sure we can use gpu
    use_gpu = torch.cuda.is_available()
    print('Gpu is enabled: %s' % use_gpu)

    # Replay buffer stores collected trajectories
    B = []

    # actor and critic policies are defined in networks.py
    actor = Actor(use_gpu=use_gpu)
    critic = Critic(use_gpu=use_gpu)

    # Environment is the lunar lander from OpenAI gym
    env = gym.make('LunarLander-v2')

    # task scheduler is defined in tasks.py
    task = TaskScheduler()

    # Write tensorboard logs to local logs folder
    writer = None
    if args.log:
        log_dir = root_dir / 'logs' / args.log
        writer = SummaryWriter(log_dir=str(log_dir))

    for i in range(args.num_train_cycles):
        print('Training cycle %s of %s' % (i, args.num_train_cycles))
        act(actor, env, task, B,
            num_trajectories=args.num_trajectories,
            task_period=30, writer=writer)
        learn(actor, critic, task, B,
              num_learning_iterations=args.num_learning_iterations,
              episode_batch_size=args.episode_batch_size,
              lr=0.0002, writer=writer)
        run(actor, env, min_rate=0.05, writer=writer)
        # Remove early trajectories when buffer gets too large
        B = B[-args.buffer_size:]

    # Save the model to local directory
    if args.saveas is not None:
        save_path = str(root_dir / 'models' / args.saveas)
        print('Saving models to %s' % save_path)
        torch.save(actor, save_path + '_actor.pt')
        torch.save(critic, save_path + '_critic.pt')
        print('...done')

    # Close writer
    try:
        writer.close()
    except:
        pass
