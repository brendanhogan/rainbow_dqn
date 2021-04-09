import os
import time
import json
import torch
import pickle
import argparse
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter

import memory
import agents
import environments


def log_metrics(metric_dict,episode_num):
    """
    Logs all metrics to tensorboardX SummaryWriter
    """
    for key in metric_dict.keys():
        label_text = "{}".format(key)
        logger.add_scalar(label_text,metric_dict[key],episode_num)


def run_episode(env,agent):
    """
    Run a single episode of enviornment as determined by env terminal.

    Return total reward for the episode.

    """
    # 1. Setup logging vars
    total_reward = 0

    # 1. Reset state
    state = env.reset()

    # 2. Run episode until terminal
    while True:
        # 2a. Render evironment if selected
        if args.render:
            env.render()
            time.sleep(.08)
        # 2b. Have agent act on current state
        state_np = np.array(state)
        action = agent.act_evaluate(state_np)
        # 2c. Take action in env record outcomes
        next_state, reward, terminal, _ = env.step(action)
        # 2d. Update state, logs
        state = next_state
        total_reward += reward
        if terminal:
            break

    # 3. Return updated number of iterations as well as scores
    return total_reward



if __name__ == "__main__":
    # Parse all arguments
    parser = argparse.ArgumentParser(description='Process parameters for evaluating Rainbow DQN')
    parser.add_argument('--num_episodes', '-n', type=int, default=int(10), help='number of episodes to evaluate')
    parser.add_argument('--env_name', '-en', type=str, default="PongNoFrameskip-v4", help='which enviornment to do training for')
    parser.add_argument('--model_path', type=str, help='full path to model to test')
    parser.add_argument('--agent_type', '-at', type=str, default="DoubleDQN", help='which agent to use for RL learning')
    parser.add_argument('--dueling_nets', '-dn', action='store_true',help='whether to use dueling architecture or not')
    parser.add_argument('--avg_episodes', type=int, default=10, help='number of episodes to average score over')
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')


    # Initilized for agent but never used
    parser.add_argument('--gamma', '-g', type=float, default=.99, help='discount factor')
    parser.add_argument('--epsilon_start', '-es', type=float, default=.02, help='when using e-greedy what starting epsilon should be')
    parser.add_argument('--epsilon_end', '-ee', type=float, default=.02, help='when using e-greedy what ending epsilon should be')
    parser.add_argument('--epsilon_decay', '-ed', type=float, default=1000000, help='when using e-greedy what decay rate should be')
    parser.add_argument('--target_update', '-tu', type=int, default=32000, help='frame frequency to update target network (double DQN)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0000625, help='learning rate for ADAM optimizer')
    parser.add_argument('--optimizer_eps', type=float, default=0.00015, help='epsilon ADAM optimizer')
    parser.add_argument('--v_min', type=float, default=-10, help='min value for distributional DQN')
    parser.add_argument('--v_max', type=float, default=10., help='max value for distributional DQN')
    parser.add_argument('--number_atoms', type=int, default=51, help='number of atoms for distributional DQN')
    parser.add_argument('--memory_type', '-mt', type=str, default="StandardMemory", help='which memory type to use for RL learning')
    parser.add_argument('--mem_size', '-ms', type=int, default=10, help='number of transitions to store in memory')
    parser.add_argument('--num_steps', '-ns', type=int, default=1, help='number of n-steps to look ahead')
    parser.add_argument('--alpha', type=float, default=.5, help='Priority exponent (used in priority replay)')
    parser.add_argument('--beta', type=float, default=.4, help='Reweighting of priorities ')


    args = parser.parse_args()
    if args.verbose: print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.verbose: print("Using device: {}".format(device))
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Setup tracking vars/directory and logger
    all_logging_dict = {}
    num_episodes = 0
    best_avg_reward = 0
    rewards = []
    dt_string = datetime.now().strftime("%d%m%Y%H%M%S")
    dir_name = "runs/eval_env_{}_agent_{}_memory_{}_{}".format(args.env_name,args.agent_type,args.memory_type,dt_string)
    logger = SummaryWriter(log_dir=dir_name)
    with open(os.path.join(dir_name,'command_line_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Prepare envirornment
    env = environments.get_env(args.env_name)

    # Prepare memory module
    memory_module = memory.get_module(args.memory_type, args)

    # Prepare agent
    agent = agents.get_agent(args.agent_type, env, memory_module, dir_name, device, args)

    # Load saved model
    agent.load_model(args.model_path)

    # Iterate through episodes
    for episode in range(args.num_episodes):
        # Run episode and get reward
        rewards.append(run_episode(env,agent))
        # Get average reward and log results, and add to to all_logging_dict
        reward_np = np.array(rewards)
        i = reward_np.shape[0] if reward_np.shape[0] < args.avg_episodes else args.avg_episodes
        avg_reward = reward_np[-i:].mean()
        logging_dict = {"avg_rewards":avg_reward, "reward": rewards[-1], "num_episodes": episode}
        log_metrics(logging_dict,num_episodes)
        all_logging_dict[num_episodes] = logging_dict

        # If verbose, print update
        if args.verbose:
             print("Episode: {} Last Reward: {} Avg. Reward Last {} Episodes: {}".format(episode,rewards[-1],args.avg_episodes,avg_reward))

    # Close out logger
    logger.close()
    # Save full logs
    output = open(os.path.join(dir_name,"full_logs.pkl"), 'wb')
    pickle.dump(all_logging_dict, output)
    output.close()





#
