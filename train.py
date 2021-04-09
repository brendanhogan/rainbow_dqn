import os
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


def run_episode(env,agent,memory_module,num_iters,max_iters):
    """
    Run a single episode of enviornment as determined by env terminal.

    Return number iterations, and total reward for the episode.

    """
    # 1. Setup logging vars
    total_reward = 0

    # 1. Reset state
    state = env.reset()

    # 2. Run episode until terminal
    while True:
        # 2a. Have agent act on current state
        state_np = np.array(state)
        action = agent.act(state_np)
        # 2b. Take action in env record outcomes
        next_state, reward, terminal, _ = env.step(action)
        # 2c. Add this interaction to replay buffer
        memory_module.remember(state, action, reward, next_state, terminal)
        # 2d. Teach agent
        agent.learn()
        # 2e. Update state, logs, and break if at terminal or if over num_iters
        state = next_state
        total_reward += reward
        num_iters += 1
        if terminal or num_iters > max_iters:
            break

    # 3. Return updated number of iterations as well as scores
    return num_iters, total_reward



if __name__ == "__main__":
    # Parse all arguments
    parser = argparse.ArgumentParser(description='Process parameters for training Rainbow DQN')
    # General parameters
    parser.add_argument('--num_frames', '-n', type=int, default=int(50e6), help='number of iterations to do training')
    parser.add_argument('--gamma', '-g', type=float, default=.99, help='discount factor')
    parser.add_argument('--seed', '-s', type=int, default=1994, help='Set random seed for numpy, torch and cuda')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--avg_episodes_save', type=int, default=10, help='number of episodes to average score over, to check if should save model')

    # Environment parameters
    parser.add_argument('--env_name', '-en', type=str, default="PongNoFrameskip-v4", help='which enviornment to do training for')

    # Agent parameters
    parser.add_argument('--agent_type', '-at', type=str, default="DoubleDQN", help='which agent to use for RL learning')
    parser.add_argument('--dueling_nets', '-dn', action='store_true',help='whether to use dueling architecture or not')
    parser.add_argument('--batch_size', '-bs', type=int, default=32, help='batch size for learning')
    parser.add_argument('--epsilon_start', '-es', type=float, default=1, help='when using e-greedy what starting epsilon should be')
    parser.add_argument('--epsilon_end', '-ee', type=float, default=.02, help='when using e-greedy what ending epsilon should be')
    parser.add_argument('--epsilon_decay', '-ed', type=float, default=1000000, help='when using e-greedy what decay rate should be')
    parser.add_argument('--target_update', '-tu', type=int, default=32000, help='frame frequency to update target network (double DQN)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0000625, help='learning rate for ADAM optimizer')
    parser.add_argument('--optimizer_eps', type=float, default=0.00015, help='epsilon ADAM optimizer')
    parser.add_argument('--v_min', type=float, default=-10.0, help='min value for distributional DQN')
    parser.add_argument('--v_max', type=float, default=10.0, help='max value for distributional DQN')
    parser.add_argument('--number_atoms', type=int, default=51, help='number of atoms for distributional DQN')
    parser.add_argument('--start_learn', type=int, default=20000, help='number of frames to experience before starting to learn')


    # Memory parameters
    parser.add_argument('--memory_type', '-mt', type=str, default="StandardMemory", help='which memory type to use for RL learning')
    parser.add_argument('--mem_size', '-ms', type=int, default=1000000, help='number of transitions to store in memory')
    parser.add_argument('--num_steps', '-ns', type=int, default=3, help='number of n-steps to look ahead')
    parser.add_argument('--alpha', type=float, default=.5, help='Priority exponent (used in priority replay)')
    parser.add_argument('--beta', type=float, default=.4, help='Reweighting of priorities ')




    args = parser.parse_args()
    if args.verbose: print(args)

    # Setup PyTorch, and seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.verbose: print("Using device: {}".format(device))
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Setup tracking vars/directory and logger
    all_logging_dict = {}
    num_iters = 0
    num_episodes = 0
    best_avg_reward = -np.inf
    rewards = []
    dt_string = datetime.now().strftime("%d%m%Y%H%M%S")
    dir_name = "runs/train_env_{}_agent_{}_dueling_{}_nstep_{}_memory_{}_{}".format(args.env_name,args.agent_type,args.dueling_nets,args.num_steps,args.memory_type,dt_string)
    logger = SummaryWriter(log_dir=dir_name)
    with open(os.path.join(dir_name,'command_line_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Prepare envirornment
    env = environments.get_env(args.env_name)

    # Prepare memory module
    memory_module = memory.get_module(args.memory_type, args)

    # Prepare agent
    agent = agents.get_agent(args.agent_type, env, memory_module, dir_name, device, args)

    # Iterate through for set number of iterations
    while num_iters < args.num_frames:
        # Run an episode
        num_iters, reward = run_episode(env,agent,memory_module,num_iters,args.num_frames)

        # Append reward, send to agent and get full logging dictionary back
        rewards.append(reward)

        # Get average reward and log results, and add to to all_logging_dict
        reward_np = np.array(rewards)
        i = reward_np.shape[0] if reward_np.shape[0] < args.avg_episodes_save else args.avg_episodes_save
        avg_reward = reward_np[-i:].mean()
        logging_dict = {"avg_rewards":avg_reward, "reward": rewards[-1], "num_iters": num_iters, "num_episodes": num_episodes}
        log_metrics(logging_dict,num_episodes)
        all_logging_dict[num_episodes] = logging_dict

        # Save model if the average of last runs is better than current best
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save_model(os.path.join(dir_name,"current_best_model.pth"))
            with open(os.path.join(dir_name,'best_parameters.txt'), 'w') as file:
                file.write(json.dumps(str(logging_dict)))

        # Increment episodes
        num_episodes += 1

        # If verbose, print update
        if args.verbose and num_episodes % 20 ==0:
             print("Episode: {} Frames: {} Last Reward: {} Avg. Reward Last {} Episodes: {}".format(num_episodes,num_iters,rewards[-1],args.avg_episodes_save,avg_reward))

    # Close environment
    env.close()
    # Close out logger
    logger.close()
    # Save full logs
    output = open(os.path.join(dir_name,"full_logs.pkl"), 'wb')
    pickle.dump(all_logging_dict, output)
    output.close()





#
