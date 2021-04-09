import gym
import math
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from environments import LazyFrames
from typing import List, Dict, Tuple

import memory
import models
import environments



class AgentInterface():
    """
    Informal interface for agent class.

    Any agent class implemented needs to have these methods, that accepts and
    returns the types indicated.

    """

    def __init__(self, args: argparse.Namespace, env: gym.RewardWrapper, mem_module: memory.MemoryInterface, dir_name: str, device: str) -> None:
        """
        Agent class/module will be initialized with entire argparse Namespace,
        as well the environment class, the memory module, directory where all
        logging should be stored, and the torch device to use.
        """
        pass

    def act(self,state: LazyFrames) -> int:
        """
        Given an environment state LazyFrame (can be used as np.array)
        should return a selected action as int to take in environment.
        """
        pass

    def act_evaluate(self,state: LazyFrames) -> int:
        """
        Given an environment state LazyFrame (can be used as np.array)
        should return a selected action as int to take in environment. This
        method is only used during evaluation.
        """
        pass

    def learn(self) -> None:
        """
        Method will be called every transition, and only opportunity for agent
        to learn. So all tracking of number of steps acted/passed and when to
        start training should be done internally.
        """
        pass

    def save_model(self,model_path: str) -> None:
        """
        Will be called when agent surpasses current best average performance.
        Gives agent opportunity to checkpoint model (state_dict (PyTorch),
        pickle file (SkLearn) etc.), model file should be saved at model_path.
        """
        pass

    def load_model(self,model_path: str) -> None:
        """
        Will be called when resuming training or evaluating agent. model_path
        will be path checkpointed model, which agent should load.
        """
        pass


def get_agent(agent_name: str, env: gym.RewardWrapper, mem_module: memory.MemoryInterface, dir_name: str, device: str, args: argparse.Namespace) -> AgentInterface:
    """
    Returns agent type based off agent_type string.
    """
    if agent_name == "DQN":
        return DQN(args,env,mem_module,dir_name,device)
    elif agent_name == "DoubleDQN":
        return DoubleDQN(args,env,mem_module,dir_name,device)
    elif agent_name == "NoisyDQN":
        return NoisyDQN(args,env,mem_module,dir_name,device)
    elif agent_name == "DistributionalDQN":
        return DistributionalDQN(args,env,mem_module,dir_name,device)
    else:
        raise NotImplementedError


def prepare_state(state: List[LazyFrames], device: str) -> torch.tensor:
    """Prepares np.array of LazyFrames for procesing by PyTorch Models"""
    state = state/255.0
    state = state.transpose(0,3,1,2)
    return torch.from_numpy(state).float().to(device)

def prepare_state_single(state: LazyFrames, device: str) -> torch.tensor:
    """Prepares single LazyFrame for procesing by PyTorch Models"""
    state = np.array(state)
    state = state/255.0
    state = state.transpose(2,0,1)
    state = np.expand_dims(state,axis=0)
    return torch.from_numpy(state).float().to(device)



class DQN(AgentInterface):

    def __init__(self, args, env, mem_module, dir_name, device):
        self.args = args
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.memory = mem_module
        self.dir_name = dir_name
        self.device = device

        if args.dueling_nets:
            self.policy_net = models.DuelingCNNModel(n_actions=self.action_space).to(self.device)
        else:
            self.policy_net = models.CNNModel(n_actions=self.action_space).to(self.device)
        self.total_steps = 0
        # Setup optimizer and loss
        self.opt = torch.optim.Adam(self.policy_net.parameters(), lr=self.args.learning_rate, eps=args.optimizer_eps)
        self.loss_fn = torch.nn.SmoothL1Loss(reduction="none")

    def load_model(self,model_path):
        self.policy_net.load_state_dict(torch.load(model_path,map_location=torch.device(self.device)))

    def save_model(self,model_path):
        torch.save(self.policy_net.state_dict(),model_path)

    def act(self, state):
        self.total_steps += 1
        sample = random.random()
        eps_threshold = self.args.epsilon_end + (self.args.epsilon_start - self.args.epsilon_end)* math.exp(-1. * self.total_steps / self.args.epsilon_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                torch_state = prepare_state_single(state,self.device)
                q_values_torch = self.policy_net(torch_state)
                q_values = q_values_torch.cpu().numpy().squeeze()
            return np.argmax(q_values)
        else:
            return random.randrange(self.action_space)

    def act_evaluate(self, state):
        self.policy_net.eval()
        with torch.no_grad():
            torch_state = prepare_state_single(state,self.device)
            q_values_torch = self.policy_net(torch_state)
            q_values = q_values_torch.cpu().numpy().squeeze()
        return np.argmax(q_values)

    def learn(self):
        if len(self.memory.memory) < self.args.batch_size or self.total_steps < self.args.start_learn:
            return
        batch, idxs, weights = self.memory.sample(self.args.batch_size)
        loss_value = 0

        # Process as batch
        start_states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        for bt in batch:
            start_states.append(bt[0])
            actions.append(bt[1])
            rewards.append(bt[2])
            next_states.append(bt[3])
            dones.append(bt[4])
        start_states = np.array(start_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        not_dones = np.logical_not(dones).astype(int)

        # Prepare for input
        torch_state = prepare_state(start_states,self.device)
        torch_state_next = prepare_state(next_states,self.device)

        # Predict reqward for next state
        with torch.no_grad():
            output_next_state = self.policy_net(torch_state_next).cpu().numpy().squeeze()
        # get max reward
        predicted_max_reward = np.max(output_next_state,axis=1)
        # Mask out if done not done =0
        predicted_max_reward = np.multiply(predicted_max_reward,not_dones)
        # Now make total q update as reward + discounted reward
        q_update = torch.from_numpy(np.add(rewards,self.args.gamma**self.memory.num_steps * predicted_max_reward)).float().to(self.device)

        # Now make predictions for these states
        predicted_q_values = self.policy_net(torch_state)
        # Edit selected action with true reward + estimated discounted award for next state
        real_q_values = predicted_q_values.detach().clone()
        real_q_values[range(self.args.batch_size),torch.from_numpy(actions)] = q_update
        # Compute loss
        self.opt.zero_grad()
        loss = self.loss_fn(predicted_q_values,real_q_values).sum(1)
        # Reweight loss as per memory
        torch_weights = torch.from_numpy(weights.reshape(-1)).to(self.device)
        torch_weights.requires_grad = False
        loss_value = torch.mean(loss * torch_weights)
        # back propagate and clip gradient values
        loss_value.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        # Optimize
        self.opt.step()
        # Use loss to inform memory sampling, add small noise to loss val
        priority_from_loss = loss.data.cpu().numpy() + .000001
        self.memory.update_priority_levels(idxs,priority_from_loss)

class DoubleDQN(AgentInterface):

    def __init__(self, args, env, mem_module, dir_name, device):
        self.args = args
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.memory = mem_module
        self.dir_name = dir_name
        self.device = device

        if args.dueling_nets:
            self.policy_net = models.DuelingCNNModel(n_actions=self.action_space).to(self.device)
            self.target_net = models.DuelingCNNModel(n_actions=self.action_space).to(self.device)
        else:
            self.policy_net = models.CNNModel(n_actions=self.action_space).to(self.device)
            self.target_net = models.CNNModel(n_actions=self.action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.total_steps = 0
        # Setup optimizer and loss
        self.opt = torch.optim.Adam(self.policy_net.parameters(), lr=self.args.learning_rate, eps=args.optimizer_eps)
        self.loss_fn = torch.nn.SmoothL1Loss(reduction="none")

    def load_model(self,model_path):
        self.policy_net.load_state_dict(torch.load(model_path,map_location=torch.device(self.device)))

    def save_model(self,model_path):
        torch.save(self.policy_net.state_dict(),model_path)

    def act(self, state):
        self.total_steps += 1
        # If total steps at level update target net
        if self.total_steps % self.args.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        # Now take action
        sample = random.random()
        eps_threshold = self.args.epsilon_end + (self.args.epsilon_start - self.args.epsilon_end)* math.exp(-1. * self.total_steps / self.args.epsilon_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                torch_state = prepare_state_single(state,self.device)
                q_values_torch = self.policy_net(torch_state)
                q_values = q_values_torch.cpu().numpy().squeeze()
            return np.argmax(q_values)
        else:
            return random.randrange(self.action_space)

    def act_evaluate(self, state):
        self.policy_net.eval()
        with torch.no_grad():
            torch_state = prepare_state_single(state,self.device)
            q_values_torch = self.policy_net(torch_state)
            q_values = q_values_torch.cpu().numpy().squeeze()
        return np.argmax(q_values)

    def learn(self):
        if len(self.memory.memory) < self.args.batch_size or self.total_steps < self.args.start_learn:
            return
        batch, idxs, weights = self.memory.sample(self.args.batch_size)
        loss_value = 0

        # Process as batch
        start_states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        for bt in batch:
            start_states.append(bt[0])
            actions.append(bt[1])
            rewards.append(bt[2])
            next_states.append(bt[3])
            dones.append(bt[4])
        start_states = np.array(start_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        not_dones = np.logical_not(dones).astype(int)

        # Prepare for input
        torch_state = prepare_state(start_states,self.device)
        torch_state_next = prepare_state(next_states,self.device)

        # Predict reqward for next state
        with torch.no_grad():
            output_next_state = self.target_net(torch_state_next).cpu().numpy().squeeze()
        # get max reward
        predicted_max_reward = np.max(output_next_state,axis=1)
        # Mask out if done not done =0
        predicted_max_reward = np.multiply(predicted_max_reward,not_dones)
        # Now make total q update as reward + discounted reward
        q_update = torch.from_numpy(np.add(rewards,self.args.gamma**self.memory.num_steps * predicted_max_reward)).float().to(self.device)

        # Now make predictions for these states
        predicted_q_values = self.policy_net(torch_state)
        # Edit selected action with true reward + estimated discounted award for next state
        real_q_values = predicted_q_values.detach().clone()
        real_q_values[range(self.args.batch_size),torch.from_numpy(actions)] = q_update
        # Compute loss
        self.opt.zero_grad()
        loss = self.loss_fn(predicted_q_values,real_q_values).sum(1)
        # Reweight loss as per memory
        torch_weights = torch.from_numpy(weights.reshape(-1)).to(self.device)
        torch_weights.requires_grad = False
        loss_value = torch.mean(loss * torch_weights)
        # back propagate and clip gradient values
        loss_value.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        # Optimize
        self.opt.step()
        # Use loss to inform memory sampling, add small noise to loss val
        priority_from_loss = loss.data.cpu().numpy() + .000001
        self.memory.update_priority_levels(idxs,priority_from_loss)

class NoisyDQN(AgentInterface):

    def __init__(self, args, env, mem_module, dir_name, device):
        self.args = args
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.memory = mem_module
        self.dir_name = dir_name
        self.device = device

        self.policy_net = models.NoisyDuelingCNNModel(n_actions=self.action_space).to(self.device)
        self.target_net = models.NoisyDuelingCNNModel(n_actions=self.action_space).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.train()
        self.target_net.eval()
        self.total_steps = 0
        # Setup optimizer and loss
        self.opt = torch.optim.Adam(self.policy_net.parameters(), lr=self.args.learning_rate, eps=args.optimizer_eps)
        self.loss_fn = torch.nn.SmoothL1Loss(reduction="none")

    def load_model(self,model_path):
        self.policy_net.load_state_dict(torch.load(model_path,map_location=torch.device(self.device)))

    def save_model(self,model_path):
        torch.save(self.policy_net.state_dict(),model_path)

    def act(self, state):
        self.total_steps += 1
        # If total steps at level update target net
        if self.total_steps % self.args.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.policy_net.train()
            self.target_net.eval()
        # Now take action
        with torch.no_grad():
            torch_state = prepare_state_single(state,self.device)
            q_values_torch = self.policy_net(torch_state)
            q_values = q_values_torch.cpu().numpy().squeeze()
        return np.argmax(q_values)

    def act_evaluate(self, state):
        self.policy_net.eval()
        with torch.no_grad():
            torch_state = prepare_state_single(state,self.device)
            q_values_torch = self.policy_net(torch_state)
            q_values = q_values_torch.cpu().numpy().squeeze()
        return np.argmax(q_values)


    def learn(self):
        if len(self.memory.memory) < self.args.batch_size or self.total_steps < self.args.start_learn:
            return
        batch, idxs, weights = self.memory.sample(self.args.batch_size)
        loss_value = 0

        # Process as batch
        start_states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        for bt in batch:
            start_states.append(bt[0])
            actions.append(bt[1])
            rewards.append(bt[2])
            next_states.append(bt[3])
            dones.append(bt[4])
        start_states = np.array(start_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        not_dones = np.logical_not(dones).astype(int)

        # Prepare for input
        torch_state = prepare_state(start_states,self.device)
        torch_state_next = prepare_state(next_states,self.device)

        # Predict reqward for next state
        with torch.no_grad():
            output_next_state = self.target_net(torch_state_next).cpu().numpy().squeeze()
        # get max reward
        predicted_max_reward = np.max(output_next_state,axis=1)
        # Mask out if done not done =0
        predicted_max_reward = np.multiply(predicted_max_reward,not_dones)
        # Now make total q update as reward + discounted reward
        q_update = torch.from_numpy(np.add(rewards,self.args.gamma**self.memory.num_steps * predicted_max_reward)).float().to(self.device)

        # Now make predictions for these states
        predicted_q_values = self.policy_net(torch_state)
        # Edit selected action with true reward + estimated discounted award for next state
        real_q_values = predicted_q_values.detach().clone()
        real_q_values[range(self.args.batch_size),torch.from_numpy(actions)] = q_update
        # Compute loss
        self.opt.zero_grad()
        loss = self.loss_fn(predicted_q_values,real_q_values).sum(1)
        # Reweight loss as per memory
        torch_weights = torch.from_numpy(weights.reshape(-1)).to(self.device)
        torch_weights.requires_grad = False
        loss_value = loss * torch_weights
        loss_value = torch.mean(loss * torch_weights)
        # back propagate and clip gradient values
        loss_value.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        # Optimize
        self.opt.step()
        # Use loss to inform memory sampling, add small noise to loss val
        priority_from_loss = loss.data.cpu().numpy() + .000001
        self.memory.update_priority_levels(idxs,priority_from_loss)

        # Reset noise
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

class DistributionalDQN(AgentInterface):


    def __init__(self, args, env, mem_module, dir_name, device):
        self.args = args
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.memory = mem_module
        self.dir_name = dir_name
        self.device = device


        # Model settings
        self.v_min = args.v_min
        self.v_max = args.v_max
        self.number_atoms = args.number_atoms
        self.support_vec = torch.linspace(self.v_min, self.v_max, self.number_atoms).to(self.device)

        self.policy_net = models.DistributionalNoisyDuelingCNNModel(self.number_atoms,n_actions=self.action_space).to(self.device)
        self.target_net = models.DistributionalNoisyDuelingCNNModel(self.number_atoms,n_actions=self.action_space).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.train()
        self.target_net.eval()
        self.total_steps = 0
        # Setup optimizer
        self.opt = torch.optim.Adam(self.policy_net.parameters(), lr=self.args.learning_rate, eps=args.optimizer_eps)

    def load_model(self,model_path):
        self.policy_net.load_state_dict(torch.load(model_path,map_location=torch.device(self.device)))

    def save_model(self,model_path):
        torch.save(self.policy_net.state_dict(),model_path)

    def act(self, state):
        self.total_steps += 1
        # If total steps at level update target net
        if self.total_steps % self.args.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.policy_net.train()
            self.target_net.eval()
        # Now take action
        with torch.no_grad():
            torch_state = prepare_state_single(state,self.device)
            distribution = self.policy_net(torch_state)
            distribution = distribution * self.support_vec
            # Get expectation of all by summing over distribution
            expected_value = torch.sum(distribution, dim=2).cpu().numpy().squeeze()
        # Now take arg max for action with largest expected return
        return np.argmax(expected_value)

    def act_evaluate(self, state):
        self.policy_net.eval()
        with torch.no_grad():
            torch_state = prepare_state_single(state,self.device)
            distribution = self.policy_net(torch_state).data
            distribution = distribution * self.support_vec
            # Get expectation of all by summing over distribution
            expected_value = torch.sum(distribution, dim=2).cpu().numpy().squeeze()
        # Now take arg max for action with largest expected return
        return np.argmax(expected_value)

    def learn(self):
        if len(self.memory.memory) < self.args.batch_size or self.total_steps < self.args.start_learn:
            return 0
        batch, idxs, weights = self.memory.sample(self.args.batch_size)
        loss_value = 0

        # Process as batch
        start_states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        for bt in batch:
            start_states.append(bt[0])
            actions.append(bt[1])
            rewards.append(bt[2])
            next_states.append(bt[3])
            dones.append(bt[4])
        start_states = np.array(start_states)
        actions = torch.from_numpy(np.array(actions)).to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).to(self.device)
        next_states = np.array(next_states)
        dones = torch.from_numpy(np.array(dones).astype(int)).to(self.device)

        # Prepare for input
        torch_state = prepare_state(start_states,self.device)
        torch_state_next = prepare_state(next_states,self.device)

        # Prepare delta z
        delta_z = float(self.v_max - self.v_min) / (self.number_atoms - 1)
        # Now get predicted distribution
        pred_distribution = self.policy_net(torch_state)
        # Change to be just action distributions
        pred_distribution = pred_distribution[range(self.args.batch_size),actions]
        # Get projection distribituion
        with torch.no_grad():
            # Get target predicted distribution
            next_distribution = self.target_net(torch_state_next)
            # Get q values by multiplying by support
            q_vals = next_distribution * self.support_vec
            # Select best action according to target nets q vals
            next_action = torch.sum(q_vals,dim=2).argmax(1)
            # Only select distribution for best predicted action
            next_distribution = next_distribution[range(self.args.batch_size), next_action]

            # Reshape for easier calculations
            rewards = rewards.unsqueeze(1)
            dones = dones.unsqueeze(1)
            support_vec = self.support_vec.unsqueeze(0)

            Tz = rewards + (1 - dones) * (self.args.gamma**self.memory.num_steps) * support_vec
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            b  = (Tz - self.v_min) / delta_z

            l  = b.floor().to(torch.int64)
            u  = b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.number_atoms - 1)) * (l == u)] += 1

            # Compute offset
            offset = torch.linspace(0, (self.args.batch_size - 1) * self.number_atoms, self.args.batch_size).unsqueeze(1).long().expand(self.args.batch_size, self.number_atoms).to(self.device)

            # Compute projected distribituion
            projected_distribution = torch.zeros(next_distribution.size()).float().to(self.device)
            projected_distribution.view(-1).index_add_(0, (l + offset).view(-1), (next_distribution * (u.float() - b)).float().view(-1))
            projected_distribution.view(-1).index_add_(0, (u + offset).view(-1), (next_distribution * (b - l.float())).float().view(-1))


        # Compute KL loss between projected distribution and predicted
        loss = -torch.sum(projected_distribution * pred_distribution.clamp(min=1e-7).log(), 1)
        # Now weight loss as per sample weight and take mean
        torch_weights = torch.from_numpy(weights.reshape(-1)).to(self.device)
        torch_weights.requires_grad = False
        loss_value = (loss * torch_weights).mean()
        loss_value = loss_value.mean()
        # Back propagate and clip gradient values
        self.opt.zero_grad()
        loss_value.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        # Optimize
        self.opt.step()
        # Use loss to inform memory sampling, add small noise to loss val
        priority_from_loss = loss.data.cpu().numpy() + .000001
        self.memory.update_priority_levels(idxs,priority_from_loss)

        # Reset noise
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
