import random
import argparse
import numpy as np
from collections import deque
from environments import LazyFrames
from typing import List, Dict, Tuple



class MemoryInterface():
    """
    Informal interface for memory class.

    Any memory class implemented needs to have these methods, that accepts and
    returns the types indicated.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Memory class/module will be initialized with entire argparse Namespace.
        If necessary can add items to argparse in main script.
        """
        pass

    def remember(self, state: LazyFrames, action: int, reward: int, next_state: LazyFrames, terminal: bool) -> None:
        """
        Every transition experienced by agent will be passed to remember.

        LazyFrames can be used as just np.array, but are highly recomended for
        limiting memory usage.
        """
        pass

    def sample(self,sample_size: int) -> Tuple[List[Tuple[LazyFrames,int,int,LazyFrames,bool]],np.array,np.array]:
        """
        Will recieve integer request for a sample size of the transitions.

        Should return a tuple of a list of tuples of the sampled transitions
        (state,action,reward,next_state,terminal), a (sample_size,) np.array of
        the indexes of the samples within the memory, and another (sample_size,)
        np.array of weights for the samples. The weights are applied to the loss
        function, so to ignore this feature can return a np.ones(sample_size).
        The indexes are only for self reference. I.e. they are passed back to
        the module in the "update_priority_levels" method and are not used
        anywhere else. So can ignore them by also passing np.ones(sample_size).
        """
        pass

    def update_priority_levels(self,idxs: np.array, vals: np.array) -> None:
        """
        Will recieve from agent indices (as provided by sample method above),
        and corresponding loss values.

        Used in priority experince replay.

        """

        pass


def get_module(mem_type: str, args: argparse.Namespace) -> MemoryInterface:
    """
    Returns memory type based off mem_type string.
    """
    if mem_type == "StandardMemory":
        return StandardMemory(args)
    elif mem_type == "PrioritizedReplay":
        return PrioritizedReplay(args)
    else:
        raise NotImplementedError




class StandardMemory(MemoryInterface):

    def __init__(self,args):
        # Make memory deque
        self.memory = deque(maxlen=args.mem_size)
        self.gamma = args.gamma
        # If n_steps > 1 want to return discounted cumulative reward and nth state
        # So build seperate buffer to build to that
        self.num_steps = args.num_steps
        if self.num_steps > 1:
            self.n_step_buffer = deque(maxlen=args.num_steps)

    def remember(self,state, action, reward, next_state, terminal):
        # If no forward look ahead just store transition directly
        if self.num_steps == 1:
            self.memory.append((state, action, reward, next_state, terminal))
        else:
            # Otherwise append to n_step_buffer
            self.n_step_buffer.append((state, action, reward, next_state, terminal))
            # Only can append transition after buffer is at full size
            if len(self.n_step_buffer) == self.num_steps:
                self.memory.append(self.calculate_n_step_transition())

    def calculate_n_step_transition(self):
        """
        Uses n_step_buffer to calculate n_step transition, by storing reward as
        discounted reward, and next_state as nth state.

        For example if n=3 (most common emperically):
        reward <- n_step_buffer[reward][0] + GAMMA*n_step_buffer[reward][1] + GAMMA**2*n_step_buffer[reward][2]
        next_state <- n_step_buffer[next_state][3]

        """
        state = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]
        next_state = self.n_step_buffer[self.num_steps-1][3]
        terminal = self.n_step_buffer[self.num_steps-1][4]
        # Now calculate reward
        reward = 0
        for i in range(self.num_steps):
            reward += self.gamma**i*self.n_step_buffer[i][2]
        # Now return all
        return (state, action, reward, next_state, terminal)

    def sample(self,sample_size):
        batch = random.sample(self.memory, sample_size)
        return batch, np.ones(sample_size), np.ones(sample_size)


class PrioritizedReplay(MemoryInterface):
    """
    Implementation of prioritized replay without sum trees for simpler understanding.
    """

    def __init__(self,args):
        # Make memory deque
        self.memory = deque(maxlen=args.mem_size)
        self.memory_size = args.mem_size
        self.gamma = args.gamma
        # Set hyperparameters
        self.alpha = args.alpha
        self.beta = args.beta
        # Set beta increase schedule
        self.beta_increase = (1.0-self.beta)/float(args.num_frames)
        # Track all priorities, because will have to make inserts, faster if is array
        self.priority_idx = 0
        self.priority_level = np.zeros((args.mem_size), dtype=np.float32)
        # If n_steps > 1 want to return discounted cumulative reward and nth state
        # So build seperate buffer to build to that
        self.num_steps = args.num_steps
        if self.num_steps > 1:
            self.n_step_buffer = deque(maxlen=args.num_steps)



    def remember(self,state, action, reward, next_state, terminal):
        # If no forward look ahead just store transition directly
        if self.num_steps == 1:
            # 1. Append transition
            self.memory.append((state, action, reward, next_state, terminal))
            # 2. Set at highest priority, either 1 if buffer empty or max value if any entries
            self.priority_level[self.priority_idx] = 1 if len(self.memory)==1 else self.priority_level.max()
            # 3. Set index as +1 mod total memory
            self.priority_idx += 1
            self.priority_idx = self.priority_idx % self.memory_size
        else:
            # Otherwise append to n_step_buffer
            self.n_step_buffer.append((state, action, reward, next_state, terminal))
            # Only can append transition after buffer is at full size
            if len(self.n_step_buffer) == self.num_steps:
                # 1. Append transition
                self.memory.append(self.calculate_n_step_transition())
                # 2. Set at highest priority, either 1 if buffer empty or max value if any entries
                self.priority_level[self.priority_idx] = 1 if len(self.memory)==1 else self.priority_level.max()
                # 3. Set index as +1 mod total memory
                self.priority_idx += 1
                self.priority_idx = self.priority_idx % self.memory_size

    def sample(self,sample_size):
        total_transitions = len(self.memory)
        # 1. Clip priorities if buffer not yet filled
        sample_priority_levels = self.priority_level[:total_transitions]
        # 2. Compute actual probablities by raising to alpha and dividing by sum
        sample_priority_levels = np.power(sample_priority_levels,self.alpha)
        sample_priority_levels = np.divide(sample_priority_levels,sample_priority_levels.sum())
        # 3. Sample indices
        idxs_to_sample = np.random.choice(total_transitions, sample_size, p=sample_priority_levels)
        # 4. Calculate weight for each sample
        weights  = total_transitions * sample_priority_levels[idxs_to_sample]
        weights = np.power(weights,-self.beta)
        weights = np.divide(weights,weights.max())
        # 5. Create batch
        batch = [self.memory[idx] for idx in idxs_to_sample]
        # 6. Increment beta
        self.beta += self.beta_increase
        self.beta = min(self.beta, 1.0)
        # 7. Return batch and weights
        return batch, idxs_to_sample, weights

    def update_priority_levels(self,idxs,losses):
        for i in range(idxs.shape[0]):
            self.priority_level[idxs[i]] = losses[i]

    def calculate_n_step_transition(self):
        """
        Uses n_step_buffer to calculate n_step transition, by storing reward as
        discounted reward, and next_state as nth state.

        For example if n=3 (most common emperically):
        reward <- n_step_buffer[reward][0] + GAMMA*n_step_buffer[reward][1] + GAMMA**2*n_step_buffer[reward][2]
        next_state <- n_step_buffer[next_state][3]
        """
        state = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]
        next_state = self.n_step_buffer[self.num_steps-1][3]
        terminal = self.n_step_buffer[self.num_steps-1][4]
        # Now calculate reward
        reward = 0
        for i in range(self.num_steps):
            reward += self.gamma**i*self.n_step_buffer[i][2]
        # Now return all
        return (state, action, reward, next_state, terminal)





#
