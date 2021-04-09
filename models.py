import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):

        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(3136, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc4(x))
        return self.head(x)

class DuelingCNNModel(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):

        super(DuelingCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Value net
        self.value_fc = nn.Linear(3136, 512)
        self.value_pred = nn.Linear(512, 1)

        # Advantage net
        self.advantage_fc = nn.Linear(3136, 512)
        self.advantage_pred = nn.Linear(512, n_actions)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0],-1)

        # Compute value
        value_val = self.value_pred(F.relu(self.value_fc(x)))

        # Compute advantage
        advantage_val = self.advantage_pred(F.relu(self.advantage_fc(x)))

        # Return estimated q val
        return value_val + (advantage_val - advantage_val.mean())


class NoisyNet(nn.Module):
    """
    Re-formatted but originally from: https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb

    """
    def __init__(self, in_dim, out_dim, std=0.5):
        super(NoisyNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.std = std

        # Weight for linear layer
        self.mean_weight = nn.Parameter(torch.FloatTensor(out_dim, in_dim))
        self.std_weight = nn.Parameter(torch.FloatTensor(out_dim, in_dim))

        # Bias for linear layer
        self.mean_bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.std_bias = nn.Parameter(torch.FloatTensor(out_dim))

        # Register as buffer
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_dim, in_dim))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_dim))

        # Reset values
        self.reset_params()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            w = self.mean_weight + self.std_weight.mul(self.weight_epsilon)
            b = self.mean_bias + self.std_bias.mul(self.bias_epsilon)
        else:
            w = self.mean_weight
            b = self.mean_bias
        return F.linear(x, w, b)

    def reset_params(self):
        # Set mean range
        mean_range = 1/math.sqrt(self.mean_weight.size(1))

        # Fill weight and bias data
        self.mean_weight.data.uniform_(-mean_range, mean_range)
        self.std_weight.data.fill_(self.std / math.sqrt(self.std_weight.size(1)))
        self.mean_bias.data.uniform_(-mean_range, mean_range)
        self.std_bias.data.fill_(self.std / math.sqrt(self.std_bias.size(0)))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_dim)
        epsilon_out = self.scale_noise(self.out_dim)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self.scale_noise(self.out_dim))

    def scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x



class NoisyDuelingCNNModel(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):

        super(NoisyDuelingCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Value noisy net
        self.value_fc = NoisyNet(3136, 512)
        self.value_pred = NoisyNet(512, 1)

        # Advantage noisy net
        self.advantage_fc = NoisyNet(3136, 512)
        self.advantage_pred = NoisyNet(512, n_actions)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0],-1)

        # Compute value
        value_val = self.value_pred(F.relu(self.value_fc(x)))

        # Compute advantage
        advantage_val = self.advantage_pred(F.relu(self.advantage_fc(x)))

        # Return estimated q val
        return value_val + (advantage_val - advantage_val.mean())

    def reset_noise(self):
        self.value_fc.reset_noise()
        self.value_pred.reset_noise()
        self.advantage_fc.reset_noise()
        self.advantage_pred.reset_noise()


class DistributionalNoisyDuelingCNNModel(nn.Module):
    def __init__(self, number_atoms,in_channels=4, n_actions=14):

        super(DistributionalNoisyDuelingCNNModel, self).__init__()

        self.number_atoms = number_atoms
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Value noisy net
        self.value_fc = NoisyNet(3136, 512)
        self.value_pred = NoisyNet(512, self.number_atoms)

        # Advantage noisy net
        self.advantage_fc = NoisyNet(3136, 512)
        self.advantage_pred = NoisyNet(512, n_actions*self.number_atoms)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0],-1)

        # Compute advantage
        advantage_val = self.advantage_pred(F.relu(self.advantage_fc(x)))
        advantage_val = advantage_val.view(-1, self.n_actions, self.number_atoms)

        # Compute value
        value_val = self.value_pred(F.relu(self.value_fc(x)))
        value_val = value_val.view(-1, 1, self.number_atoms)

        # Compute q distribution
        q_atoms = value_val + (advantage_val - advantage_val.mean(dim=1, keepdim=True))

        # Take softmax
        q_atoms = F.softmax(q_atoms,dim=2)
        return q_atoms

    def reset_noise(self):
        self.value_fc.reset_noise()
        self.value_pred.reset_noise()
        self.advantage_fc.reset_noise()
        self.advantage_pred.reset_noise()




#
