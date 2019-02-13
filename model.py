import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FullyConnectedNetwork(nn.Module):
    
    def __init__(self, state_size, output_size, hidden_size, output_gate=None):
        super(FullyConnectedNetwork, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)  ## (33,512)
        self.linear2 = nn.Linear(hidden_size, hidden_size) ## (512,512)
        self.linear3 = nn.Linear(hidden_size, output_size) ## (512,4)
        self.output_gate = output_gate

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        if self.output_gate:
            x = self.output_gate(x)
        return x


class PPOPolicyNetwork(nn.Module):
    
    def __init__(self, config):
        super(PPOPolicyNetwork, self).__init__()
        state_size = config['environment']['state_size']   # state size -> 33
        action_size = config['environment']['action_size'] # action_size -> 4
        hidden_size = config['hyperparameters']['hidden_size'] # hidden_dim ->512
        device = config['pytorch']['device']

        self.actor_body = FullyConnectedNetwork(state_size, action_size, hidden_size, F.tanh) ## maps 33 -> 4 range (-1 to 1)
        self.critic_body = FullyConnectedNetwork(state_size, 1, hidden_size)                  ## maps 33 -> 1 (value)
        self.std = nn.Parameter(torch.ones(1, action_size))
        self.to(device)
        
    def forward(self, obs, action=None):
        obs = torch.Tensor(obs)
        a = self.actor_body(obs)
        v = self.critic_body(obs)
        #print (a.size())
        #print (v.size())
        dist = torch.distributions.Normal(a, self.std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
                 
        return action, log_prob, torch.Tensor(np.zeros((log_prob.size(0), 1))), v




