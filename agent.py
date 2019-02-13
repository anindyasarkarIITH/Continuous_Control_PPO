#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# Modified by Jeremi Kaczmarczyk (jeremi.kaczmarczyk@gmail.com) 2018 
# For Udacity Deep Reinforcement Learning Nanodegree

import numpy as np
import torch
import torch.nn as nn

from imported_utils import Batcher


class PPOAgent(object):
    
    def __init__(self, environment, brain_name, policy_network, optimizier, config):
        self.config = config
        self.hyperparameters = config['hyperparameters']
        self.network = policy_network
        self.optimizier = optimizier
        self.total_steps = 0
        #self.all_rewards = np.zeros(config['environment']['number_of_agents'])
        self.all_rewards = np.ones(config['environment']['number_of_agents'])
        
        self.episode_rewards = []
        self.environment = environment
        self.brain_name = brain_name
        
        env_info = environment.reset(train_mode=True)[brain_name]    
        self.states = env_info.vector_observations  # self.states --> [1,33]            
        

    def step(self):
        rollout = []
        hyperparameters = self.hyperparameters

        env_info = self.environment.reset(train_mode=True)[self.brain_name]    
        self.states = env_info.vector_observations  
        states = self.states
        for _ in range(hyperparameters['rollout_length']):   ## rollout_length --> 2048
        #for _ in range(66):
            actions, log_probs, _, values = self.network(states)   ## returning action,log_prob,values
            env_info = self.environment.step(actions.cpu().detach().numpy())[self.brain_name]  ## Take step in environment
            next_states = env_info.vector_observations  # resulting next state
            rewards = env_info.rewards  # getting rewards from environment
            #print (type(rewards))
            terminals = np.array([1 if t else 0 for t in env_info.local_done]) # (shape (1))
            self.all_rewards += rewards
            #### FOR THE TERMINAL STATE	
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.all_rewards[i])
                    self.all_rewards[i] = 0
                    
                    
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - terminals])
            states = next_states
        
        self.states = states
        pending_value = self.network(states)[-1]
        rollout.append([states, pending_value, None, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)   # [None, None] #len is same as rollout...
        advantages = torch.Tensor(np.zeros((self.config['environment']['number_of_agents'], 1))) ## [1,1]
        #print (advantages.size())
        #print (len(processed_rollout))
        #print (processed_rollout)
        returns = pending_value.detach()
        #expected_returns = returns.mean()
        #print (expected_returns)
        
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i] ## rewards is a list
            
            terminals = torch.Tensor(terminals).unsqueeze(1)
            
            rewards = torch.Tensor(rewards).unsqueeze(1)
            actions = torch.Tensor(actions)
            states = torch.Tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + hyperparameters['discount_rate'] * terminals * returns ## v(s) = r + y*v(s+1)
            ## td_err is same as A(s,a) in literature

            td_error = rewards + hyperparameters['discount_rate'] * terminals * next_value.detach() - value.detach() # td_err=q(s,a) - v(s)
            # calc. of discounted advantage= A(s,a) + y^1*A(s+1,a+1) + ...
            advantages = advantages * hyperparameters['tau'] * hyperparameters['discount_rate'] * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]
        
        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        #print (states.size()) ; print (advantages.size()) ; print (actions.size()) ; print (log_probs_old.size()); print (returns.size())
        advantages = (advantages - advantages.mean()) / advantages.std()
        #print ([states.size(0) // hyperparameters['mini_batch_number']])
        batcher = Batcher(states.size(0) // hyperparameters['mini_batch_number'], [np.arange(states.size(0))])
        
        ### optimization process done here
        for _ in range(hyperparameters['optimization_epochs']):   #10
            batcher.shuffle()
            
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = torch.Tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]                
                sampled_returns = returns[batch_indices]                
                sampled_advantages = advantages[batch_indices]
                _, log_probs, entropy_loss, values = self.network(sampled_states, sampled_actions)  # [2,4]...[2,33]
                #print (entropy_loss)  ### regularizer but in this case it's value is 0
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - hyperparameters['ppo_clip'],
                                          1.0 + hyperparameters['ppo_clip']) * sampled_advantages
                '''
                # calculation of entropy
                # add in 1.e-10 to avoid log(0) which gives nan
                entropy_loss = -(log_probs*torch.log(sampled_log_probs_old+1.e-10)+ \
                    (1.0-log_probs)*torch.log(1.0-sampled_log_probs_old+1.e-10))
                '''
                policy_loss = -torch.min(obj, obj_clipped).mean(0) - hyperparameters['entropy_coefficent'] * entropy_loss.mean()
                # loss = -(clipped surrogate objective - entropy loss) = -clipped surrogate objective - (-entropy loss)

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()  ## Added new loss for state-value calc.

                self.optimizier.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), hyperparameters['gradient_clip'])
                self.optimizier.step()
            
        steps = hyperparameters['rollout_length'] * self.config['environment']['number_of_agents']
        self.total_steps += steps
        
        
