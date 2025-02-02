import math
import random
import torch
import torch.optim as optim
import torch.nn as nn
from algorithms.dqn.network import DQN
from algorithms.dqn.replay_buffer import ReplayMemory, Transition
from configs.dqn_config import DQN_CONFIG

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
    
class DQNAgent:
    
    def __init__(self, n_observations, n_actions, total_training_steps,config=DQN_CONFIG):
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=config["lr"], amsgrad=True)
        self.memory = ReplayMemory(config["memory_capacity"])
        self.steps_done = 0
        self.n_actions = n_actions

        # Load hyperparameters from the config
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.eps_start = config["eps_start"]
        self.eps_end = config["eps_end"]
        self.eps_decay = config["eps_decay"]
        self.tau = config["tau"]
        self.total_training_steps = total_training_steps
        self.eps_threshold = 0
    def store_transition(self,state,action,next_state,reward):
        self.memory.push(state,action,next_state,reward)
    def select_action(self, state):
        sample = random.random()
        # self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
        #     math.exp(-1. * self.steps_done / self.eps_decay)
        # self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
        #     (1 - self.steps_done / self.total_training_steps)
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * (self.steps_done / self.total_training_steps) * self.eps_decay)
        #self.eps_threshold = max(self.eps_end, self.eps_start - (self.eps_start - self.eps_end) * (self.steps_done / self.total_training_steps))
        self.steps_done += 1
        if sample > self.eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(random.randint(0, self.n_actions - 1), device=device, dtype=torch.long).view(1, 1)
    def choose_greedy_action(self,state):
        with torch.no_grad():
            return self.policy_net(state).max(1).indices.view(1, 1)
        

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
        
        return loss.item()