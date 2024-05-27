import numpy as np
import random
import gym
from gym import spaces
from operator import itemgetter

from utils import get_random_seed, is_empty 
from blocksim.blocksim import Blocksim

class BlocksimEnv(gym.Env): 
    def __init__(self, nodes_dict=None, connections=None, broadcast_rules=None, nodes=None, sim_duration=60) -> None: #
        self.reward = []
        self.nodes = nodes
        self.nodes_dict = nodes_dict
        self.connections = connections
        self.broadcast_rules = broadcast_rules
        self.sim_duration = sim_duration
        
        if not self.broadcast_rules:
            sim = Blocksim(nodes_dict=self.nodes_dict, connections=self.connections, duration=self.sim_duration)
            self.broadcast_rules = sim.broadcast_rules.copy()
            del sim           
        
        if not self.nodes:
            self.nodes = list(self.broadcast_rules['new_blocks'].keys())     
        
        self.node = random.choice(self.nodes)        
        self.broadcast_order = self.broadcast_rules["new_blocks"][self.node]       
        
        self.sim = None
        self.statistics = {}
        
        self.action_size = len(self.broadcast_order)
        self.observation_size = self.action_size
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_size, ), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_size, ), dtype=np.float32)

    def get_observation(self):  
        propagation_times = self.sim.get_neighbours_tx_propagation_time(self.node).copy()
        self.statistics = self.sim.get_statistics().copy()
        self.statistics.update({'propagation_times' : propagation_times})
        obs = [propagation_times[n] for n in self.broadcast_order]
        obs = np.array(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=10, posinf=10, neginf=10)
        return obs
    
    def exec_sim(self):
        for i in range(10):
            try:
                self.sim = Blocksim(seed=get_random_seed(),
                                    duration=self.sim_duration,
                                    nodes_dict=self.nodes_dict, 
                                    connections=self.connections, 
                                    broadcast_rules=self.broadcast_rules, 
                                    randomize_connections=False)
                self.sim.run_sim()
                return
            except:
                print(f'exec_sim: sim error, retry..{i+1}')   
    
    def reset(self):
        if self.sim:
            del self.sim            
        self.node = random.choice(self.nodes)
        self.broadcast_order = self.broadcast_rules["new_blocks"][self.node]     
        self.exec_sim()
        return self.get_observation()
    
    def calc_reward(self):       
        sync_time = np.mean(np.nan_to_num(self.statistics['sync_time'], nan=10, posinf=10, neginf=10)) if not is_empty(self.statistics['sync_time']) else 10
        sync_rate = self.statistics['sync_blocks'] / self.statistics['blocks_count'] if self.statistics['blocks_count'] > 0 else 0
        reward = sync_rate / sync_time
        return reward

    def step(self, action):
        done = True
        self.broadcast_order = itemgetter(*np.argsort(action))(self.broadcast_order)
        self.broadcast_rules["new_blocks"][self.node] = self.broadcast_order
        self.exec_sim()
        obs = self.get_observation()  
        self.statistics.update({'broadcast_order' : self.broadcast_order})
        reward = self.calc_reward()
        self.reward.append(reward)      
        return obs, reward, done, self.statistics

    def render(self, mode='console'):
        if mode != 'console':
          raise NotImplementedError()
        pass

    def get_reward(self):
        return self.reward
    
    def close(self):
        pass
