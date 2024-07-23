import os
import numpy as np
from stable_baselines3 import PPO
from operator import itemgetter
from utils import is_empty 

class Agent():
    def __init__(self, env, weights='./weights/*.sav', approach='PPO', verbose=False):
        self.env = env
        self.model = None
        self.verbose = verbose
        self.approach = approach
        self.weights = weights

        if approach == 'PPO':
                model_class = PPO
        else:
                model_class = None
                print(f'{approach} - not implemented!')
                raise ValueError

        if os.path.exists(self.weights) and model_class:
                self.model = model_class.load(self.weights, force_reset=False, verbose=self.verbose)
                if self.verbose:
                    print(f'model is loaded {approach}: {self.weights}')
        else:
                print(f'did not find {approach}: {self.weights}')
                raise ValueError

    def get_uv_tx_propagation_time(self, u, v):
        if f'{u}_{v}' in self.env.data['tx_propagation'] and not is_empty(self.env.data['tx_propagation'][f'{u}_{v}']):
            return list(self.env.data['tx_propagation'][f'{u}_{v}'].values())
        else:
            return list(self.env.data['tx_propagation'][f'{v}_{u}'].values())
    
    def get_uv_average_tx_propagation_time(self, u, v):
        times = self.get_uv_tx_propagation_time(u, v)
        if not is_empty(times):
            return np.mean(times)

    def get_neighbours_tx_propagation_time(self, node):
        neighbours = [k for k, v in node.active_sessions.items()]
        propagation_time = {v : self.get_uv_average_tx_propagation_time(node.address, v) for v in neighbours}
        return propagation_time
   
    def get_observation(self, node, broadcast_order):
        propagation_times = self.get_neighbours_tx_propagation_time(node)      
        obs = [propagation_times[n] for n in broadcast_order]
        obs = np.array(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=10, posinf=10, neginf=10)
        return obs

    def predict(self, node, broadcast_order):
        obs = self.get_observation(node, broadcast_order)
        action, _ = self.model.predict(obs, deterministic=True)
        _broadcast_order = itemgetter(*np.argsort(action))(broadcast_order)
        if self.verbose:
            print(f'{node.address}: base order: {broadcast_order[:5]}...')
            print(f'{node.address}: predicted order: {_broadcast_order[:5]}...')
        return _broadcast_order

    def __del__(self):
        del self.model
        del self.env