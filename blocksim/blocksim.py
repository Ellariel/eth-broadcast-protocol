import os, time, random, json
#import pandas as pd
import numpy as np
from json import dumps as dump_json
from blocksim.world import SimulationWorld
from blocksim.node_factory import NodeFactory
from blocksim.transaction_factory import TransactionFactory
from blocksim.models.network import Network

from agent import Agent

from utils import get_random_seed, is_empty, set_random_seed

class Blocksim():
    def __init__(self, type='ethereum', duration=60, seed=13, verbose=False, nodes_dict=None, connections=None, broadcast_rules=None, randomize_connections=True, use_agent=False):
        self.seed = seed
        self.now = int(time.time())  # current time
        self.duration = duration  # seconds
        self.verbose = verbose
        self.type = type
        self.use_agent = use_agent
        self.nodes_dict = nodes_dict
        self.connections = connections
        self.broadcast_rules = broadcast_rules
        self.randomize_connections = randomize_connections
        
        set_random_seed(self.seed)

        self.world = SimulationWorld(
            self.duration,
            self.now,
            f'input-parameters/{self.type}_config.json',
            'input-parameters/latency.json',
            'input-parameters/throughput-received.json',
            'input-parameters/throughput-sent.json',
            'input-parameters/delays.json')
            
        self.world.env.data['msg_counter'] = 0
        self.world.env.data['verbose'] = self.verbose
        self.world.env.data['randomize_connections'] = self.randomize_connections
        self.network = Network(self.world.env, 'NetworkXPTO')
        self.world.env.data['use_agent'] = Agent(self.world.env, weights=self.use_agent) if self.use_agent else False
        
        if 'block_propagation_final' not in self.world.env.data:
                    self.world.env.data['block_propagation_final'] = {}
        if 'created_blocks' not in self.world.env.data:
                    self.world.env.data['created_blocks'] = []       

        node_factory = NodeFactory(self.world, self.network)
        self.nodes_list = node_factory.create_nodes(self.nodes_dict['miners'], self.nodes_dict['non_miners'])
        self.world.env.data['nodes_list'] = [i.address for i in self.nodes_list]
        self.miners = [i.address for i in self.nodes_list if i.is_mining]
        
        if not self.broadcast_rules:
            self.broadcast_rules = {"new_blocks": {}}
            for i in self.miners:
                self.broadcast_rules['new_blocks'][i] = self.world.env.data['nodes_list'].copy()
                self.broadcast_rules['new_blocks'][i].remove(i)
        
        self.world.env.data['broadcast_rules'] = self.broadcast_rules
     
        self._connect_nodes(connections=self.connections, randomized=False)

        self.world.env.process(self.network.start_heartbeat())
        transaction_factory = TransactionFactory(self.world)
        transaction_factory.broadcast(1, 10, 15, self.nodes_list)

    def get_node(self, address):
        for n in self.nodes_list:
            if n.address == address:
                return n
            
    def get_uv_tx_propagation_time(self, u, v):
        if f'{u}_{v}' in self.world.env.data['tx_propagation'] and not is_empty(self.world.env.data['tx_propagation'][f'{u}_{v}']):
            return list(self.world.env.data['tx_propagation'][f'{u}_{v}'].values())
        else:
            return list(self.world.env.data['tx_propagation'][f'{v}_{u}'].values())
    
    def get_uv_average_tx_propagation_time(self, u, v):
        times = self.get_uv_tx_propagation_time(u, v)
        if not is_empty(times):
            return np.mean(times)

    def get_neighbours_tx_propagation_time(self, u):
        node = self.get_node(u)
        neighbours = [k for k, v in node.active_sessions.items()]
        propagation_time = {v : self.get_uv_average_tx_propagation_time(u, v) for v in neighbours}
        return propagation_time

    def write_report(self, report_file):
            path = f'output/{report_file}.json'
            if not os.path.exists(path):
                print(path)
                with open(path, 'w') as f:
                    f.write(dump_json(self.world.env.data))

    def report_node_chain(self):
        for node in self.nodes_list:
            head = node.chain.head
            chain_list = []
            num_blocks = 0
            for i in range(head.header.number):
                b = node.chain.get_block_by_number(i)
                chain_list.append(str(b.header))
                num_blocks += 1
            chain_list.append(str(head.header))
            key = f'{node.address}_chain'
            self.world.env.data[key] = {
                'head_block_hash': f'{head.header.hash[:8]} #{head.header.number}',
                'number_of_blocks': num_blocks,
                'chain_list': chain_list
            }

    def _connect_nodes(self, connections=None, exeptions=None, randomized=True):
        for i in self.nodes_list:
            if connections and i.address in connections:
                node_list = connections[i.address]
            else:
                node_list = [n.address for n in self.nodes_list]

            if not exeptions:
                connection_list = [n for n in self.nodes_list if n.address in node_list]
            else:
                if i.address in exeptions:
                    if isinstance(exeptions[i.address], list):
                        connection_list = [n for n in self.nodes_list if n.address not in exeptions[i.address] and n.address in node_list]
                    else:
                        connection_list = [n for n in self.nodes_list if exeptions[i.address] not in n.address and n.address in node_list]
                else:
                    exclusion = []
                    for k, v in exeptions.items():
                        if isinstance(v, list) and i.address in v:
                            exclusion.append(k)
                        else:
                            if v in i.address:
                                exclusion.append(k) 
                    connection_list = [n for n in self.nodes_list if n.address not in exclusion and n.address in node_list]

            if len(connection_list):
                if randomized:
                    random.shuffle(connection_list)
                i.connect(connection_list)

    def get_statistics(self, data=None):
        if data:
            if isinstance(data, str):
                with open(data, 'r') as f:
                    data = json.load(f)
        else:
            data = self.world.env.data
            
        n = len(data['nodes_list'])
        r = {'n': n,
                'msg_count': data['msg_counter'],
                'blocks_count': len(data['created_blocks']),
                'forks_count': np.sum([data[i] for i in data.keys() if 'forks' in i]),
                'sync_time': [v['final_time'] - v['initial_time'] for k, v in data['block_propagation_final'].items() if v['node_counter'] >= n // 2],
                'sync_rate': [v['node_counter'] / n for k, v in data['block_propagation_final'].items()],
                }
        r.update({'sync_blocks': len(r['sync_time'])})     
        return r

    def run_sim(self, report_to_file=None):
        self.world.start_simulation()
        self.report_node_chain()
        if report_to_file:
            self.write_report(report_to_file)
        else:
            return self.world.env.data
    
    def __del__(self):
        self.world._env.data.clear()
        self.world._env = None
        del self.world
        del self.nodes_list
        del self.network
   
# test BlockSim   
if __name__ == '__main__':

    base_dict = {
        "miners": {
            "Ohio": {
                "how_many": 25,
                "mega_hashrate_range": "(20, 40)"
            },
            "Tokyo": {
                "how_many": 25,
                "mega_hashrate_range": "(20, 40)"
            },
            "Ireland": {
                "how_many": 25,
                "mega_hashrate_range": "(20, 40)"
            }
        },
        "non_miners": {
            "Ohio": {
                "how_many": 25
            },
            "Tokyo": {
                "how_many": 25
            },
            "Ireland": {
                "how_many": 25
            }
        }
    }

    duration = 60
    type = 'ethereum'
    start_time = time.time()
    seed = 311 #get_random_seed()    
    sim = Blocksim(type=type,
                    duration=duration, 
                    nodes_dict=base_dict, 
                    seed=seed, 
                    verbose=False, 
                    randomize_connections=True,
                    use_agent=False)
        
    sim.run_sim()

    print(f"seed: {seed}, execution time: {time.time() - start_time}\nstatistics:", sim.get_statistics())