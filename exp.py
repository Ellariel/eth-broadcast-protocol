import os, gc, pickle, argparse
from tqdm import tqdm

from blocksim.blocksim import Blocksim
from utils import get_random_seed, set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument('--type', default='ethereum', type=str)
parser.add_argument('--duration', default=60, type=int)
parser.add_argument('--k', default=300, type=int)
parser.add_argument('--test', default='PPO-env-1.sav.0.416', type=str) #'PPO-env-1.sav.0.442'
args = parser.parse_args()

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

def simulation(type, duration, base_dict, randomize_connections, seed, use_agent):
    sim = Blocksim(type=type, duration=duration, 
                   nodes_dict=base_dict, 
                   randomize_connections=randomize_connections, 
                   seed=seed, 
                   use_agent=use_agent)
    sim.run_sim()
    return [sim.get_statistics()]

base_dir = './'
results_dir = os.path.join(base_dir, 'results')
weights_dir = os.path.join(base_dir, 'weights')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

duration = args.duration
type = args.type
k = args.k
test = args.test
id = test.split('.')[-1]

print(f"k: {k}, type: {type}, duration: {duration}, test: '{test}', id: {id}")

model_results = []
random_sameseed_results = []
set_random_seed()
for i in tqdm(range(k), total=k):
        seed = get_random_seed()
        model_results += simulation(type=type, duration=duration, base_dict=base_dict, randomize_connections=True, 
                                    seed=seed, use_agent=os.path.join(weights_dir, test))
        random_sameseed_results += simulation(type=type, duration=duration, base_dict=base_dict, randomize_connections=True, 
                                    seed=seed, use_agent=False)
        gc.collect() 
                  
with open(os.path.join(results_dir, f"k_results-{id}-{type}-{duration}-{k}.pickle"), 'wb') as f:
    pickle.dump({'model' : model_results,
                 'random_sameseed' : random_sameseed_results}, f)          
        
