import os, time, random, pickle, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from stable_baselines3.common.env_util import make_vec_env

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='env', type=str)
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--timesteps', default=1e4, type=int)
parser.add_argument('--attempts', default=1000, type=int)
parser.add_argument('--approach', default='PPO', type=str)
parser.add_argument('--n_envs', default=16, type=int)
args = parser.parse_args()

timesteps = args.timesteps
approach = args.approach
epochs = args.epochs
attempts = args.attempts
n_envs = args.n_envs

if args.env == 'env':
    version='env'
    from env import BlocksimEnv

base_dir = './'
snapshots_dir = os.path.join(base_dir, 'snapshots')
weights_dir = os.path.join(base_dir, 'weights')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

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

file_mask = f'{approach}-{version}-{n_envs}'

for a in range(attempts):
    E = make_vec_env(lambda: BlocksimEnv(nodes_dict=base_dict), n_envs=n_envs)

    lf = os.path.join(results_dir, f'{file_mask}.log')
    log = pd.read_csv(lf, sep=';', compression='zip') if os.path.exists(lf) else None
    f = os.path.join(weights_dir, f'{file_mask}.sav')

    if approach == 'PPO':
        model_class = PPO
    else:
        model_class = None
        print(f'{approach} - not implemented!')
        raise ValueError

    if os.path.exists(f) and model_class:
        model = model_class.load(f, E, force_reset=False, verbose=0)
        print(f'model is loaded {approach}: {f}')
    else:
        print(f'did not find {approach}: {f}')
        model = model_class("MlpPolicy", E, verbose=0) 

    for epoch in range(1, epochs + 1):
        model.learn(total_timesteps=timesteps, progress_bar=True)

        reward = E.env_method('get_reward')
        mean_reward = np.mean(reward, axis=1)
        max_mean_reward = np.max(mean_reward)
        print(f"max mean reward: {max_mean_reward:.3f}~{mean_reward}")
        print(f'n_envs: {n_envs}, epoch: {epoch}/{epochs}, v: {version}, a: {a}')
        model.save(f)
        model.save(f + f'.{max_mean_reward:.3f}')
        print('saved:', f + f'.{max_mean_reward:.3f}')

        log = pd.concat([log, pd.DataFrame.from_dict({'time' : time.time(),
                                                'approach' : approach,
                                                'max_mean_reward' : max_mean_reward,
                                                'mean_reward' : mean_reward,
                                                'epoch' : epoch,
                                                'epochs' : epochs,
                                                'version' : version,
                                                'n_envs' : n_envs,
                                                'total_timesteps' : timesteps,
                                                'filename' : f,
                                                }, orient='index').T], ignore_index=True)
        log.to_csv(lf, sep=';', index=False, compression='zip')