## Sustainable broadcasting in Blockchain Network with Reinforcement Learning

## Working paper

Valko, D., & Kudenko, D. (2024). Sustainable broadcasting in Blockchain Network with Reinforcement Learning. arXiv. https://doi.org/10.48550/arXiv.2407.15616

```sh
@misc{ValkoKudenko2022,
title={Sustainable broadcasting in Blockchain Network with Reinforcement Learning}, 
author={Danila Valko and Daniel Kudenko},
year={2024},
publisher={arXiv},
howpublished={arXiv},
doi = {https://doi.org/10.48550/arXiv.2407.15616},
}
```

### Setup
BlockSim requires some virtual environment with certain dependencies, see `requirements.txt`.
```sh
conda create -n blocksim python=3.9
conda activate blocksim
pip install -r requirements.txt 
```

### Run
* test simulator
```sh
python -m blocksim.blocksim
```
* run training
```sh
source activate blocksim && python train.py --n_envs 4
```
* run experiments
```sh
python exp.py --k 1000
```

## References

* We used BlockSim - a framework for modeling and simulating blockchain protocols [(Faria & Correia, 2019)](https://static.carlosfaria.pt/file/personal-assets/papers/blocksim-blockchain-simulator.pdf), see [GitHub](https://github.com/carlosfaria94/blocksim).

