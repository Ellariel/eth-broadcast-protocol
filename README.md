## Sustainable broadcasting in blockchain networks with reinforcement learning

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

## Published paper

Valko, D., & Kudenko, D. (2025). Sustainable broadcasting in blockchain networks with reinforcement learning. BUIS-Tage 2025 – Smarte Und Nachhaltige Infrastrukturen, 237–244. https://doi.org/10.2370/9783819104107

```sh
@inproceedings{ValkoKudenko2025,
author = {Danila Valko and Daniel Kudenko},
title = {Sustainable broadcasting in blockchain networks with reinforcement learning},
year = {2025},
pages = {237-244},
booktitle = {BUIS-Tage 2025 – Smarte und Nachhaltige Infrastrukturen},
isbn = {978-3-8191-0410-7},
howpublished = {Shaker Verlag GmbH},
doi = {10.2370/9783819104107},
}
```

## Preprint

Valko, D., & Kudenko, D. (2024). Sustainable broadcasting in Blockchain Network with Reinforcement Learning. arXiv. https://doi.org/10.48550/arXiv.2407.15616

```sh
@misc{ValkoKudenko2024,
title={Sustainable broadcasting in blockchain networks with reinforcement learning}, 
author={Danila Valko and Daniel Kudenko},
year={2024},
publisher={arXiv},
howpublished={arXiv},
doi = {https://doi.org/10.48550/arXiv.2407.15616},
}
```

## References

* We used BlockSim - a framework for modeling and simulating blockchain protocols [(Faria & Correia, 2019)](https://static.carlosfaria.pt/file/personal-assets/papers/blocksim-blockchain-simulator.pdf), see [GitHub](https://github.com/carlosfaria94/blocksim).

