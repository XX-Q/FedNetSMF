# FedNetSMF
## Installation
```bash
pip install -r requirements.txt
```
## Quick Start
### Data Preparation
```bash
python data_utils.py
```
This script helps you quickly partition data, enabling you to simulate experiments with different parties under subgraph scenarios. You need to prepare the original format of the dataset in the corresponding directory under `./datasets/`.
### Run FedNetMF
```bash
python fed_netmf_block.py
```
### Run FedNetSMF
```bash
python fed_netsmf_sparse_multi_party.py
```