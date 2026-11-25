# Privacy-Preserving Federated Learning for Decentralized Credit Risk

A PyTorch simulation of Federated Averaging (FedAvg) across heterogeneous, non-IID financial data silos.

## Features

- Non-IID partitioning of synthetic credit-risk data
- Configurable participating clients and communication rounds
- Local model training that keeps client records in memory locally
- Sample-weighted FedAvg aggregation
- Global accuracy and ROC-AUC reporting
- Automated tests

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python federated_sim.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Example

```bash
python federated_sim.py --clients 5 --rounds 15 --participation 0.8
```

## Project structure

- `federated_sim.py` - simulation implementation
- `tests/test_federated_sim.py` - unit tests
- `requirements.txt` - runtime dependencies

## Scope

This is an educational simulation. It does not implement secure aggregation, encrypted model updates, or differential privacy.
