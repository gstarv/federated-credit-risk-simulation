import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from federated_sim import (
    CreditRiskMLP,
    create_non_iid_clients,
    federated_averaging,
)


def test_credit_risk_model_output_shape():
    model = CreditRiskMLP(input_dim=14)
    output = model(torch.zeros(3, 14))
    assert output.shape == (3, 1)


def test_non_iid_partition_has_all_samples():
    X = np.zeros((20, 14), dtype=float)
    y = np.array([0] * 10 + [1] * 10)
    loaders = create_non_iid_clients(X, y, num_clients=5, seed=42)
    assert len(loaders) == 5
    assert sum(len(loader.dataset) for loader in loaders) == 20


def test_federated_averaging_rejects_empty_updates():
    model = CreditRiskMLP(input_dim=14)
    try:
        federated_averaging(model, [], [])
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
