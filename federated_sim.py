import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


class CreditRiskMLP(nn.Module):
    """Lightweight neural classifier for tabular credit-risk data."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_non_iid_clients(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int = 5,
    seed: int = 42,
):
    """Partition data into heterogeneous client silos."""
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")

    rng = np.random.default_rng(seed)
    sorted_idx = np.argsort(y + rng.normal(0, 0.35, size=len(y)))
    splits = np.array_split(sorted_idx, num_clients)

    client_loaders = []
    for idx in splits:
        dataset = TensorDataset(
            torch.tensor(X[idx], dtype=torch.float32),
            torch.tensor(y[idx], dtype=torch.float32).unsqueeze(1),
        )
        client_loaders.append(DataLoader(dataset, batch_size=32, shuffle=True))
    return client_loaders


def local_train(
    model: nn.Module,
    loader: DataLoader,
    epochs: int = 2,
    lr: float = 0.01,
) -> dict:
    """Train a local copy without sharing raw records with the coordinator."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

    return copy.deepcopy(model.state_dict())


def federated_averaging(
    global_model: nn.Module,
    client_weights: list,
    client_sizes: list,
) -> None:
    """Aggregate local updates using sample-weighted FedAvg."""
    if not client_weights:
        raise ValueError("client_weights cannot be empty")
    if len(client_weights) != len(client_sizes):
        raise ValueError("Each client weight must have a corresponding client size")

    total_samples = sum(client_sizes)
    if total_samples <= 0:
        raise ValueError("Total client sample size must be positive")

    new_state = copy.deepcopy(global_model.state_dict())
    for key in new_state:
        new_state[key] = sum(
            client_weights[i][key] * (client_sizes[i] / total_samples)
            for i in range(len(client_weights))
        )
    global_model.load_state_dict(new_state)


def evaluate(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray):
    """Compute global accuracy and ROC-AUC."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    if probs.ndim == 0:
        probs = probs.reshape(1)

    return accuracy_score(y_test, preds), roc_auc_score(y_test, probs)


def run_simulation(
    num_clients: int = 5,
    rounds: int = 15,
    participation_rate: float = 0.8,
) -> None:
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if not 0 < participation_rate <= 1:
        raise ValueError("participation_rate must be greater than 0 and at most 1")

    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    X, y = make_classification(
        n_samples=4000,
        n_features=14,
        n_informative=9,
        n_redundant=3,
        weights=[0.7, 0.3],
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    client_loaders = create_non_iid_clients(
        X_train,
        y_train,
        num_clients=num_clients,
    )
    global_model = CreditRiskMLP(input_dim=X.shape[1])

    print(
        f"Starting FedAvg Simulation | Clients: {num_clients} "
        f"| Participation: {int(participation_rate * 100)}%\n"
    )

    for round_number in range(1, rounds + 1):
        selected_count = max(1, int(num_clients * participation_rate))
        selected = rng.choice(
            num_clients,
            size=selected_count,
            replace=False,
        )

        local_weights = []
        local_sizes = []
        for client_id in selected:
            local_model = copy.deepcopy(global_model)
            weights = local_train(
                local_model,
                client_loaders[int(client_id)],
                epochs=2,
            )
            local_weights.append(weights)
            local_sizes.append(
                len(client_loaders[int(client_id)].dataset)
            )

        federated_averaging(
            global_model,
            local_weights,
            local_sizes,
        )

        if round_number in {1, 5, 10, rounds}:
            acc, auc = evaluate(global_model, X_test, y_test)
            print(
                f"Round {round_number:02d}/{rounds:02d} "
                f"| Global Accuracy: {acc:.4f} "
                f"| ROC-AUC: {auc:.4f}"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run federated credit-risk simulation"
    )
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--participation", type=float, default=0.8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(args.clients, args.rounds, args.participation)
