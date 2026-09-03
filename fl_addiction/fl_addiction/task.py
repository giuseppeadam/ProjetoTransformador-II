from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# =========================
# Features / target
# =========================
FEATURE_COLS = [
    "daily_screen_time_hours",
    "weekend_screen_time",
    "social_media_hours_prod",
    "youtube_hours",
    "gaming_hours_phone",
]
TARGET_COL = "addicted_label"

CSV_PATH = Path(
    r"C:\Users\Giuseppe\estudo\proj-transf\ProjetoTransformador-II"
    r"\dataset_novo_indice.csv - dataset_novo_indice.csv.csv"
)


# =========================
# Modelo (MLP)
# =========================
class Net(nn.Module):
    def __init__(self, num_features=len(FEATURE_COLS)):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# Carregar / particionar dados
# =========================
_X_train_full = None
_y_train_full = None
_X_test = None
_y_test = None


def _load_and_split():
    global _X_train_full, _y_train_full, _X_test, _y_test

    if _X_train_full is not None:
        return

    df = pd.read_csv(CSV_PATH)

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # normaliza usando so o treino
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    _X_train_full, _y_train_full = X_train, y_train
    _X_test, _y_test = X_test, y_test


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    _load_and_split()

    # particiona o treino em pedacos iguais (cada pedaco = uma instituicao)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(_X_train_full))
    partes = np.array_split(idx, num_partitions)
    parte = partes[partition_id]

    X_part = _X_train_full[parte]
    y_part = _y_train_full[parte]

    strat = y_part if len(np.unique(y_part)) > 1 else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_part, y_part, test_size=0.2, random_state=42, stratify=strat
    )

    trainloader = DataLoader(TabularDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    valloader = DataLoader(TabularDataset(X_val, y_val), batch_size=batch_size)
    return trainloader, valloader


def load_centralized_dataset(batch_size: int = 128):
    _load_and_split()
    return DataLoader(TabularDataset(_X_test, _y_test), batch_size=batch_size)


# =========================
# Treino / teste
# =========================
def train(net, trainloader, epochs, lr, device):
    net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for X_batch, y_batch in trainloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(net(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return running_loss / (epochs * len(trainloader))


def test(net, testloader, device):
    net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for X_batch, y_batch in testloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = net(X_batch)
            loss += criterion(logits, y_batch).item()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
