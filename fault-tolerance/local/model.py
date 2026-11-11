# =============================================================================
# model.py — Modello per la Lezione 8: Elasticity e Fault-Tolerance
# =============================================================================
#
# Identico alla Lezione 7 per continuità didattica: l'obiettivo di questa
# lezione è il meccanismo di checkpoint e la gestione della preemption,
# non l'architettura del modello.
#
# Il metodo param_bytes() è utile per stimare il volume del checkpoint:
# un file .pt salvato con torch.save(model.state_dict()) occupa circa
# param_bytes() byte (float32, nessuna compressione).
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Rete fully-connected a tre strati per classificazione binaria.

    Architettura con in_features=128:
        Input  → 128
        Layer1 → 256  (Linear + ReLU + Dropout 0.2)
        Layer2 → 256  (Linear + ReLU + Dropout 0.2)
        Output →   2  (Linear → CrossEntropyLoss gestisce softmax)

    Parametri totali: ~99.330  (~388 KB in float32)
    """

    def __init__(
        self,
        in_features:  int   = 128,
        hidden_size:  int   = 256,
        n_classes:    int   = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.fc1     = nn.Linear(in_features, hidden_size)
        self.fc2     = nn.Linear(hidden_size, hidden_size)
        self.fc3     = nn.Linear(hidden_size, n_classes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.in_features = in_features
        self.n_classes   = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_bytes(self) -> int:
        """Volume in byte del checkpoint (state_dict in float32)."""
        return sum(
            p.numel() * p.element_size()
            for p in self.parameters()
            if p.requires_grad
        )


if __name__ == "__main__":
    model = MLP()
    print(f"Parametri      : {model.param_count():,}")
    print(f"Dimensione ckpt: ~{model.param_bytes() / 1024:.1f} KB")
    dummy = torch.randn(16, 128)
    print(f"Output shape   : {model(dummy).shape}")
    print("Smoke test: OK")
