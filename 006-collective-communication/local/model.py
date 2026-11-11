# =============================================================================
# model.py — Modello per la Lezione 7: Collective Communication Pattern
# =============================================================================
#
# Il modello è intenzionalmente semplice — un MLP a tre strati — per le
# stesse ragioni della Lezione 6: l'obiettivo è osservare il pattern DDP
# e la comunicazione collettiva, non ottimizzare l'accuracy.
#
# Differenze rispetto a SimpleNet (Lezione 6):
#   - Input configurabile (128 feature sintetiche invece di 784 MNIST)
#   - Stesso schema architetturale per continuità didattica
#   - Aggiunto metodo param_bytes() per quantificare il volume
#     di dati trasferiti nell'all-reduce (rilevante per il benchmark)
#
# Durante il training DDP, PyTorch chiama all-reduce sui gradienti di
# OGNI parametro del modello dopo ogni backward pass. Il volume totale
# trasmesso è proporzionale a param_bytes(): più è grande il modello,
# più la comunicazione incide su T_comm e quindi sulla scaling efficiency.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Rete fully-connected a tre strati per classificazione su dataset sintetico
    o MNIST. Usata come modello di riferimento per gli esercizi DDP.

    Architettura con in_features=128:
        Input  → 128
        Layer1 → 256  (Linear + ReLU + Dropout 0.2)
        Layer2 → 256  (Linear + ReLU + Dropout 0.2)
        Output →   2  (Linear → CrossEntropyLoss gestisce il softmax)

    Numero di parametri con in_features=128, n_classes=2:
        fc1:  128×256 + 256  =  33.024
        fc2:  256×256 + 256  =  65.792
        fc3:  256×2   + 2    =     514
        Totale:                 99.330  (~388 KB in float32)

    Il modello è abbastanza grande da rendere visibile l'overhead dell'all-reduce
    ma abbastanza piccolo da convergere in poche epoch su CPU.
    """

    def __init__(
        self,
        in_features:  int   = 128,
        hidden_size:  int   = 256,
        n_classes:    int   = 2,
        dropout_rate: float = 0.2,
    ):
        """
        in_features  : dimensione dell'input (128 per sintetico, 784 per MNIST)
        hidden_size  : neuroni negli strati nascosti
        n_classes    : classi di output (2 per sintetico, 10 per MNIST)
        dropout_rate : probabilità di azzeramento Dropout
        """
        super().__init__()

        self.fc1     = nn.Linear(in_features, hidden_size)
        self.fc2     = nn.Linear(hidden_size, hidden_size)
        self.fc3     = nn.Linear(hidden_size, n_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Salva le dimensioni per i log e per param_bytes()
        self.in_features = in_features
        self.n_classes   = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        x : (batch_size, in_features)
        Returns: logit grezzi (batch_size, n_classes)

        Non applichiamo log_softmax qui perché nn.CrossEntropyLoss
        include internamente log_softmax + NLLLoss: più efficiente
        e numericamente più stabile della composizione separata.
        """
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def param_count(self) -> int:
        """Numero totale di parametri apprendibili."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_bytes(self) -> int:
        """
        Volume in byte dei parametri (= volume dell'all-reduce per iterazione).

        Con float32 ogni parametro occupa 4 byte. Il ring-based all-reduce
        trasferisce 2 * (N-1)/N * param_bytes() byte per processo.
        Questo metodo permette di stimare l'overhead di comunicazione prima
        di lanciare il benchmark.
        """
        return sum(
            p.numel() * p.element_size()
            for p in self.parameters()
            if p.requires_grad
        )


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    model = MLP(in_features=128, n_classes=2)

    print(f"Parametri totali : {model.param_count():,}")
    print(f"Dimensione model : {model.param_bytes() / 1024:.1f} KB")
    print(f"Volume all-reduce: ~{model.param_bytes() * 2 / 1024:.1f} KB per processo (N=2)")

    dummy = torch.randn(16, 128)
    out   = model(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print("Smoke test: OK")
