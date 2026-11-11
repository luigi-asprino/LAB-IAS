import torch
import torch.nn as nn
import torch.nn.functional as F


# ── SimpleNet ────────────────────────────────────────────────────────────
class SimpleNet(nn.Module):
    """
    Rete fully-connected a tre strati per classificazione MNIST.

    Scelta intenzionalmente semplice per questa lezione: l'obiettivo
    è osservare il pattern PS, non ottimizzare l'accuracy del modello.
    Un modello leggero permette di completare più epoch in 45 minuti
    senza GPU, rendendo visibile nei log la dinamica push/pull.

    Architettura:
        Input  → 784  (MNIST flattenato: 28×28 pixel, valori [0,1])
        Layer1 → 256  neuroni, ReLU, Dropout 0.2
        Layer2 → 128  neuroni, ReLU, Dropout 0.2
        Output →  10  neuroni, LogSoftmax (una classe per cifra 0–9)

    Dimensione parametri:
        fc1:  784×256 + 256  =  200.960 parametri
        fc2:  256×128 + 128  =   32.896 parametri
        fc3:  128×10  +  10  =    1.290 parametri
        Totale:                 235.146 parametri  (~920 KB in float32)

    La dimensione ridotta è deliberata: con un modello più grande
    il PS diventerebbe il collo di bottiglia sulla rete, nascondendo
    la dinamica asincrona che vogliamo osservare.
    """

    def __init__(self, dropout_rate: float = 0.2):
        """
        dropout_rate : probabilità di azzerare un neurone durante il training.
                       Valore di default 0.2 — basso, per non rallentare
                       la convergenza su un dataset semplice come MNIST.
        """
        super(SimpleNet, self).__init__()

        # nn.Linear(in, out) definisce uno strato fully-connected.
        # I pesi vengono inizializzati da PyTorch con Kaiming uniform
        # di default — adatto per strati seguiti da ReLU.
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # Dropout: durante il training azzera casualmente il dropout_rate%
        # dei neuroni ad ogni forward pass, riducendo l'overfitting.
        # Durante la valutazione (model.eval()) viene disabilitato
        # automaticamente da PyTorch.
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Definisce il grafo computazionale del forward pass.
        PyTorch costruisce il grafo dinamicamente ad ogni chiamata,
        il che permette di usare qualsiasi struttura Python (if, loop)
        all'interno del forward — utile per architetture più complesse.

        x : tensore di input, shape (batch_size, 784)
            I valori devono essere già normalizzati in [0, 1],
            come prodotto da dataset.py con transforms.ToTensor().
        """

        # Strato 1: proiezione lineare + ReLU + Dropout
        # F.relu è la versione funzionale: equivalente a nn.ReLU()
        # ma non aggiunge un modulo al grafo del modello — preferibile
        # per operazioni senza parametri apprendibili.
        x = self.dropout(F.relu(self.fc1(x)))

        # Strato 2: stessa struttura del primo
        x = self.dropout(F.relu(self.fc2(x)))

        # Strato 3: proiezione finale sulle 10 classi.
        # F.log_softmax restituisce log-probabilità invece di probabilità:
        # più stabile numericamente e compatibile con nn.NLLLoss.
        # In train_ps.py usiamo nn.CrossEntropyLoss che internamente
        # combina log_softmax + NLLLoss, quindi qui si può anche omettere
        # log_softmax e restituire i logit grezzi — scelta esplicitata
        # nel commento sotto.
        x = F.log_softmax(self.fc3(x), dim=1)

        return x

    def get_param_count(self) -> int:
        """
        Utility per stampare il numero totale di parametri apprendibili.
        Utile durante la lezione per collegare la dimensione del modello
        al volume di dati trasferiti ad ogni push/pull verso il PS.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Funzioni di utilità ──────────────────────────────────────────────────

def save_model(model: SimpleNet, path: str) -> None:
    """
    Salva i pesi del modello su disco (locale o GCS mount).

    Salva solo lo state_dict — il dizionario dei pesi — e non
    l'intera struttura del modello. È la pratica raccomandata da PyTorch:
    più portabile tra versioni diverse e più leggero del pickle completo.

    path : percorso di destinazione, es. "/gcs/bucket/models/model.pth"
           oppure un path locale "/tmp/model.pth"
    """
    torch.save(model.state_dict(), path)
    print(f"[model] Pesi salvati in: {path}")


def load_model(path: str, dropout_rate: float = 0.2) -> SimpleNet:
    """
    Carica i pesi da disco e restituisce un'istanza di SimpleNet pronta
    per l'inferenza o per continuare il training.

    map_location="cpu" forza il caricamento su CPU indipendentemente
    da dove il modello era stato salvato (es. GPU). Necessario nei
    container Vertex AI senza GPU per evitare errori di device mismatch.

    path         : percorso del file .pth
    dropout_rate : deve coincidere con quello usato durante il training
    """
    model = SimpleNet(dropout_rate=dropout_rate)
    model.load_state_dict(
        torch.load(path, map_location="cpu")
    )
    # Imposta il modello in modalità valutazione:
    # disabilita Dropout e BatchNorm (se presenti) per l'inferenza.
    model.eval()
    print(f"[model] Pesi caricati da: {path}")
    return model


# ── Smoke test (eseguito solo se il file viene lanciato direttamente) ────
# Utile durante lo sviluppo per verificare che il modello compili
# e che le dimensioni di input/output siano corrette, senza avviare
# il training completo.
if __name__ == "__main__":

    model = SimpleNet()

    print(f"Parametri totali: {model.get_param_count():,}")
    # Output atteso: Parametri totali: 235,146

    # Batch sintetico: 8 campioni, 784 feature ciascuno
    dummy_input = torch.randn(8, 784)
    output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")   # torch.Size([8, 784])
    print(f"Output shape: {output.shape}")         # torch.Size([8, 10])

    # Verifica che le log-probabilità sommino a 0 (exp → probabilità sommano a 1)
    probs = output.exp()
    print(f"Somma probabilità (primo campione): {probs[0].sum().item():.6f}")
    # Output atteso: 1.000000

    # Test save/load round-trip
    save_model(model, "/tmp/test_model.pth")
    model_loaded = load_model("/tmp/test_model.pth")
    print("Round-trip save/load: OK")