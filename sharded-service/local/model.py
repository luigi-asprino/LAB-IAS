# =============================================================================
# model.py — Modello per la Lezione 9: Sharded Service Pattern
# =============================================================================
#
# Questo modulo definisce un Transformer semplificato progettato per essere
# partizionato tra più processi o macchine (shard). Ogni shard istanzia
# solo la porzione del modello che gli compete.
#
# Struttura del modello completo (total_layers=8, d_model=256):
#
#   [Shard 0]  Embedding (vocab → d_model)
#              TransformerBlock 0
#              TransformerBlock 1
#              TransformerBlock 2
#              TransformerBlock 3
#   [Shard 1]  TransformerBlock 4
#              TransformerBlock 5
#              TransformerBlock 6
#              TransformerBlock 7
#              Head di classificazione (d_model → n_classes)
#
# Con N_SHARDS=4 ogni shard ospita 2 layer (total_layers // N_SHARDS = 2).
# L'embedding rimane sempre su shard 0, la head su shard N-1.
#
# Perché questo design?
# ---------------------
# In un modello reale come GPT-2 Large (~1.5B param, ~6 GB in fp32) o
# LLaMA-7B (~28 GB in fp32), i pesi non entrano in una singola GPU.
# Distribuire i layer tra più nodi (layer sharding / pipeline parallelism)
# è la strategia più semplice da implementare: ogni nodo carica solo
# i propri pesi, il tensore di attivazione viaggia in rete da shard a shard.
#
# In questa lezione usiamo un modello leggero (d_model=256, 8 layer) che
# gira su CPU: l'obiettivo è osservare il PATTERN, non ottimizzare la
# performance. Le stesse idee si applicano a modelli da miliardi di parametri.
#
# Dimensioni indicative (d_model=256, 8 layer, vocab=30522):
#   Embedding:          30522 × 256 ≈ 7.8M parametri (~30 MB)
#   Ogni TransformerBlock: ~525K parametri (~2 MB)
#   Head:               256 × 2 = 512 parametri (trascurabile)
#   Totale:             ~12M parametri (~46 MB)
# =============================================================================

import torch
import torch.nn as nn


# =============================================================================
# BLOCCO ELEMENTARE — TransformerBlock
# =============================================================================

class TransformerBlock(nn.Module):
    """
    Blocco Transformer standard: Multi-Head Self-Attention + Feed-Forward.

    È l'unità di partizionamento: ogni shard ne riceve un sottoinsieme
    contiguo. Un blocco con d_model=256, nhead=4, dim_ff=512 ha:

        MultiheadAttention: 4 × (256×64 + 64)×3 proiezioni + output proj
                            ≈ 4 × (256×256 + 256) × 4 = ~525K param

        FeedForward:        256×512 + 512 + 512×256 + 256
                            ≈ 262K param

        LayerNorm (×2):     2 × (256 + 256) = 1K param

    Parametri del blocco: ~526K  (~2 MB in float32)

    Input/output: tensore (batch_size, seq_len, d_model).
    La forma è invariante tra blocchi: è il contratto che rende possibile
    lo sharding — ogni shard può essere aggiunto o rimosso senza modificare
    gli shard adiacenti.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead:   int = 4,
        dim_ff:  int = 512,
        dropout: float = 0.1,
    ):
        """
        d_model : dimensione del vettore di embedding (deve essere divisibile per nhead)
        nhead   : numero di teste dell'attenzione
        dim_ff  : dimensione interna del feed-forward
        dropout : probabilità di dropout (applicato internamente a MHA e FF)
        """
        super().__init__()

        # Multi-Head Self-Attention.
        # batch_first=True: il tensore ha forma (batch, seq, d_model)
        # invece della forma default (seq, batch, d_model) di PyTorch.
        # Scegliamo batch_first=True per coerenza con la serializzazione
        # JSON (batch sulla prima dimensione è più intuitivo).
        self.attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = nhead,
            dropout     = dropout,
            batch_first = True,
        )

        # Feed-Forward Network a due layer con ReLU.
        # La dimensione interna dim_ff > d_model è standard nei Transformer:
        # introduce capacità di apprendimento non lineare tra i layer di attenzione.
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

        # Layer Normalization applicata DOPO residual (post-norm, stile originale).
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        output: (batch_size, seq_len, d_model)  — stessa forma, contratto garantito

        Flusso:
          1. Self-attention: ogni token "guarda" tutti gli altri token della sequenza.
          2. Residual + LayerNorm: x = LayerNorm(x + attn_out)
          3. Feed-forward: trasformazione per-token indipendente.
          4. Residual + LayerNorm: x = LayerNorm(x + ff_out)
        """
        # ── Blocco 1: Self-Attention con residual ───────────────────────────
        # query=key=value=x → self-attention (non cross-attention)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # ── Blocco 2: Feed-Forward con residual ─────────────────────────────
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


# =============================================================================
# MODELLO PARTIZIONABILE — ShardedTransformer
# =============================================================================

class ShardedTransformer(nn.Module):
    """
    Transformer partizionato: ogni istanza ospita solo i layer del proprio shard.

    Il modello viene suddiviso in N_SHARDS partizioni di eguale dimensione.
    Ogni shard riceve in input un tensore e restituisce un tensore di uguale
    forma (tranne l'ultimo shard, che restituisce i logit finali).

    Parametri
    ----------
    total_layers : numero totale di TransformerBlock del modello completo
    shard_id     : indice di questo shard (0-indexed)
    n_shards     : numero totale di shard
    d_model      : dimensione del vettore di embedding
    nhead        : teste dell'attenzione
    dim_ff       : dimensione interna del feed-forward
    vocab_size   : dimensione del vocabolario (usata da shard 0 per l'embedding)
    n_classes    : classi di output (usata dall'ultimo shard per la head)

    Proprietà dello sharding
    ------------------------
    - Shard 0 ha in aggiunta: Embedding (vocab_size → d_model)
    - Shard N-1 ha in aggiunta: Head lineare (d_model → n_classes)
    - Gli shard intermedi ospitano solo TransformerBlock
    - Il numero di layer per shard è total_layers // n_shards
      (assumiamo total_layers divisibile per n_shards)
    """

    def __init__(
        self,
        total_layers: int = 8,
        shard_id:     int = 0,
        n_shards:     int = 2,
        d_model:      int = 256,
        nhead:        int = 4,
        dim_ff:       int = 512,
        vocab_size:   int = 30522,
        n_classes:    int = 2,
    ):
        super().__init__()

        if total_layers % n_shards != 0:
            raise ValueError(
                f"total_layers ({total_layers}) deve essere divisibile per "
                f"n_shards ({n_shards})"
            )

        self.shard_id = shard_id
        self.n_shards = n_shards
        self.d_model  = d_model

        # ── Calcolo dei layer assegnati a questo shard ───────────────────────
        layers_per_shard = total_layers // n_shards
        start_layer      = shard_id * layers_per_shard
        end_layer        = start_layer + layers_per_shard

        # ModuleList registra i blocchi come sotto-moduli PyTorch:
        # necessario perché .parameters() li includa e lo state_dict funzioni.
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, nhead=nhead, dim_ff=dim_ff)
            for _ in range(layers_per_shard)
        ])

        # ── Componenti specifiche dello shard 0 (ingresso della pipeline) ───
        if shard_id == 0:
            # Embedding: mappa token ID interi → vettori float di dim d_model.
            # Questo è il "gate d'ingresso" dell'intera pipeline.
            # In un modello reale, i pesi di embedding sono inizializzati con
            # pesi pre-addestrati (es. da BERT o GPT); qui li inizializziamo
            # casualmente per scopi didattici.
            self.embedding = nn.Embedding(vocab_size, d_model)

        # ── Componenti specifiche dell'ultimo shard (uscita della pipeline) ─
        if shard_id == n_shards - 1:
            # Head di classificazione: estrae il vettore del [CLS] token
            # (primo token della sequenza) e lo proietta sullo spazio delle classi.
            # In BERT e modelli simili, il [CLS] token aggrega informazioni
            # sull'intera sequenza tramite l'attenzione.
            self.head = nn.Linear(d_model, n_classes)

        # Informazioni di diagnostica
        self._layers_range = (start_layer, end_layer)
        self._layers_per_shard = layers_per_shard

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward del singolo shard.

        Se shard_id == 0:        x è (batch, seq_len) di token ID interi
                                  → dopo embedding diventa (batch, seq_len, d_model)
        Se shard_id intermedio:  x è (batch, seq_len, d_model) di float
        Se shard_id == n-1:      restituisce (batch, n_classes) — logit finali

        Questa asimmetria di tipo (input) è un dettaglio implementativo:
        il shard_server.py gestisce la conversione di dtype prima di chiamare forward.
        """
        # ── Shard 0: converte token ID in vettori di embedding ───────────────
        if self.shard_id == 0:
            # x.long() garantisce dtype corretto per nn.Embedding
            x = self.embedding(x.long())

        # ── Tutti gli shard: passa il tensore attraverso i propri TransformerBlock
        for block in self.blocks:
            x = block(x)

        # ── Ultimo shard: classificazione sul [CLS] token ────────────────────
        if self.shard_id == self.n_shards - 1:
            # x[:, 0, :] → estrae il primo token di ogni sequenza nel batch
            # Forma: (batch_size, d_model) → (batch_size, n_classes)
            x = self.head(x[:, 0, :])

        return x

    def param_count(self) -> int:
        """Numero totale di parametri di questo shard."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_bytes(self) -> int:
        """Dimensione in byte dei parametri di questo shard (float32 = 4 byte/param)."""
        return sum(
            p.numel() * p.element_size()
            for p in self.parameters()
            if p.requires_grad
        )

    def activation_bytes(self, batch_size: int = 1, seq_len: int = 16) -> int:
        """
        Dimensione stimata in byte del tensore di attivazione che viaggia
        in rete tra questo shard e il successivo.

        Il tensore di attivazione ha forma (batch_size, seq_len, d_model)
        e dtype float32 (4 byte per elemento).

        Questo è il COSTO DI COMUNICAZIONE di ogni richiesta di inferenza:
        più è grande, più incide sulla latenza di rete tra shard.
        """
        return batch_size * seq_len * self.d_model * 4  # 4 byte per float32

    def info(self) -> str:
        """Stringa di diagnostica per il log di avvio."""
        has_emb  = hasattr(self, "embedding")
        has_head = hasattr(self, "head")
        return (
            f"ShardedTransformer("
            f"shard={self.shard_id}/{self.n_shards}, "
            f"layer={self._layers_range[0]}-{self._layers_range[1]-1}, "
            f"embedding={'sì' if has_emb else 'no'}, "
            f"head={'sì' if has_head else 'no'}, "
            f"param={self.param_count():,}, "
            f"size={self.param_bytes()/1024/1024:.1f} MB)"
        )


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=== Test ShardedTransformer ===\n")

    TOTAL_LAYERS = 8
    N_SHARDS     = 2
    BATCH        = 2
    SEQ_LEN      = 16
    D_MODEL      = 256

    # Simula il flusso completo attraverso 2 shard
    print("Flusso completo (N_SHARDS=2):")
    x = torch.randint(0, 30522, (BATCH, SEQ_LEN))  # token ID interi

    for sid in range(N_SHARDS):
        shard = ShardedTransformer(
            total_layers=TOTAL_LAYERS,
            shard_id=sid,
            n_shards=N_SHARDS,
            d_model=D_MODEL,
        )
        shard.eval()

        with torch.no_grad():
            x = shard(x)

        act_kb = shard.activation_bytes(BATCH, SEQ_LEN) / 1024
        print(f"  Shard {sid}: {shard.info()}")
        print(f"           → output shape: {x.shape}, attivazione: {act_kb:.0f} KB")

    print(f"\nOutput finale (logit): {x}")
    print("\nSmoke test: OK")
