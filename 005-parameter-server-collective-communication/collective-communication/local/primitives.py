# =============================================================================
# primitives.py — Le primitive collettive di torch.distributed, una per volta
# =============================================================================
#
# Lancio (CPU, backend gloo, nessun costo):
#   torchrun --nproc_per_node=4 primitives.py
#
# Ogni rank parte da un tensore diverso e stampa cosa possiede DOPO ogni
# primitiva. Nell'ultima parte implementiamo a mano il ring all-reduce
# (reduce-scatter + all-gather) con sole send/recv punto-punto, e lo
# confrontiamo con dist.all_reduce.
# =============================================================================

import os
import time

import torch
import torch.distributed as dist


def show(title, rank, world, t):
    """Stampa ordinata: ogni rank scrive a turno, separato da barriere."""
    dist.barrier()
    if rank == 0:
        print(f"\n── {title} " + "─" * (60 - len(title)))
    for r in range(world):
        dist.barrier()
        if r == rank:
            print(f"  rank {rank}: {t.tolist()}", flush=True)
            time.sleep(0.02)
    dist.barrier()


def ring_all_reduce(t: torch.Tensor) -> torch.Tensor:
    """
    Ring all-reduce (SUM) con sole operazioni punto-punto.

    Il tensore è diviso in N chunk. In 2·(N-1) passi ogni rank invia un chunk
    al vicino di destra e ne riceve uno dal vicino di sinistra:
      1. reduce-scatter (N-1 passi): alla fine il rank r possiede la somma
         completa del chunk (r+1) mod N;
      2. all-gather     (N-1 passi): i chunk completi fanno il giro dell'anello.
    Ogni rank invia 2·(N-1)/N · |t| elementi, indipendentemente da N:
    per questo l'algoritmo è bandwidth-optimal.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    right, left = (rank + 1) % world, (rank - 1) % world
    chunks = list(t.clone().chunk(world))
    recv = torch.empty_like(chunks[0])

    # Fase 1 — reduce-scatter
    for step in range(world - 1):
        send_idx = (rank - step) % world
        recv_idx = (rank - step - 1) % world
        req = dist.isend(chunks[send_idx].contiguous(), dst=right)
        dist.recv(recv, src=left)
        req.wait()
        chunks[recv_idx] += recv

    # Fase 2 — all-gather
    for step in range(world - 1):
        send_idx = (rank - step + 1) % world
        recv_idx = (rank - step) % world
        req = dist.isend(chunks[send_idx].contiguous(), dst=right)
        dist.recv(recv, src=left)
        req.wait()
        chunks[recv_idx] = recv.clone()

    return torch.cat(chunks)


def main():
    dist.init_process_group(backend="gloo", init_method="env://")
    rank, world = dist.get_rank(), dist.get_world_size()

    # Dato di partenza: il rank r possiede [r, r, r, r] (+1 per leggibilità)
    def start():
        return torch.full((world,), float(rank + 1))

    show("dato iniziale", rank, world, start())

    # 1. BROADCAST: il rank 0 invia il suo tensore a tutti.
    #    È ciò che DDP fa al wrapping del modello (pesi iniziali uguali).
    t = start() if rank == 0 else torch.zeros(world)
    dist.broadcast(t, src=0)
    show("broadcast(src=0)", rank, world, t)

    # 2. REDUCE: la somma arriva SOLO al rank 0 (gli altri restano sporchi).
    #    È l'aggregazione del parameter server: un nodo raccoglie tutto.
    t = start()
    dist.reduce(t, dst=0, op=dist.ReduceOp.SUM)
    show("reduce(dst=0, SUM)", rank, world, t)

    # 3. ALL-REDUCE: tutti ottengono la somma. È la primitiva di DDP.
    t = start()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    show("all_reduce(SUM)", rank, world, t)

    # 4. ALL-GATHER: ognuno riceve i tensori di tutti.
    out = [torch.zeros(1) for _ in range(world)]
    dist.all_gather(out, torch.tensor([float(rank + 1)]))
    show("all_gather", rank, world, torch.cat(out))

    # 5. SCATTER: il rank 0 distribuisce un pezzo diverso a ciascuno.
    #    (Concettualmente: lo sharding dei dati della lezione 3.)
    piece = torch.zeros(1)
    pieces = [torch.tensor([10.0 * (i + 1)]) for i in range(world)] if rank == 0 else None
    dist.scatter(piece, scatter_list=pieces, src=0)
    show("scatter(src=0)", rank, world, piece)

    # 6. RING ALL-REDUCE fatto a mano = reduce-scatter + all-gather.
    t = torch.arange(world * 2, dtype=torch.float32) + 100 * rank
    mine = ring_all_reduce(t)
    ref = t.clone()
    dist.all_reduce(ref)
    show("ring_all_reduce manuale", rank, world, mine)
    if rank == 0:
        ok = torch.allclose(mine, ref)
        print(f"\nCoincide con dist.all_reduce? {ok}")
        print(f"Elementi inviati per rank: 2·(N-1)/N·|t| = "
              f"{2 * (world - 1) / world * t.numel():.0f} su |t|={t.numel()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
