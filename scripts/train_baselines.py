"""Script oficial, simétrico e reprodutível para treino dos baselines (Protocolo v4).

Treina Temporal Meta-SASRec e Temporal Meta-HSTU sob o mesmo regime:
- 25 épocas com AdamW e CosineAnnealingLR
- 256 negativos uniformes por sequência de treino (warm_items)
- Escala de logits = 10.0
- Mesma precisão bfloat16
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from verify_audit_fairness import TemporalMetaHSTU, TemporalMetaSASRec


def train_model(
    model: nn.Module,
    model_name: str,
    pre_hist: np.ndarray,
    pre_ts: np.ndarray,
    pre_targets: np.ndarray,
    warm_items_t: torch.Tensor,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_neg: int = 256,
    logit_scale: float = 10.0,
    device: torch.device | None = None,
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=================================================================")
    print(f"Treino de Baseline: {model_name}")
    print(f"  Epocas: {epochs} | Batch: {batch_size} | LR: {lr} | WD: {weight_decay}")
    print(f"  Negativos por sequencia: {n_neg} | Logit Scale: {logit_scale:.1f}")
    print("=================================================================")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = len(pre_targets)
    total_steps = epochs * ((n_train + batch_size - 1) // batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda")

    indices = np.arange(n_train)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(indices)
        total_loss = 0.0
        steps = 0

        for start in range(0, n_train, batch_size):
            batch_idx = indices[start : start + batch_size]
            B = len(batch_idx)

            h_t = torch.from_numpy(pre_hist[batch_idx]).to(device=device, dtype=torch.long)
            ts_t = torch.from_numpy(pre_ts[batch_idx]).to(device=device, dtype=torch.long)
            targets = torch.from_numpy(pre_targets[batch_idx]).to(device=device, dtype=torch.long)

            neg_idx = torch.randint(0, len(warm_items_t), (B, n_neg), device=device)
            negs = warm_items_t[neg_idx]

            eval_items = torch.cat([targets.unsqueeze(1), negs], dim=1)
            labels = torch.zeros(B, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                states = model.sequence_states(h_t, ts_t)
                valid = h_t.ne(0)
                positions = valid.sum(1).clamp_min(1) - 1
                batch_range = torch.arange(B, device=device)
                last_state = states[batch_range, positions]

                cand_emb = F.normalize(model.items(eval_items), dim=-1)
                logits = (last_state.unsqueeze(1) * cand_emb).sum(-1) * logit_scale
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        print(
            f"  [{model_name}] Epoca {epoch:02d}/{epochs:02d} | "
            f"Loss: {avg_loss:.4f} | Tempo: {time.time() - t0:.1f}s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino simetrico dos baselines SASRec e HSTU")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed/ml32m-global-temporal-v4")
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    with open(args.data_dir / "protocol.json", encoding="utf-8") as f:
        proto = json.load(f)

    n_items = proto["item_count"] + 1
    max_length = 200

    csv.field_size_limit(2**31 - 1)
    seq_path = args.data_dir / "hstu_training_sequences.csv"
    train_items_list, train_ts_list = [], []
    with open(seq_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            train_items_list.append([int(x) for x in r["sequence_item_ids"].split(",")])
            train_ts_list.append([int(x) for x in r["sequence_timestamps"].split(",")])

    n_train = len(train_items_list)
    counts = np.zeros(n_items, dtype=np.int32)
    for items in train_items_list:
        for item in items:
            if 0 < item < n_items:
                counts[item] += 1
    warm_items = np.flatnonzero(counts > 0)
    warm_items_t = torch.tensor(warm_items, device=device, dtype=torch.long)

    print("Pre-tensorizando sequencias de treino...")
    pre_hist = np.zeros((n_train, max_length), dtype=np.int32)
    pre_ts = np.zeros((n_train, max_length), dtype=np.int64)
    pre_targets = np.zeros(n_train, dtype=np.int32)
    for idx in range(n_train):
        s_items = train_items_list[idx]
        s_ts = train_ts_list[idx]
        if len(s_items) > 1:
            w_items = s_items[-max_length:]
            w_ts = s_ts[-max_length:]
            L = len(w_items) - 1
            pre_hist[idx, :L] = w_items[:-1]
            pre_ts[idx, :L] = w_ts[:-1]
            pre_targets[idx] = w_items[-1]
        else:
            pre_targets[idx] = s_items[0]

    # Treinar SASRec
    sasrec = TemporalMetaSASRec(n_items, 64, max_length, 4, 2, 0.2).to(device)
    train_model(
        sasrec,
        "Temporal Meta-SASRec",
        pre_hist,
        pre_ts,
        pre_targets,
        warm_items_t,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )
    torch.save(sasrec.state_dict(), "artifacts/temporal_sasrec_ml32m.pt")
    print("Checkpoint gravado: artifacts/temporal_sasrec_ml32m.pt")

    # Treinar HSTU
    hstu = TemporalMetaHSTU(n_items, 64, max_length, 4, 2, 0.2).to(device)
    train_model(
        hstu,
        "Temporal Meta-HSTU",
        pre_hist,
        pre_ts,
        pre_targets,
        warm_items_t,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )
    torch.save(hstu.state_dict(), "artifacts/temporal_hstu_ml32m.pt")
    print("Checkpoint gravado: artifacts/temporal_hstu_ml32m.pt")


if __name__ == "__main__":
    main()
