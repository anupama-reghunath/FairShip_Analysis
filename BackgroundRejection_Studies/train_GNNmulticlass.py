#!/usr/bin/env python3
"""
train_multiclass.py
===================
Train a 3-class SBT-veto Graph Neural Network.

The script is *config-driven*:

    python scripts/train_multiclass.py \
        --path   /eos/experiment/ship/user/anupamar/NN_data/root_files/wMuonBack
        --config  configs/simple.yaml \
        --epochs  30 \
        --out     models/multiclass_simple.pth

Flags
-----
--path                  : path to ROOT files
--config                :  YAML file parsed into heliumveto_GNN.shared.util.GraphConfig
--batch                 :  batch size (default 32)
--epochs                :  training epochs (default 25)
--lr                    :  learning-rate (default 1e-3)
--device                :  "cuda" | "cpu" (auto if omitted)
--out                   :  where to save the `.pth` weights
"""

from __future__ import annotations
import argparse, glob, itertools, sys, time
from pathlib import Path
from typing import List

import numpy as np
import uproot
import torch
from torch_geometric.loader import DataLoader

from heliumveto_GNN.shared.util.config import GraphConfig
from heliumveto_GNN.shared.util.geometry import XYZ          # loads SBT geometry
from heliumveto_GNN.shared.gnn.graphcoder import event_to_graph
from heliumveto_GNN.multiclass.model.gnn_model import GNNMulti


# ------------------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------------------

def _unique_by_job(files):
    """
    Keep only one file per job index in a list of ROOT-file paths.
    """
    seen = set()
    unique = []
    for f in sorted(files):          # 'sorted' keeps the result deterministic
        m = re.search(r'_job_(\d+)\.root$', os.path.basename(f))
        if m:
            job = m.group(1)
            if job not in seen:      # first time this job id shows up
                seen.add(job)
                unique.append(f)
    return unique

def root_files(patterns: List[str]) -> List[Path]:
    """Expand a list of glob patterns into sorted Paths."""
    return sorted(
        Path(p).resolve() for pattern in patterns for p in glob.glob(pattern)
    )

def lift_graphs(paths: List[Path], class_id: int, cfg: GraphConfig):
    """Yield graphs (Data objects) with .y label attached."""
    for fp in paths:
        # load full array for this ROOT file (many events)
        arr_all = uproot.open(fp)["tree;1"]["inputmatrix"].array(library="np")
        for ev in arr_all:
            try:
                g = event_to_graph(ev,XYZ, cfg)
            except ValueError:
                continue                 # skip "no SBT hits" events
            g.y = torch.tensor([class_id], dtype=torch.long)
            yield g

def build_dataset(data_path, cfg: GraphConfig):
    
    neu_files_raw       = glob.glob(f"{data_path}/NNdata_neuDIS_MuBack_batch_*.root")
    signal_files_raw    = glob.glob(f"{data_path}/NNdata_signal_MuBack_batch_*.root")
    mu_files_raw        = glob.glob(f"{data_path}/NNdata_muDIS_MuBack_batch_*.root")

    neu_files       = _unique_by_job(neu_files_raw)    
    signal_files    = _unique_by_job(signal_files_raw)   
    mu_files        = _unique_by_job(mu_files_raw)   

    sig = list(lift_graphs(root_files(args.signal), 0, cfg))
    neu = list(lift_graphs(root_files(args.neu),    1, cfg))
    mu  = list(lift_graphs(root_files(args.mu),     2, cfg))

    graphs = sig + neu + mu
    print(f"Loaded {len(graphs)} graphs  "
          f"({len(sig)} signal, {len(neu)} νDIS, {len(mu)} μDIS)")
    return graphs


# ------------------------------------------------------------------------------
# Training utilities
# ------------------------------------------------------------------------------

def train_epoch(model, loader, loss_fn, opt, device):
    model.train()
    running = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        logits = model(batch)
        loss   = loss_fn(logits, batch.y.squeeze())  # .y shape (B,1)
        loss.backward()
        opt.step()
        running += loss.item()
    return running / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    loss, correct, n = 0.0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss  += loss_fn(logits, batch.y.squeeze()).item()
        pred   = logits.argmax(dim=1)
        correct += (pred == batch.y.squeeze()).sum().item()
        n      += batch.y.size(0)
    return loss / len(loader), correct / n


# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

def main():

    p = argparse.ArgumentParser()
    
    p.add_argument("--path",   type=str, required=False, help="path to ROOT files")
    p.add_argument("--config", required=True, help="GraphConfig YAML within configs")
    p.add_argument("--batch",  type=int, default=32, help="mini-batch size")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"], help="override device; default auto")
    p.add_argument("--out",    type=Path, default=Path("multiclass_model.pth"))
    
    args = p.parse_args()

    cfg  = GraphConfig.from_yaml(Path(args.config))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    graphs = build_dataset(args.path, cfg)
    # Simple split: 80 % train, 20 % val
    split = int(0.8 * len(graphs))
    train_loader = DataLoader(graphs[:split], batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(graphs[split:], batch_size=args.batch)

    sample = graphs[0]
    model = GNNMulti(node_dim   = sample.nodes.shape[1],
                     edge_dim   = sample.edges.shape[1],
                     global_dim = sample.graph_globals.shape[1]).to(device)

    opt      = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn  = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, loss_fn, opt, device)
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}  "
              f"train_loss {train_loss:.4f}  "
              f"val_loss {val_loss:.4f}  "
              f"val_acc {val_acc:.3f}  "
              f"{dt:.1f}s")

        # simple LR drop
        if epoch == 20:
            for g in opt.param_groups:
                g["lr"] = args.lr / 10

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"✓ Saved weights  →  {args.out.resolve()}")

if __name__ == "__main__":
    main()