#!/usr/bin/env python3
"""
train_binary.py
===============
Train a *binary* SBT-veto GNN:
    class 0 → signal (+MuBack)
    class 1 → combined background (νDIS + μDIS)

Example
-------
python scripts/train_binary.py \
    --path   /eos/experiment/ship/user/anupamar/NN_data/root_files/wMuonBack \
    --config configs/simple.yaml \
    --epochs 30 \
    --out    models/binary_simple.pth
"""
from __future__ import annotations
import argparse, glob, os, re, time
from pathlib import Path
from typing import List

import numpy as np
import uproot
import torch
from torch_geometric.loader import DataLoader

from heliumveto_GNN.shared.util.config   import GraphConfig
from heliumveto_GNN.shared.util.geometry import XYZ
from heliumveto_GNN.shared.gnn.graphcoder import event_to_graph
from heliumveto_GNN.binary.model.gnn_model import GNNBinary  # <── binary model

# ------------------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------------------
def _unique_by_job(files):
    """Return at most one ROOT file per job id."""
    seen, uniq = set(), []
    for f in sorted(files):
        m = re.search(r'_job_(\d+)\.root$', os.path.basename(f))
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            uniq.append(f)
    return uniq

def lift_graphs(paths: List[Path], label: int, cfg: GraphConfig):
    """Yield Data graphs with .y == label (float for BCE)."""
    for fp in paths:
        arr_all = uproot.open(fp)["tree;1"]["inputmatrix"].array(library="np")
        for ev in arr_all:
            try:
                g = event_to_graph(ev, XYZ, cfg)
            except ValueError:
                continue
            g.y = torch.tensor([float(label)], dtype=torch.float)  # BCE expects float
            yield g

def build_dataset(data_path: str, cfg: GraphConfig):
    sig_files_raw = glob.glob(f"{data_path}/NNdata_signal_MuBack_batch_*.root")
    neu_files_raw = glob.glob(f"{data_path}/NNdata_neuDIS_MuBack_batch_*.root")
    mu_files_raw  = glob.glob(f"{data_path}/NNdata_muDIS_MuBack_batch_*.root")

    sig_files = _unique_by_job(sig_files_raw)
    bkg_files = _unique_by_job(neu_files_raw + mu_files_raw)

    sig_graphs = list(lift_graphs(sig_files, 0, cfg))
    bkg_graphs = list(lift_graphs(bkg_files, 1, cfg))

    graphs = sig_graphs + bkg_graphs
    print(f"Loaded {len(graphs)} graphs "
          f"({len(sig_graphs)} signal, {len(bkg_graphs)} background)")
    return graphs

# ------------------------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------------------------
def train_epoch(model, loader, loss_fn, opt, device):
    model.train(); run = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        logit = model(batch)                       # shape (B,)
        loss  = loss_fn(logit, batch.y.squeeze())
        loss.backward(); opt.step()
        run += loss.item()
    return run / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval(); loss=0.0; correct=0; n=0
    for batch in loader:
        batch = batch.to(device)
        logit = model(batch)
        loss += loss_fn(logit, batch.y.squeeze()).item()
        pred = (torch.sigmoid(logit) > 0.5).float()
        correct += (pred == batch.y.squeeze()).sum().item()
        n += batch.y.size(0)
    return loss/len(loader), correct/n

# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path",   required=True, help="folder containing ROOT files")
    p.add_argument("--config", required=True, help="GraphConfig YAML")
    p.add_argument("--batch",  type=int, default=32)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--device", choices=["cuda","cpu"], default=None)
    p.add_argument("--out",    type=Path, default=Path("binary_model.pth"))
    args = p.parse_args()

    cfg     = GraphConfig.from_yaml(Path(args.config))
    device  = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    graphs = build_dataset(args.path, cfg)
    split  = int(0.8*len(graphs))
    train_loader = DataLoader(graphs[:split],  batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(graphs[split:], batch_size=args.batch)

    sample = graphs[0]
    model  = GNNBinary(node_dim  = sample.nodes.shape[1],
                       edge_dim  = sample.edges.shape[1],
                       global_dim= sample.graph_globals.shape[1]).to(device)

    opt     = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, loss_fn, opt, device)
        vl, va = eval_epoch(model,  val_loader,   loss_fn, device)
        print(f"Epoch {epoch:02d}  train {tr:.4f}  val {vl:.4f}  acc {va:.3f}  {time.time()-t0:.1f}s")
        if epoch == 20:
            for g in opt.param_groups: g["lr"] = args.lr/10

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"✓ Saved weights → {args.out.resolve()}")

if __name__ == "__main__":
    main()
