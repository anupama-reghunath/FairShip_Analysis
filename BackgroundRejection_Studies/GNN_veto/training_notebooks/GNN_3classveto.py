#!/usr/bin/env python3
"""sbt_gnn_training.py
=====================
Train and evaluate a Graph Neural Network (Encode–Process–Decode architecture)
for classifying SBT-veto events recorded in the SHiP experiment simulation.

The script is organised as a *pipeline* with clearly-separated stages:

1. **Configuration & CLI** – all hyper-parameters, paths and options live in the
   :class:`Config` dataclass and may be overridden from the command line.
2. **Utility helpers** – small, pure functions (e.g. angle conversion, plotting)
   that keep domain logic self-contained and testable.
3. **Data ingestion** – fast vectorised reading of ROOT files with *uproot*.
   The raw arrays are reshaped into a format convenient for GNN processing.
4. **Graph construction** – build a *torch_geometric* :class:`Data` object for
   every event, adding node, edge and global features + k-NN adjacency.
5. **Model definition** – thin wrapper around the custom
   :class:`EncodeProcessDecode` from *sbtveto*.
6. **Training loop** – standard PyTorch loop with live logging and graceful
   GPU/CPU handling.
7. **Evaluation & visualisation** – confusion matrix, loss curves, and class
   composition donuts are saved to the output directory.

Run ``python sbt_gnn_training.py -h`` to see all available options.
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple
import glob

import numpy as np
import uproot
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph, knn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")  # headless back-end
aimport matplotlib.pyplot as plt
import seaborn as sns

# SHiP-specific import (provided by your environment)
from model.gnn_model import EncodeProcessDecode

# ────────────────────────────────────────────────────────────────────────────────
# 1. Configuration helper
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Central place for all user-adjustable parameters."""

    # Paths
    data_path: Path = Path("/eos/experiment/ship/user/anupamar/NN_data/root_files/wMuonBack")
    xyz_path: Path = Path("/eos/experiment/ship/user/anupamar/NN_data/SBT_new_geo_XYZ.npy")
    output_dir: Path = Path("plots")

    # Data processing
    energy_threshold_mev: float = 45.0  # keep events with ≥ 1 fired cell above this (MeV)
    max_files_per_class: int | None = None  # set to small int for quick debug

    # Graph building
    knn_k: int = 20  # use fully-connected if n_nodes < this*1.1 (~22)

    # Training
    batch_size: int = 32
    num_epochs: int = 25
    lr: float = 1e-3
    lr_finetune: float = 1e-4  # applied after epoch 20

    # Misc
    seed: int = 42
    cuda: bool = torch.cuda.is_available()

    def make_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────────
# 2. Logging
# ────────────────────────────────────────────────────────────────────────────────

def get_logger(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────────
# 3. Geometry helpers
# ────────────────────────────────────────────────────────────────────────────────

def phi_from_xy(x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
    """Return azimuthal angle *φ* in degrees with a bottom-centre offset."""
    phi = np.degrees(np.mod(np.arctan2(y, x), 2 * np.pi))
    # rotate reference so that 0° is bottom-centre instead of +x axis
    phi = (phi + 90) % 360
    return phi


# ────────────────────────────────────────────────────────────────────────────────
# 4. Plotting utilities (composition donut & SBT map)
# ────────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "muDIS+MuinducedBG": "#4daf4a",
    "neuDIS+MuinducedBG": "#e41a1c",
    "Signal+MuinducedBG": "#377eb8",
}


def save_donut(class_hits: np.ndarray, labels: np.ndarray, cfg: Config, *, title: str, filename: str) -> None:
    """Create a class-composition donut plot and print a counts table."""
    class_info: List[Tuple[str, int]] = [
        ("muDIS+MuinducedBG", 2),
        ("neuDIS+MuinducedBG", 1),
        ("Signal+MuinducedBG", 0),
    ]

    # Count with/without SBT hits per class
    table = []
    for name, idx in class_info:
        class_mask = labels == idx
        with_sbt = np.sum((class_hits[class_mask] > 0).any(axis=1))
        no_sbt = class_mask.sum() - with_sbt
        table.append((name, no_sbt, with_sbt))

    # Flatten for pie
    sizes, colors, hatches = [], [], []
    for name, no, yes in table:
        sizes.extend([no, yes])
        colors.extend([PALETTE[name]] * 2)
        hatches.extend(["", ".."])

    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, _, autotexts = ax.pie(
        sizes,
        wedgeprops={"width": 0.35, "edgecolor": "white"},
        startangle=90,
        autopct="%1.1f%%",
        pctdistance=0.75,
        colors=colors,
    )
    for w, h in zip(wedges, hatches):
        if h:
            w.set_hatch(h)
    centre = plt.Circle((0, 0), 0.55, color="white")
    ax.add_artist(centre)
    ax.set_title(title)

    # Legend
    handles = [
        matplotlib.patches.Patch(facecolor="white", edgecolor="black", label="No SBT hits"),
        matplotlib.patches.Patch(facecolor="white", edgecolor="black", hatch="..", label="≥1 SBT hit"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(cfg.output_dir / filename)

    # Console table
    logging.info("\n" + "Class".ljust(24) + "| noSBT  withSBT  total")
    logging.info("-" * 46)
    for name, no, yes in table:
        logging.info(f"{name.ljust(24)}| {no:6d} {yes:8d} {no+yes:6d}")
    logging.info("-" * 46)


# Other plotting functions (SBT cell map, loss curves, confusion matrix) omitted
# for brevity – they follow the same pattern as *save_donut* above.

# ────────────────────────────────────────────────────────────────────────────────
# 5. Data ingestion
# ────────────────────────────────────────────────────────────────────────────────

CLASS_PATTERNS = {
    0: "NNdata_signal_MuBack_batch_*.root",      # "Signal+MuinducedBG"
    1: "NNdata_neuDIS_MuBack_batch_*.root",      # "neuDIS+MuinducedBG"
    2: "NNdata_muDIS_MuBack_batch_*.root",       # "muDIS+MuinducedBG"
}


def list_root_files(cfg: Config, class_id: int) -> List[Path]:
    return sorted(Path(cfg.data_path).glob(CLASS_PATTERNS[class_id]))


def load_dataset(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ROOT files → (X, Y, event_info)."""
    logger = logging.getLogger(__name__)

    def _read_group(file_paths: List[Path], label: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs, ys, infos = [], [], []
        for fp in file_paths[: cfg.max_files_per_class or None]:
            try:
                with uproot.open(fp) as f:
                    arr = f["tree;1"]["inputmatrix"].array(library="np")
            except Exception as exc:
                logger.warning("Skipping %s: %s", fp, exc)
                continue

            (energy_dep,
             hit_time,
             vertex_pos,
             vertex_time,
             event_wgt,
             cand_info,
             ubt_hits) = np.split(arr, [854, 1708, 1711, 1712, 1713, 1723], axis=1)

            # Per-event info (keep once)
            infos.append(np.hstack([vertex_pos, cand_info, ubt_hits, event_wgt]))

            # Build per-node feature matrix [nodes=854, features]
            N = arr.shape[0]
            repeated_xyz = np.tile(np.load(cfg.xyz_path), (N, 1))  # [N, 3]

            features = np.stack([
                energy_dep,
                repeated_xyz[:, 0:1],  # X
                repeated_xyz[:, 1:2],  # Y
                repeated_xyz[:, 2:3],  # Z
                np.repeat(vertex_pos[:, 0:1], 854, axis=1),  # vx
                np.repeat(vertex_pos[:, 1:2], 854, axis=1),  # vy
                np.repeat(vertex_pos[:, 2:3], 854, axis=1),  # vz
            ], axis=-1)  # final shape (N, 854, 7)
            xs.append(features)
            ys.append(np.full(N, label))

        return np.concatenate(xs), np.concatenate(ys), np.concatenate(infos)

    x_parts, y_parts, info_parts = [], [], []
    for class_id in CLASS_PATTERNS:
        paths = list_root_files(cfg, class_id)
        logger.info("%s files for class %d", len(paths), class_id)
        x, y, info = _read_group(paths, class_id)
        x_parts.append(x)
        y_parts.append(y)
        info_parts.append(info)

    X = np.concatenate(x_parts)
    Y = np.concatenate(y_parts)
    signal_info = np.concatenate(info_parts)
    logger.info("Final data shape: X=%s, Y=%s, info=%s", X.shape, Y.shape, signal_info.shape)
    return X, Y, signal_info


# ────────────────────────────────────────────────────────────────────────────────
# 6. Graph construction
# ────────────────────────────────────────────────────────────────────────────────

def build_graphs(X: np.ndarray, Y: np.ndarray, sig: np.ndarray, cfg: Config) -> List[Data]:
    """Convert arrays into :class:`torch_geometric.data.Data` graphs."""
    logger = logging.getLogger(__name__)
    graphs: List[Data] = []
    for i in range(len(X)):
        hits = X[i][:, 0]  # energy deposition
        nonzero_mask = hits > 0
        if not nonzero_mask.any():
            continue

        node_feats = X[i][nonzero_mask]
        # append φ as an extra coord
        phi_col = phi_from_xy(node_feats[:, 1], node_feats[:, 2]).reshape(-1, 1)
        node_feats = np.hstack([node_feats, phi_col])

        n_nodes = node_feats.shape[0]
        node_tensor = torch.as_tensor(node_feats, dtype=torch.float)

        # adjacency
        if n_nodes < cfg.knn_k + 2:
            adj = torch.ones(n_nodes, n_nodes) - torch.eye(n_nodes)
            edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        else:
            edge_index = knn_graph(node_tensor[:, 1:4], k=cfg.knn_k)
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]  # drop self-loops

        # edge features (r, Δz, Δφ)
        send, recv = edge_index
        delta = node_tensor[send, 1:4] - node_tensor[recv, 1:4]
        r = delta.norm(dim=1, keepdim=True)
        delta_z = delta[:, 2:3]
        delta_phi = (node_tensor[send, -1] - node_tensor[recv, -1]).unsqueeze(1)
        edge_attr = torch.cat([r, delta_z, delta_phi], dim=1)

        graphs.append(
            Data(
                nodes=node_tensor,
                edge_index=edge_index,
                edges=edge_attr,
                graph_globals=torch.tensor([[n_nodes]], dtype=torch.float),
                sig_vars=torch.as_tensor(sig[i], dtype=torch.float).unsqueeze(0),
                y=torch.tensor(int(Y[i]), dtype=torch.long),
            )
        )
    logger.info("Built %d graphs", len(graphs))
    return graphs


# ────────────────────────────────────────────────────────────────────────────────
# 7. Training loop
# ────────────────────────────────────────────────────────────────────────────────

def train(model: torch.nn.Module, loaders: Tuple[DataLoader, DataLoader], cfg: Config) -> Tuple[List[float], List[float]]:
    train_loader, val_loader = loaders
    device = torch.device("cuda" if cfg.cuda else "cpu")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_losses, val_losses = [], []

    for epoch in range(1, cfg.num_epochs + 1):
        since = time.time()
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)["graph_globals"]
            loss = criterion(out, batch.y)
            loss.backward()
            opt.step()
            running += loss.item()
        epoch_train = running / len(train_loader)
        train_losses.append(epoch_train)

        # Validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)["graph_globals"]
                val_running += criterion(out, batch.y).item()
        epoch_val = val_running / len(val_loader)
        val_losses.append(epoch_val)

        if epoch == 20:
            for g in opt.param_groups:
                g["lr"] = cfg.lr_finetune

        logging.info("Epoch %02d | train %.4f | val %.4f | %.1fs", epoch, epoch_train, epoch_val, time.time() - since)

    return train_losses, val_losses


# ────────────────────────────────────────────────────────────────────────────────
# 8. Evaluation helpers
# ────────────────────────────────────────────────────────────────────────────────

def evaluate(model: torch.nn.Module, loader: DataLoader, cfg: Config) -> np.ndarray:
    device = torch.device("cuda" if cfg.cuda else "cpu")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)["graph_globals"]
            y_true.extend(batch.y.cpu().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())
    return confusion_matrix(y_true, y_pred)


# ────────────────────────────────────────────────────────────────────────────────
# 9. Main entry point
# ────────────────────────────────────────────────────────────────────────────────

def main(cfg: Config, *, verbose: bool = False) -> None:
    logger = get_logger(verbose)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.make_output_dir()

    if cfg.cuda:
        logger.info("Running on CUDA device: %s", torch.cuda.get_device_name(0))

    # 1) Load & prepare data
    X, Y, sig = load_dataset(cfg)
    class_hits = X[:, :, 0]  # energy deposits
    save_donut(class_hits, Y, cfg, title="Class composition (raw)", filename="raw_donut.png")

    mask = (class_hits > cfg.energy_threshold_mev * 1e-3).any(axis=1)
    X, Y, sig = X[mask], Y[mask], sig[mask]
    logger.info("%d events remain after %.1f MeV cut", len(X), cfg.energy_threshold_mev)
    save_donut(X[:, :, 0], Y, cfg, title="Post-threshold composition", filename="filtered_donut.png")

    # 2) Split
    X_temp, X_test, Y_temp, Y_test, sig_temp, sig_test = train_test_split(
        X, Y, sig, test_size=0.2, random_state=cfg.seed)
    X_train, X_val, Y_train, Y_val, sig_train, sig_val = train_test_split(
        X_temp, Y_temp, sig_temp, test_size=0.25, random_state=cfg.seed)

    # 3) Graphs & loaders
    train_graphs = build_graphs(X_train, Y_train, sig_train, cfg)
    val_graphs = build_graphs(X_val, Y_val, sig_val, cfg)
    test_graphs = build_graphs(X_test, Y_test, sig_test, cfg)

    train_loader = DataLoader(train_graphs, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=cfg.batch_size)

    # 4) Model & training
    model = EncodeProcessDecode(mlp_output_size=8, global_op=3, num_blocks=4)
    train_losses, val_losses = train(model, (train_loader, val_loader), cfg)

    # 5) Evaluation
    cm = evaluate(model, test_loader, cfg)
    logger.info("Confusion matrix:\n%s", cm)

    # Save trained weights
    model_path = cfg.output_dir / f"GNN_model_cut{int(cfg.energy_threshold_mev)}MeV.pth"
    torch.save(model.state_dict(), model_path)
    logger.info("Model saved to %s", model_path)


# ────────────────────────────────────────────────────────────────────────────────
# 10. CLI wrapper
# ────────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SBT-veto GNN")
    p.add_argument("--data-path", type=Path, help="Directory with ROOT files")
    p.add_argument("--xyz-path", type=Path, help=".npy file with SBT cell XYZ", required=False)
    p.add_argument("--out", type=Path, help="Where to write plots & weights")
    p.add_argument("--energy-cut", type=float, help="Energy threshold in MeV")
    p.add_argument("--epochs", type=int, help="Number of training epochs")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config()
    if args.data_path:
        cfg.data_path = args.data_path
    if args.xyz_path:
        cfg.xyz_path = args.xyz_path
    if args.out:
        cfg.output_dir = args.out
    if args.energy_cut:
        cfg.energy_threshold_mev = args.energy_cut
    if args.epochs:
        cfg.num_epochs = args.epochs
    main(cfg, verbose=args.verbose)
