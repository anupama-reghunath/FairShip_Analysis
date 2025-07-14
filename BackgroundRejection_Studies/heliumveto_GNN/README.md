
├─ heliumveto_gnn/         # top-level Python package  ← **new name**
│  ├─ shared/              # code common to both GNN types
│  │  ├─ blocks/
│  │  ├─ gnn/
│  │  └─ util/
│  ├─ binary/              # **Binary-classification GNN**
│  │  └─ model/            # gnn_model.py (2 classes), nn_model.py
│  │ 
│  ├─ multiclass/          # **Multi-class GNN**
│  │  └─model/            # gnn_model.py (3 classes), nn_model.py
│  │
│  ├─ tests
│  │	├─ test_geometry.py
│  │	├─ test_graphcoder.py
│  │	├─ test_blocks.py
│  │	├─ test_models.py
│  │	├─ test_end2end.py
│  │	└─ conftest.py          ← tiny fixtures shared by all tests
│  └─ README.md
│
├─ train_binary.py      # uses heliumveto_GNN.binary
└─ train_multiclass.py  # uses heliumveto_GNN.multiclass
