# heliumveto_GNN/shared/util/config.py
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class GraphConfig:
    use_energy: bool = True          # always True in your “simplest” case
    use_xyz: bool = True             # include fixed detector coords
    use_phi: bool = False            # add φ column?
    use_time: bool = False           # add hit time column?
    add_edge_distance: bool = True   # at least one edge feature
    k_nn: int = 20                   # connectivity

    @classmethod
    def from_yaml(cls, path: Path):
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw) if raw else cls()