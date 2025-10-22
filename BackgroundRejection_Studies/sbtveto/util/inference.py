import numpy as np
import torch
from torch_geometric.nn import knn
from torch_geometric.data import Data
from sbtveto.model.nn_model import NN
import joblib


def nn_output(model, data, scalar):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    x = scalar.transform(data)
    x = torch.tensor(x, dtype =torch.float32 ).to(device)
    output = model(x)
    sbt_decision = (torch.max(output, dim = 1).indices != 0)#veto returns True if not signal(0)
    classification=torch.max(output, dim = 1).indices
    return output, sbt_decision,classification


def adjacency(n_dau):
    """ generates a fully connected adjacency
        for a mother to daughters """
    A=np.ones((n_dau+1, n_dau+1))
    return A


def gnn_output(model, x, sbt_xyz):

    Ncells = sbt_xyz.shape[1]  


    # energy:             (N_events, Ncells)
    energy = np.expand_dims(x[:, :Ncells], 0)

    # geometry (X, Y, Z): (3, Ncells)  replicated to N_events
    geom   = np.repeat(sbt_xyz[:3, :][None, :, :], x.shape[0], axis=0)

    # vertex position:    (N_events, 3, Ncells)  ← NEW
    vx = np.repeat(x[:, -3:-2], Ncells, axis=1)       # vertex x
    vy = np.repeat(x[:, -2:-1], Ncells, axis=1)       # vertex y
    vz = np.repeat(x[:, -1:  ], Ncells, axis=1)       # vertex z
    vxyz = np.stack([vx, vy, vz], axis=1)

    # φ coordinate:       (N_events, 1, Ncells) ← NEW
    phi = np.expand_dims(np.arctan2(geom[:,1], geom[:,0]), 1)

    # STACK → shape: (N_events, 8, Ncells)
    X = np.concatenate([energy, geom, vxyz, phi], axis=1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    # ---------------- Node matrix ----------------
    X_event = X[0]                     # (8, Ncells)
    hits    = X_event[0] > 0           # energy > 0
    Xcon    = X_event[:, hits].T       # (nHits, 8)  ← exactly what model expects

    if Xcon.shape[0] == 0:
        print("No fired SBT cells – falling back to NN.")
        ...  # (keep your NN fallback block)
        
    # ------------- Edge index --------------------
    Xcon2 = torch.tensor(Xcon, dtype=torch.float)
    if Xcon.shape[0] < 22:
        A = adjacency(Xcon.shape[0]-1)
        tensor_edge_index = torch.tensor(A, dtype=torch.float).nonzero().t()
    else:
        tensor_edge_index = knn(Xcon2, Xcon2, k=5)
        tensor_edge_index = tensor_edge_index[:, tensor_edge_index[0]!=tensor_edge_index[1]]

    edge_index = tensor_edge_index.numpy()

    # ------------- Edge attributes --------------
    r3d  = torch.norm(Xcon2[edge_index[0], 1:4] - Xcon2[edge_index[1], 1:4], dim=1)
    dZ   = Xcon2[edge_index[0], 3] - Xcon2[edge_index[1], 3]
    dPhi = Xcon2[edge_index[0], 7] - Xcon2[edge_index[1], 7]
    edge_features = torch.stack([r3d, dZ, dPhi], dim=1)

    # ------------- Global attribute -------------
    nHits = Xcon.shape[0]
    global_features = torch.tensor([[nHits]], dtype=torch.float)

    edgepos = torch.zeros(edge_index.shape[1], dtype=torch.long)
    graph   = Data(nodes=Xcon2,
                   edge_index=tensor_edge_index,
                   edges=edge_features,
                   graph_globals=global_features,
                   edgepos=edgepos)

    graph['receivers'] = graph.edge_index[1]
    graph['senders']   = graph.edge_index[0]
    graph.batch        = torch.zeros(graph.nodes.shape[0], dtype=torch.long)


    # load model weights in here need to improve this
    #model(graph.clone().detach())
    #if device == 'cpu':
    #    model.load_state_dict(torch.load('data/SBT_vacuum_multiclass_4block_GNN_noUBT.pt', weights_only=True,
    #                                     map_location=torch.device('cpu')))
    #else:
    #    model.load_state_dict(torch.load('data/SBT_vacuum_multiclass_4block_GNN_noUBT.pt', weights_only=True))
    
    model.eval()
    graph.to(device)
    model.to(device)
    output_graph = model(graph)

    sbt_decision = (torch.max(output_graph.graph_globals, dim = 1).indices != 0)#veto returns True if not signal(0)

    classification = torch.max(output_graph.graph_globals, dim = 1).indices
    return output_graph.graph_globals, sbt_decision, classification

def gnn_output_binary(model, inputmatrix, XYZ, threshold=0.6):
    """
    Run a 1-output GNN and return (veto_decision, P(background)).
    Works with the same 'inputmatrix' + 'XYZ' interface as gnn_output().
    """
    logits, _, _ = gnn_output(model, inputmatrix, XYZ)   # logits shape (1,1)
    prob_bg = torch.sigmoid(logits)[0, 0].item()         # scalar ∈ [0,1]
    return (prob_bg > threshold), prob_bg

def gnn_output_deltaT(model, x, sbt_xyz):
    """
    Runs GNN on input x of shape (1, 854, 5): [E, Δt, vx, vy, vz].
    Also adds an additional phi feature onto the deltaT
    Returns: (logits, veto_decision, classification)
    """
    Ncells = sbt_xyz.shape[1]

    # x shape: (1, 854, 5)
    energy    = x[0, :, 0]         # shape: (854,)
    delta_t   = x[0, :, 1]         # shape: (854,)
    vx_array  = x[0, :, 2]         
    vy_array  = x[0, :, 3]
    vz_array  = x[0, :, 4]

    # stack features
    geom = sbt_xyz  # shape: (3, 854)
    phi  = np.arctan2(geom[1], geom[0])  # shape: (854,)

    # build final feature matrix: (854, 8)
    features = np.stack([
        energy,
        geom[0], geom[1], geom[2],     # X, Y, Z
        vx_array, vy_array, vz_array,  # vertex x/y/z
        delta_t,
        phi,                       
    ], axis=1)

    # Keep only non-zero energy hits
    valid_mask = energy > 0
    Xcon = features[valid_mask]  # shape: (nHits, 8)

    if Xcon.shape[0] == 0:
        # no valid hits
        dummy_logits = torch.tensor([[0.0]])   # logit → P(bg)=0.5
        return dummy_logits, False, torch.tensor([0])

    # Edge index
    Xcon2 = torch.tensor(Xcon, dtype=torch.float)
    if Xcon.shape[0] < 22:
        A = adjacency(Xcon.shape[0] - 1)
        tensor_edge_index = torch.tensor(A, dtype=torch.float).nonzero().t()
    else:
        tensor_edge_index = knn(Xcon2, Xcon2, k=5)
        tensor_edge_index = tensor_edge_index[:, tensor_edge_index[0] != tensor_edge_index[1]]

    edge_index = tensor_edge_index.numpy()

    # Edge attributes
    r3d  = torch.norm(Xcon2[edge_index[0], 1:4] - Xcon2[edge_index[1], 1:4], dim=1)
    dZ   = Xcon2[edge_index[0], 3] - Xcon2[edge_index[1], 3]
    dPhi = Xcon2[edge_index[0], 7] - Xcon2[edge_index[1], 7]  # uses Δt column here (if you drop phi)

    edge_features = torch.stack([r3d, dZ, dPhi], dim=1)

    # Global features
    global_features = torch.tensor([[Xcon.shape[0]]], dtype=torch.float)

    edgepos = torch.zeros(edge_index.shape[1], dtype=torch.long)
    graph = Data(
        nodes=Xcon2,
        edge_index=tensor_edge_index,
        edges=edge_features,
        graph_globals=global_features,
        edgepos=edgepos,
        receivers=tensor_edge_index[1],
        senders=tensor_edge_index[0],
        batch=torch.zeros(Xcon2.shape[0], dtype=torch.long)
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    graph = graph.to(device)
    model = model.to(device)

    output_graph = model(graph)

    sbt_decision = torch.max(output_graph.graph_globals, dim=1).indices != 0
    classification = torch.max(output_graph.graph_globals, dim=1).indices
    return output_graph.graph_globals, sbt_decision, classification



def gnn_output_binary_wdeltaT(model, inputmatrix, XYZ, threshold=0.6):
    """
    Run a 1-output GNN and return (veto_decision, P(background)).
    Works with the same 'inputmatrix' + 'XYZ' interface as gnn_output().
    """
    logits, _, _ = gnn_output_deltaT(model, inputmatrix, XYZ)   # logits shape (1,1)
    prob_bg = torch.sigmoid(logits)[0, 0].item()         # scalar ∈ [0,1]
    return (prob_bg > threshold), prob_bg