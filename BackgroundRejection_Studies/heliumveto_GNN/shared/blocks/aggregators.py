from torch_scatter import scatter_sum

def agg_edges_to_nodes(edge_attr, receivers, num_nodes):
    """
    Sum incoming edge attributes for every node.
    edge_attr : (E, F_e)
    receivers : (E,)  destination node index per edge
    """
    return scatter_sum(edge_attr, receivers, dim=0, dim_size=num_nodes)

def agg_nodes_to_globals(node_attr, batch):
    """
    Sum (or mean) all node attributes per graph in a mini-batch.
    """
    return scatter_sum(node_attr, batch, dim=0)

def agg_edges_to_globals(edge_attr, batch):
    return scatter_sum(edge_attr, batch, dim=0)