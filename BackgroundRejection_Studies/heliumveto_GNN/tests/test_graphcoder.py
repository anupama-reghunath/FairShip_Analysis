def test_graph_has_fired_nodes(fake_graph):
    # should keep 7 nodes, one per fired cell
    assert fake_graph.nodes.shape[0] == 7
    # edge_index must have shape (2, E)
    assert fake_graph.edge_index.shape[0] == 2
    # edge attributes one-to-one with edges
    assert fake_graph.edges.shape[0] == fake_graph.edge_index.shape[1]