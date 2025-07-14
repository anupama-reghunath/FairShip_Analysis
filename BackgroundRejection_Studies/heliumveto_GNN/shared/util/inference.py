def load_binary(path, *dims):
    model = GNNBinary(*dims)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def predict(model, graph, thr=0.5):
    import torch
    with torch.no_grad():
        p = torch.sigmoid(model(graph))
    return p.item(), p.item() > thr