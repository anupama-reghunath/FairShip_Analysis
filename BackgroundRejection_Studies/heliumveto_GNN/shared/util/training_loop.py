import torch, time
from tqdm import tqdm

def train_loop(model, loader, criterion, opt, device="cpu"):
    model.train()
    running = 0.0
    for batch in tqdm(loader, desc="train"):
        batch = batch.to(device)
        opt.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        opt.step()
        running += loss.item()
    return running / len(loader)