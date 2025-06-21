import torch

def to_device(batch, device, non_blocking=False):
    if isinstance(batch, (list, tuple)):
        return type(batch)([
            to_device(u, device, non_blocking)
            for u in batch])
    elif isinstance(batch, dict):
        return type(batch)([
            (k, to_device(v, device, non_blocking))
            for k, v in batch.items()])
    elif isinstance(batch, torch.Tensor) and batch.device != device:
        batch = batch.to(device, non_blocking=non_blocking)
    else:
        return batch
    return batch

def to_numpy(batch, non_blocking=False):
    if isinstance(batch, (list, tuple)):
        return type(batch)([
            to_numpy(u, non_blocking)
            for u in batch])
    elif isinstance(batch, dict):
        return type(batch)([
            (k, to_numpy(v, non_blocking))
            for k, v in batch.items()])
    elif isinstance(batch, torch.Tensor):
        batch = batch.cpu().numpy()
    else:
        return batch
    return batch

def to_item(batch, non_blocking=False):
    if isinstance(batch, (list, tuple)):
        return type(batch)([
            to_item(u, non_blocking)
            for u in batch])
    elif isinstance(batch, dict):
        return type(batch)([
            (k, to_item(v, non_blocking))
            for k, v in batch.items()])
    elif isinstance(batch, torch.Tensor):
        batch = batch.cpu().item()
    else:
        return batch
    return batch