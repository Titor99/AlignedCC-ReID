import torch

def train_collate_gcn_mask(batch):
    imgs, masks, pids, _, pathes = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs), pids, pathes, torch.cat(masks)

def val_collate_gcn_mask(batch):
    imgs, masks, pids, camids, paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, paths, torch.cat(masks, dim=0)