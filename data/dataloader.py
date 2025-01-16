import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ObjectsDataset(Dataset):
    """
    Creates dataloader for object-centric data.

    Args:
        X: numpy array of observations
        Z: numpy array of ground-truth latents
        transform: torchvision transformation for data

    Returns:
        inferred batch of observations and ground-truth latents
    """
    def __init__(self, X, masks, prov_mask=True, transform=None):
        self.obs = X
        self.masks = masks
        self.transform = transform
        self.prov_mask = prov_mask

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        x = self.obs[idx]

        if self.transform is not None:
            x = self.transform(x)

        if not self.prov_mask:
            masks = torch.zeros(x.shape)
        else:
            masks = self.masks[idx]

        return x, masks


def get_dataloader(args):
    """
    Generates or loads pre-generated dataset and creates training and validation dataloaders

    Args:
        args: Command line arguments from train_model.py

    Returns:
        train and validation PyTorch Dataloaders
    """
    if args.data == "sprites":
        data_path = "data/datasets/4_obj_sprites.npz"
        X, Z, masks = np.load(data_path)['arr_0'], np.load(data_path)['arr_1'], np.load(data_path)['arr_2']
        X_train, X_val, X_test = X[0:90000], X[90000:95000], X[95000:100000]
        _, mask_val, mask_test = masks[0:90000], masks[90000:95000], masks[95000:100000]
        transform = transforms.ToTensor()

    elif args.data == "clevr":
        data_path = "data/datasets/clevr_6.npz"
        X, Z, masks = np.load(data_path)['arr_0'], np.load(data_path)['arr_1'], np.load(data_path)['arr_2']
        X_train, X_val, X_test = X[0:49483], X[49483:51483], X[51483:53483]
        _, mask_val, mask_test = masks[0:49483], masks[49483:51483], masks[51483:53483]
        transform = transforms.ToTensor()

    elif args.data == "clevrtex":
        X_train = np.load("data/datasets/clevrtex_train.npz")['arr_0'].astype(np.float32)
        X_val = np.load("data/datasets/clevrtex_val.npz")['arr_0'].astype(np.float32)
        mask_val = np.load("data/datasets/clevrtex_val_mask.npz")['arr_0'].astype(np.float32)
        X_test = np.load("data/datasets/clevrtex_test.npz")['arr_0'].astype(np.float32)
        mask_test = np.load("data/datasets/clevrtex_test_mask.npz")['arr_0'].astype(np.float32)
        transform = None

    # create dataloaders
    train_loader = DataLoader(ObjectsDataset(X_train, masks=None, prov_mask=False, transform=transform),
                              batch_size=args.batch_size,
                              persistent_workers=True,
                              num_workers=1,
                              pin_memory=True,
                              shuffle=True)

    val_loader = DataLoader(ObjectsDataset(X_val, mask_val, transform=transform),
                            batch_size=50,
                            persistent_workers=True,
                            num_workers=1,
                            pin_memory=True,
                            shuffle=True)

    test_loader = DataLoader(ObjectsDataset(X_test, mask_test, transform=transform),
                            batch_size=10,
                            persistent_workers=True,
                            num_workers=1,
                            pin_memory=True,
                            shuffle=False)

    return train_loader, val_loader, test_loader



