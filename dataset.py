import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms

import checkerboard

class CheckerboardDataset(Dataset):
    def __init__(self, dataset_size, n_grid_points: int = 128, noisy_points: int = 200, device: str = "cpu", num_squares=4, method=checkerboard.METHODS[3], show_checkerboards=None):
        self.dataset_size = dataset_size
        self.n_grid_points = n_grid_points
        self.noisy_points = noisy_points
        print(f'Noisy points: {noisy_points}')
        self.device = device
        self.num_squares = num_squares
        self.seeds = torch.randint(0, 100000, size=(dataset_size,))
        self.method = method
        self.items = self.generate_items()
        if show_checkerboards is not None:
            import matplotlib.pyplot as plt
            import os
            _, axs = plt.subplots(1, 4, figsize=(20, 5))
            for i in range(4):
                axs[i].imshow(self[i][0], cmap="gray")
                axs[i].axis("off")
            plt.tight_layout()
            if not os.path.exists(show_checkerboards):
                os.makedirs(show_checkerboards)
            plt.savefig(os.path.join(show_checkerboards, "checkerboard.png"))


    def generate_items(self):
        dataset = [self.method(n_grid_points=self.n_grid_points, noisy_points=self.noisy_points, device=self.device, num_squares=self.num_squares, seed=self.seeds[idx]) for idx in range(len(self))]
        dataset = torch.stack(dataset, dim=0).unsqueeze(1)
        return dataset

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        return self.items[idx]
    

def get_image_shape(modelConfig):
    if modelConfig["dataset"] == "checkerboard":
        return (1, modelConfig["img_size"], modelConfig["img_size"])
    elif modelConfig["dataset"].lower() == "mnist":
        return (1, 28, 28)
    elif modelConfig["dataset"].lower() == "cifar10":
        return (3, 32, 32)
    else:
        raise ValueError("Dataset not recognized. Please choose from checkerboard, MNIST, or CIFAR10.")
    

def create_dataset(modelConfig, return_loader=False):
    if modelConfig["dataset"] == "checkerboard":
        data = CheckerboardDataset(
            dataset_size=modelConfig["train_size"],
            n_grid_points=modelConfig["img_size"],
            noisy_points=int(modelConfig["noisy_points"] * modelConfig["img_size"] ** 2 / 2),
            device=modelConfig["device"],
            num_squares=modelConfig["checkerboard_squares"],
            method=checkerboard.METHODS[modelConfig["checkerboard_method"]] if modelConfig["checkerboard_method"] in range(len(checkerboard.METHODS)) else checkerboard.create_checkerboard,
            show_checkerboards=modelConfig['sampled_dir'] if modelConfig['show_original'] else None
        )

    elif modelConfig["dataset"].lower() == "mnist":
        data = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    elif modelConfig["dataset"].lower() == "cifar10":
        data = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip(),
        ]))

    else:
        raise ValueError("Dataset not recognized. Please choose from checkerboard, MNIST, or CIFAR10.")

    if return_loader:
        return torch.utils.data.DataLoader(data, batch_size=modelConfig["batch_size"], shuffle=True)
    return data
