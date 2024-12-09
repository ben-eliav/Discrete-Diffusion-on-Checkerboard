import numpy as np
import matplotlib.pyplot as plt
import torch

def inf_train_gen(n_grid_points: int = 128, batch_size: int = 200, device: str = "cpu", num_squares=4, seed: int = 0):
    assert n_grid_points % num_squares == 0, "number of grid points has to be divisible by num_squares"
    assert num_squares % 2 == 0, "num_squares has to be even"
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_grid_points = n_grid_points // num_squares
    x1 = torch.randint(low=0, high=n_grid_points * num_squares, size=(batch_size,), device=device)  # any random height on the checkerboard - (y)
    samples_x2 = torch.randint(low=0, high=n_grid_points, size=(batch_size,), device=device)  # sample an x value within a square
    x2 = (
        samples_x2
        + (num_squares-2) * n_grid_points  # moving to the final two columns of squares: (num_squares-1, num_squares)
        - torch.randint(low=0, high=num_squares//2, size=(batch_size,), device=device) * 2 * n_grid_points  # moving to any of the other pairs of columns
        + (torch.floor(x1 / n_grid_points) % 2) * n_grid_points  # if in an even numbered row of squares, move to the square to the right.
    )  # x2 is an (x) point that paired with x1 will always correspond to a point within a square that is "white" (assuming the diagonal is white).
    x_end = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1)  # why are we multiplying by 1.0?
    return x_end.long()

def create_checkerboard1(n_grid_points: int = 128, batch_size: int = 200, device: str = "cpu", num_squares=4, seed=0):
    """Creates a black image and adds white noise where white tiles would be"""
    generated = inf_train_gen(n_grid_points, batch_size, device, num_squares, seed)
    image = torch.zeros(n_grid_points, n_grid_points)
    image[generated[:, 0], generated[:, 1]] = 1
    return image

def create_checkerboard2(n_grid_points: int = 128, batch_size: int = 200, device: str = "cpu", num_squares=4, seed=0):
    """Creates a checkerboard and adds noise to white tiles"""
    generated = inf_train_gen(n_grid_points, batch_size, device, num_squares, seed)
    image = torch.zeros(n_grid_points, n_grid_points)
    # white out the squares that would be white in a checkerboard
    for i in range(num_squares):
        for j in range(num_squares):
            if (i + j) % 2 == 0:
                image[i * n_grid_points // num_squares:(i + 1) * n_grid_points // num_squares,
                      j * n_grid_points // num_squares:(j + 1) * n_grid_points // num_squares] = 1
    
    image[generated[:, 0], generated[:, 1]] = 0
    return image

def create_checkerboard3(n_grid_points: int = 128, batch_size: int = 200, device: str = "cpu", num_squares=4, seed=0):
    """Creates a checkerboard with a random switch between where white and black can be"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    p = torch.rand(2) > 0.5
    checkerboard2 = create_checkerboard2(n_grid_points, batch_size, device, num_squares, seed)
    if p[0]:
        checkerboard2 = torch.roll(checkerboard2, shifts=n_grid_points // num_squares, dims=1)
    checkerboard2 = p[1]*checkerboard2 + (~p[1]) * (1 - checkerboard2)
    return checkerboard2

def create_checkerboard4(n_grid_points: int = 128, batch_size: int = 200, device: str = "cpu", num_squares=8, seed=0):
    """Creates a checkerboard with a random switch between where white and black can be and a random number of squares in each row"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_squares = 2 ** np.random.randint(1, int(np.log2(num_squares))+1)
    return create_checkerboard3(n_grid_points, batch_size, device, num_squares, seed)

def create_checkerboard(*args, **kwargs):
    methods = [create_checkerboard1, create_checkerboard2, create_checkerboard3, create_checkerboard4]
    if kwargs.get("method") is not None and type(kwargs.get("method")) == int and 0 <= kwargs.get("method") < len(methods):
        return methods[kwargs.pop("method")](*args, **kwargs)
    else:
        return methods[np.random.randint(0, len(methods))](*args, **kwargs)
        

def display_checkerboards(n_grid_points: int = 256, batch_size: int = 200, device: str = "cpu", num_squares=8, seed=89):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        axs[i].imshow(create_checkerboard(n_grid_points, batch_size, device, num_squares, seed, method=i), cmap="gray")
        axs[i].axis("off")
    plt.tight_layout()
    plt.show()
        

if __name__ == "__main__":
    display_checkerboards()