import torch
import torch.nn as nn
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np
from PIL import Image

from d3pm import *
from discrete_unet import *
from checkerboard import *
from dataset import create_dataset, get_image_shape
from utils import *


def sample(modelConfig, model, diffusion, device, shape, num_samples, N, train_epoch=None):
    """
    Sample images from the trained model.
    """
    model.eval()
    diffusion.eval()
    with torch.no_grad():
        init_noise = torch.randint(0, N, (num_samples, *shape)).to(device)
        images = diffusion.sample_with_image_sequence(init_noise, stride=40)
        gif = []
        for image in images:
            x_as_image = make_grid(image.float() / (N - 1), nrow=4).permute(1, 2, 0).cpu().numpy()
            img = (x_as_image * 255).astype(np.uint8)
            gif.append(Image.fromarray(img))
        last_image = gif[-1]
        if train_epoch is not None:
            gif[0].save(modelConfig["sampled_dir"] + f"{modelConfig['sampledImgName']}_{train_epoch}.gif", save_all=True, append_images=gif[1:], duration=100, loop=0)
            last_image.save(modelConfig["sampled_dir"] + f"{modelConfig['sampledImgName']}_{train_epoch}.png")
        gif[0].save(modelConfig["sampled_dir"] + f"{modelConfig['sampledImgName']}.gif", save_all=True, append_images=gif[1:], duration=100, loop=0)
        last_image.save(modelConfig["sampled_dir"] + f"{modelConfig['sampledImgName']}.png")


def train(modelConfig):
    """
    Train and save a diffusion model based on the chosen dataset.
    """
    device = torch.device(modelConfig["device"])
    N = modelConfig["num_classes"]

    train_loader = create_dataset(modelConfig, return_loader=True)
    for x in train_loader:
        if modelConfig['dataset'].lower() in ['mnist', 'cifar10']:
            x = x[0]
        _, C, H, W = x.shape
        break

    model = UNet(C, modelConfig["channel"], modelConfig["channel_mult"], modelConfig["attn"], modelConfig["num_res_blocks"], modelConfig["dropout"], N).to(device)
    if modelConfig["training_load_weight"]:
        model.load_state_dict(torch.load(modelConfig["training_load_weight"]))
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=modelConfig["lr"])

    """ For now, we will assume that the diffusion method is discrete (D3PM) """
    d3pm = D3PM(model, modelConfig["T"], N, hybrid_loss_coeff=0.0).to(device)
    d3pm.train()
    best_loss = float("inf")

    for epoch in range(modelConfig["epoch"]):
        loss_ema = None
        loading_bar = tqdm(train_loader)
        for x in loading_bar:
            if modelConfig['dataset'].lower() in ['mnist', 'cifar10']:
                x = x[0]
            optimizer.zero_grad()
            x = x.to(device)
            x = (x * (N - 1)).round().long().clamp(0, N - 1)
            loss, _ = d3pm(x)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), modelConfig["grad_clip"])
            optimizer.step()

            if loss_ema is None:
                loss_ema = loss.item()  # first iteration
            else:
                loss_ema = loss_ema * 0.99 + loss.item() * 0.01
            loading_bar.set_description(f"Epoch {epoch} --- loss: {loss_ema:.4f}, grad_norm: {norm:.4f}")

        if loss_ema < best_loss and modelConfig["save_weight_dir"] is not None:
            best_loss = loss_ema
            torch.save(model.state_dict(), modelConfig["save_weight_dir"] + f"ckpt_{epoch}.pt")

        if modelConfig["show_process"] and (epoch % (modelConfig["epoch"] // 10) == 0 or epoch == modelConfig["epoch"] - 1):
            assert modelConfig["sampled_dir"] is not None, "Provide a directory to save the sampled images."
            sample(modelConfig, model, d3pm, device, (C, H, W), 4, N, epoch)
            model.train()
        
    print("Training complete.")



def test(modelConfig):
    load_weight = find_file_with_largest_number(modelConfig["save_weight_dir"])
    device = torch.device(modelConfig["device"])
    N = modelConfig["num_classes"]
    C, H, W = get_image_shape(modelConfig)

    try:
        model = UNet(C, modelConfig["channel"], modelConfig["channel_mult"], modelConfig["attn"], modelConfig["num_res_blocks"], modelConfig["dropout"], N).to(device)
        model.load_state_dict(torch.load(load_weight, weights_only=True))
    except:
        print('No weights found. Use --state train to create weights.')
        return
    d3pm = D3PM(model, modelConfig["T"], N, hybrid_loss_coeff=0.0).to(device)    
    sample(modelConfig, model, d3pm, device, (C, H, W), 4, N)