import torch
import torch.nn as nn
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np
from PIL import Image


from d3pm import *
from discrete_unet import *
from checkerboard import *
from dataset import create_dataset

def train(modelConfig):
    """
    Train and save a diffusion model based on the chosen dataset.
    """
    device = torch.device(modelConfig["device"])
    N = modelConfig["num_classes"]

    train_loader = create_dataset(modelConfig, return_loader=True)
    for x in train_loader:
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

        if loss_ema < best_loss:
            best_loss = loss_ema
            torch.save(model.state_dict(), modelConfig["save_weight_dir"] + f"ckpt_{epoch}_.pt")

        if modelConfig["show_process"] and (epoch % (modelConfig["epoch"] // 10) == 0 or epoch == modelConfig["epoch"] - 1):
            model.eval()
            with torch.no_grad():
                init_noise = torch.randint(0, N, (4, C, H, W)).to(device)
                images = d3pm.sample_with_image_sequence(init_noise, stride=40)
                gif = []
                for image in images:
                    x_as_image = make_grid(image.float() / (N - 1), nrow=4).permute(1, 2, 0).cpu().numpy()
                    img = (x_as_image * 255).astype(np.uint8)
                    gif.append(Image.fromarray(img))
                gif[0].save(modelConfig["sampled_dir"] + f"sampled_{epoch}.gif", save_all=True, append_images=gif[1:], duration=100, loop=0)
                last_image = gif[-1]
                last_image.save(modelConfig["sampled_dir"] + f"sampled_{epoch}.png")
            model.train()
        
    print("Training complete.")



def test(modelConfig):
    pass