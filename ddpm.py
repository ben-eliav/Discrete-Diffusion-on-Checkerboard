import torch
import torch.nn as nn
import torch.nn.functional as F


def gather_reshape(v, t, x_shape):
    """
    :param v: tensor of parameters (such as beta)
    :param t: tensor of indices, for each element in batch has a corresponding index
    :param x_shape: shape of the output tensor (to easily find x_t)
    """
    out = torch.gather(v, index=t, dim=0).float().to(v.device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class DiffusionLoss(nn.Module):
    def __init__(self, T, model, beta_1, beta_T, device):
        super().__init__()
        self.T = T
        self.model = model
        self.register_buffer('beta', torch.linspace(beta_1, beta_T, T))
        a = 1 - self.beta
        alpha = torch.cumprod(a, 0)
        self.register_buffer('sqrt_alpha', alpha.sqrt())
        self.register_buffer('sqrt_1m_alpha', (1 - alpha).sqrt())
        self.device = device

    def forward(self, train_x):
        B, C, H, W = train_x.shape
        t = torch.randint(self.T, (B, )).to(self.device)  # time index is uniform (int)
        noise = torch.randn_like(train_x).to(self.device)  # epsilon
        x_t = gather_reshape(self.sqrt_alpha, t, train_x.shape) * train_x + gather_reshape(self.sqrt_1m_alpha, t, train_x.shape) * noise  # according to equation
        loss = F.mse_loss(self.model(x_t, t), noise)
        return loss


class DiffusionSampler(nn.Module):
    def __init__(self, T, model, beta_1, beta_T, device):
        super().__init__()
        self.T = T
        self.model = model
        self.device = device
        self.register_buffer('beta', torch.linspace(beta_1, beta_T, T))
        a = 1 - self.beta
        alpha = torch.cumprod(a, 0)
        alpha_padded = F.pad(alpha, [1, 0], value=1)[:T]
        self.register_buffer('coeff_prev', 1 / a.sqrt())
        self.register_buffer('coeff_noise', self.coeff_prev * (1 - a) / (1 - alpha).sqrt())
        self.register_buffer('posterior_var', self.beta * (1. - alpha_padded) / (1. - alpha))
        self.register_buffer('variance', torch.cat([self.posterior_var[1:2], self.beta[1:]]))


    def forward(self, x):
        for time in reversed(range(self.T)):
            if time == 0:
                noise = 0
            else:
                noise = torch.randn_like(x)
            t = x.new_ones(x.shape[0], dtype=int) * time
            mean = gather_reshape(self.coeff_prev, t, x.shape) * x - gather_reshape(self.coeff_noise, t, x.shape) * self.model(x, t)
            var = gather_reshape(self.variance, t, x.shape)
            x = mean + torch.sqrt(var) * noise
        return x
    
    def generate_visualize(self):
        x = torch.randn(4, 1, 64, 64).to(self.device)
        xs = [x.cpu().detach().numpy()]
        for time in reversed(range(self.T)):
            t = x.new_ones(x.shape[0], dtype=int) * time
            mean = gather_reshape(self.coeff_prev, t, x.shape) * x - gather_reshape(self.coeff_noise, t, x.shape) * self.model(x, t)
            var = gather_reshape(self.variance, t, x.shape)
            if time == 0:
                x = mean
            else:
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            if time % (self.T // 10) == 0:
                xs.append(x.cpu().detach().numpy())
        return xs
