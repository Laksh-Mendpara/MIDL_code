import numpy as np
import cv2
import numpy as np
import cv2
import random
import string
import os
import PIL
#import tensorflow as tf
import torch
import warnings
warnings.filterwarnings("ignore")
import torch
import torchvision
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
#from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
#from torch.utils.data import random_split
# %matplotlib inline
import os
import torch
import torchvision
import os
import random
from sklearn.model_selection import train_test_split
#import tarfile
#from torchvision.datasets.utils import download_url
#from torch.utils.data import random_split
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import log10, sqrt
import numpy as np
from PIL import Image, ImageChops
import math
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch, torchvision
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from einops import rearrange, repeat
from functools import partial
from PIL import Image
import matplotlib.pyplot as plt
import math, os, copy
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
from PIL import Image
from SSIM_PIL import compare_ssim
import time

save_path1 = 'new_gen_image/generated_image_222.png'

'''
def SSIM(original, high):
    #print(original1)
    #print(high1)
    original = Image.fromarray(np.uint8(original))
    high = Image.fromarray(np.uint8(high))
    #print(type(original))
    #print(type(high))
    ssim = compare_ssim(original, high)
    #print(ssim)
    return ssim
'''

def SSIM(original, high):
    def ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    if not original.shape == high.shape:
        raise ValueError('Input images must have the same dimensions.')

    if original.ndim == 2:
        return ssim(original, high)
    elif original.ndim == 3:
        if original.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(original[:, :, i], high[:, :, i]))
            return np.array(ssims).mean()
        elif original.shape[2] == 1:
            return ssim(np.squeeze(original), np.squeeze(high))
    else:
        raise ValueError('Wrong input image dimensions.')


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def plot_losses(train_losses, val_losses, epoch):
    plt.figure(figsize=(15, 10))
    plt.plot(range(1, epoch + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epoch + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join("/home/dattatreyo/sr3_try/photo_output/try", "loss_curves.png"))
    plt.show()

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        # Input : tensor of value of coefficient alpha at specific step of diffusion process e.g. torch.Tensor([0.03])
        # Transform level of noise into representation of given desired dimension
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(nn.Linear(in_channels, out_channels*(1+self.use_affine_level)))

    def forward(self, x, noise_embed):
        noise = self.noise_func(noise_embed).view(x.shape[0], -1, 1, 1)
        if self.use_affine_level:
            gamma, beta = noise.chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + noise
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)

# Linear Multi-head Self-attention
class SelfAtt(nn.Module):
    def __init__(self, channel_dim, num_heads, norm_groups=32):
        super(SelfAtt,self).__init__()        
        self.groupnorm = nn.GroupNorm(norm_groups, channel_dim)
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(channel_dim, channel_dim, 1)

    def forward(self,x):
        b, c, h, w = x.size()
        x = self.groupnorm(x)
        qkv = rearrange(self.qkv(x), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        keys = F.softmax(keys, dim=-1)
        att = torch.einsum('bhdn,bhen->bhde', keys, values)
        out = torch.einsum('bhde,bhdn->bhen', att, queries)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.num_heads, h=h, w=w)

        return self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, 
                    num_heads=1, use_affine_level=False, norm_groups=32, att=True):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.att = att
        self.attn = SelfAtt(dim_out, num_heads=num_heads, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        y = self.block1(x)
        y = self.noise_func(y, time_emb)
        y = self.block2(y)
        x = y + self.res_conv(x)
        if self.att:
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channel=6, out_channel=3, inner_channel=32, norm_groups=32,
        channel_mults=[1, 2, 4, 8, 8], res_blocks=3, dropout=0, img_size=128):
        super().__init__()

        noise_level_channel = inner_channel
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Swish(), 
            nn.Linear(inner_channel * 4, inner_channel)
        )

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = img_size

        # Downsampling stage of U-net
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResBlock(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, 
                    norm_groups=norm_groups, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, 
                            norm_groups=norm_groups, dropout=dropout),
            ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, 
                        norm_groups=norm_groups, dropout=dropout, att=False)
        ])

        # Upsampling stage of U-net
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResBlock(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, 
                    norm_groups=norm_groups, dropout=dropout))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)

    def forward(self, x, noise_level):
        # Embedding of time step with noise coefficient alpha
        t = self.noise_level_mlp(noise_level)
        
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x, t)

        for layer in self.ups:
            if isinstance(layer, ResBlock):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)



class Diffusion(nn.Module):
    def __init__(self, model, device, img_size, LR_size, channels=3):
        super().__init__()
        self.channels = channels
        self.model = model.to(device)
        self.img_size = img_size
        self.LR_size = LR_size
        self.device = device

    def set_loss(self, loss_type):
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError()

    def make_beta_schedule(self, schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
        if schedule == 'linear':
            betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
        elif schedule == 'warmup':
            warmup_frac=0.1
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
            warmup_time = int(n_timestep * warmup_frac)
            betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
        elif schedule == "cosine":
            cosine_s = 8e-3
            timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(schedule)
        return betas

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        betas = self.make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        self.num_timesteps = int(len(betas))
        # Coefficient for forward diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('pred_coef1', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('pred_coef2', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # Coefficient for reverse diffusion posterior q(x_{t-1} | x_t, x_0)
        variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('variance', to_torch(variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # Predict desired image x_0 from x_t with noise z_t -> Output is predicted x_0
    def predict_start(self, x_t, t, noise):
        return self.pred_coef1[t] * x_t - self.pred_coef2[t] * noise

    # Compute mean and log variance of posterior(reverse diffusion process) distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    # Note that posterior q for reverse diffusion process is conditioned Gaussian distribution q(x_{t-1}|x_t, x_0)
    # Thus to compute desired posterior q, we need original image x_0 in ideal, 
    # but it's impossible for actual training procedure -> Thus we reconstruct desired x_0 and use this for posterior
    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        x_recon = self.predict_start(x, t, noise=self.model(torch.cat([condition_x, x], dim=1), noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return mean, posterior_log_variance

    # Progress single step of reverse diffusion process
    # Given mean and log variance of posterior, sample reverse diffusion result from the posterior
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        mean, log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return mean + noise * (0.5 * log_variance).exp()

    # Progress whole reverse diffusion process
    @torch.no_grad()
    def super_resolution(self, x_in):
        img = torch.rand_like(x_in, device=x_in.device)
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, i, condition_x=x_in)
        return img

    # Compute loss to train the model
    def p_losses(self, x_in):
        x_start = x_in
        lr_imgs = transforms.Resize(self.img_size)(transforms.Resize(self.LR_size)(x_in))
        b, c, h, w = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        sqrt_alpha = torch.FloatTensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[t-1], self.sqrt_alphas_cumprod_prev[t], size=b)
        ).to(x_start.device)
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1, 1)

        noise = torch.randn_like(x_start).to(x_start.device)
        # Perturbed image obtained by forward diffusion process at random time step t
        x_noisy = sqrt_alpha * x_start + (1 - sqrt_alpha**2).sqrt() * noise
        # The model predict actual noise added at time step t
        pred_noise = self.model(torch.cat([lr_imgs, x_noisy], dim=1), noise_level=sqrt_alpha)
        
        return self.loss_func(noise, pred_noise)

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)


# Class to train & test desired model
class SR3():
    def __init__(self, device, img_size, LR_size, loss_type, dataloader, testloader, 
                    schedule_opt, save_path, load_path=None, load=False, 
                    in_channel=6, out_channel=3, inner_channel=32, norm_groups=8, 
                    channel_mults=(1, 2, 4, 8, 8), res_blocks=3, dropout=0, lr=1e-5, distributed=False):
        super(SR3, self).__init__()
        self.dataloader = dataloader
        self.testloader = testloader
        self.device = device
        self.save_path = save_path
        self.img_size = img_size
        self.LR_size = LR_size

        model = UNet(in_channel, out_channel, inner_channel, norm_groups, channel_mults, res_blocks, dropout, img_size)
        self.sr3 = Diffusion(model, device, img_size, LR_size, out_channel)

        # Apply weight initialization & set loss & set noise schedule
        self.sr3.apply(self.weights_init_orthogonal)
        self.sr3.set_loss(loss_type)
        self.sr3.set_new_noise_schedule(schedule_opt)

        if distributed:
            assert torch.cuda.is_available()
            self.sr3 = nn.DataParallel(self.sr3)

        self.optimizer = torch.optim.Adam(self.sr3.parameters(), lr=lr)

        params = sum(p.numel() for p in self.sr3.parameters())
        print(f"Number of model parameters : {params}")

        if load:
            self.load(load_path)

    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    def train(self, epoch, verbose):
        t = next(iter(test_loader))
        #print(t)
        t1=t[0]
        #print(t1)
        fixed_imgs1 = copy.deepcopy(t1)
        fixed_imgs1 = fixed_imgs1[0].to(self.device)
        # Transform to low-resolution images
        fixed_imgs = transforms.Resize(self.img_size)(transforms.Resize(self.LR_size)(fixed_imgs1))
        train_losses = []
        val_losses = []
        training_starttime = time.time()
        box=t[1]
        #print(box)
        box = box.numpy()
        #print(box)
        print("hi")

        for i in range(epoch):
            #psnr_value=0.0
            #mse_value=0.0
            #ssim_value=0.0
            start_time = time.time()
            train_loss = 0
            for _, imgs in enumerate(self.dataloader):
                # Initial imgs are high-resolution
                #print(imgs)
                boxes = imgs[1].to(self.device)
                imgs = imgs[0].to(self.device)
                
                #print(boxes)
                
                b, c, h, w = imgs.shape
                # Extract and assign to different variable names
                x_min_new_name = boxes[0, 0, 0].item()
                y_min_new_name = boxes[0, 0, 1].item()
                x_max_new_name = boxes[0, 0, 2].item()
                y_max_new_name = boxes[0, 0, 3].item()
                '''
                # Print the values with the new variable names
                print("x_min_new_name:", x_min_new_name)
                print("y_min_new_name:", y_min_new_name)
                print("x_max_new_name:", x_max_new_name)
                print("y_max_new_name:", y_max_new_name)
                #print(boxes.shape)
                '''
                x_min_new_name = int(x_min_new_name)
                x_max_new_name = int(x_max_new_name)
                y_min_new_name = int(y_min_new_name)
                y_max_new_name = int(y_max_new_name)
            
                self.optimizer.zero_grad()
                if (x_max_new_name > x_min_new_name and y_max_new_name > y_min_new_name and
                                x_max_new_name <= w and x_min_new_name >= 0 and
                                y_max_new_name <= h and y_min_new_name >= 0):
                    #print("hiiiiiiii")
                    roi_imgs = imgs[:, :, x_min_new_name:x_max_new_name, y_min_new_name:y_max_new_name]
                    #print(roi_imgs)
                    roi_loss = self.sr3(roi_imgs)
                    roi_loss = roi_loss.sum() / int(b * c * h * w)
                alpha=0.2
                loss = self.sr3(imgs)
                loss = loss.sum() / int(b*c*h*w)

                total_loss = loss + alpha * roi_loss if (x_max_new_name > x_min_new_name and y_max_new_name > y_min_new_name and
                                                  x_max_new_name <= w and x_min_new_name >= 0 and
                                                  y_max_new_name <= h and y_min_new_name >= 0) else loss

                total_loss.backward()
                self.optimizer.step()
                train_loss += total_loss.item() * b

            self.sr3.eval()
            test_imgs = next(iter(self.testloader))
            test_imgs = test_imgs[0].to(self.device)
            b, c, h, w = test_imgs.shape

            with torch.no_grad():
                val_loss = self.sr3(test_imgs)
                val_loss = val_loss.sum() / int(b*c*h*w)
            self.sr3.train()

            train_loss = train_loss / len(self.dataloader)
            train_losses.append(train_loss)
            val_losses.append(val_loss.item())
            print(f'Epoch: {i+1} / loss:{train_loss:.3f} / val_loss:{val_loss.item():.3f}')

            os.makedirs("/home/dattatreyo/sr3_try/photo_output/originalnew", exist_ok=True)
            os.makedirs("/home/dattatreyo/sr3_try/photo_output/lownew", exist_ok=True)
            os.makedirs("/home/dattatreyo/sr3_try/photo_output/highnew", exist_ok=True)
            os.makedirs("/home/dattatreyo/sr3_try/loss_plots", exist_ok=True)

            for k in range(0,1):
                plt.figure(figsize=(20,15))
                plt.subplot(1,3,1)
                plt.axis("off")
                plt.title("Original Input")
                original=np.transpose(torchvision.utils.make_grid(fixed_imgs1[k:k+1], nrow=2, padding=1, normalize=True).cpu(),(1,2,0))
                plt.imshow(original)
                plt.savefig(os.path.join("/home/dattatreyo/sr3_try/photo_output/try", f"original_image_epoch_{i+1}_sample_{k}.png"))
                

                plt.subplot(1,3,2)
                plt.axis("off")
                plt.title("Low-Resolution Input")
                low=np.transpose(torchvision.utils.make_grid(fixed_imgs[k:k+1],nrow=2, padding=1, normalize=True).cpu(),(1,2,0))
                plt.imshow(low)
                plt.savefig(os.path.join("/home/dattatreyo/sr3_try/photo_output/try", f"low_resolution_image_epoch_{i+1}_sample_{k}.png"))

                plt.subplot(1,3,3)
                plt.axis("off")
                plt.title("Super-Resolution Result")
                high=np.transpose(torchvision.utils.make_grid(self.test(fixed_imgs[k:k+1]).detach().cpu(),nrow=2, padding=1, normalize=True),(1,2,0))
                plt.imshow(high)
                plt.savefig(os.path.join("/home/dattatreyo/sr3_try/photo_output/try", f"high_resolutionVINBIG_image_epoch_{i+1}_sample_{k}.png"))
                #plt.show()
                print(original.shape)
                original1 = original.numpy()
                print(low.shape)
                low1 = low.numpy()
                print(high.shape)
                high1 = high.numpy()
                mse_value = mse(original1, high1)
                psnr_value = PSNR(original1, high1)
                ssim_value = SSIM(original1, high1)
                print("\n")
                print("MSE between high resolution image and original image: ", mse_value)
                print("PSNR value between high resolution image and original image: ", (psnr_value - 50.0))
                print("SSIM value between super-resolution image and original input image: {:.4f}".format(round(ssim_value, 4)))
                plt.show()
            # Save model weight
            self.save(self.save_path)
            plot_losses(train_losses, val_losses, i + 1)
            end_time = time.time()
            execution_time_minutes = (end_time - start_time) / 60
            print("Execution Time: ", format(round(execution_time_minutes,2)), "minutes")

        print("\n")    
        print("\n")
        #plot_losses(train_losses, val_losses, i + 1)
        training_endtime = time.time()
        training_execution_time = training_endtime - training_starttime
        training_execution_time_minutes = training_execution_time / 60
        print("Training Execution Time:", format(round(training_execution_time_minutes,2)), "minutes")
        
    def test(self, imgs):
        imgs_lr = transforms.Resize(self.img_size)(transforms.Resize(self.LR_size)(imgs))
        self.sr3.eval()
        with torch.no_grad():
            if isinstance(self.sr3, nn.DataParallel):
                result_SR = self.sr3.module.super_resolution(imgs_lr)
            else:
                result_SR = self.sr3.super_resolution(imgs_lr)
        self.sr3.train()
        return result_SR

    def save(self, save_path):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self, load_path):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path))
        print("Model loaded successfully")

def add_artifacts(image, num_shapes):
    if image is None:
        print("Error: Unable to load the image.")
        return None
    image_with_artifacts = image.copy()
    height, width, channels = image_with_artifacts.shape

    for _ in range(num_shapes):
        # Randomly select a shape type
        shape_type = random.choice(['circle', 'rectangle', 'polygon', 'triangle', 'alphabet'])

        if shape_type == 'circle':
            x = np.random.randint(5, width - 10)  # Avoid edges
            y = np.random.randint(5, height - 10)
            radius = np.random.randint(5, 10)  # Limit size
            color = (255, 255, 255)
            thickness = 1
            cv2.circle(image_with_artifacts, (x, y), radius, color, thickness)

        elif shape_type == 'rectangle':
            x1 = np.random.randint(5, width - 15)
            y1 = np.random.randint(5, height - 15)
            x2 = x1 + np.random.randint(5, 15)
            y2 = y1 + np.random.randint(5, 15)
            color = (255, 255, 255)
            thickness = 1
            cv2.rectangle(image_with_artifacts, (x1, y1), (x2, y2), color, thickness)

        elif shape_type == 'polygon':
            num_vertices = np.random.randint(1, 2)
            vertices = [(np.random.randint(10, width - 10), np.random.randint(10, height - 10)) for _ in range(num_vertices)]
            vertices = np.array(vertices, np.int32).reshape((-1, 1, 2))
            color = (255, 255, 255)
            thickness = 1
            cv2.polylines(image_with_artifacts, [vertices], isClosed=True, color=color, thickness=thickness)


        elif shape_type == 'alphabet':
            font = cv2.FONT_HERSHEY_SIMPLEX
            alphabet = random.choice(string.ascii_uppercase)
            font_scale = np.random.uniform(0.25, 0.75)
            font_thickness = 2
            color = (255, 255, 255)
            x = np.random.randint(10, width - 30)
            y = np.random.randint(10, height - 30)
            cv2.putText(image_with_artifacts, alphabet, (x, y), font, font_scale, color, font_thickness)

    return image_with_artifacts

from torch.utils.data import Dataset
from pathlib import Path
import pydicom
from torchvision.datasets import DatasetFolder
from PIL import Image
from torchvision.transforms.functional import to_tensor
def load_dicom_as_image(file_path):
    dicom_data = pydicom.dcmread(file_path)
    image = dicom_data.pixel_array
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize the image
    image = (image * 255).astype(np.uint8)  # Convert the image to uint8
    image_pil = Image.fromarray(image)
    return image_pil

def load_dicom_as_tensor(file_path):
    dicom_data = pydicom.dcmread(file_path)
    image = dicom_data.pixel_array
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize the image
    image_tensor = torch.from_numpy(image)
    image_tensor = image_tensor.unsqueeze(0)  
    image_tensor = image_tensor.unsqueeze(0)  
    image_tensor = image_tensor.expand(-1, 3, -1, -1)  
    image_tensor = image_tensor.squeeze()  
    transform = transforms.Resize((256, 256))
    image_tensor = transform(image_tensor)

    image_pil = transforms.ToPILImage()(image_tensor)
    return image_pil


import random

class DICOMDataset(Dataset):
    def __init__(self, root, loader, csv_file, transform=None, num_artifacts=25, subset_percentage=0.5):
        self.root = root
        self.loader = loader
        self.transform = transform
        self.num_artifacts = num_artifacts
        self.subset_percentage = subset_percentage

        # Load the CSV file containing bounding box info
        try:
            self.csv_data = pd.read_csv(csv_file)
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")

        if 'image_id' not in self.csv_data.columns:
            raise ValueError("CSV file must contain 'image_id' column.")

        # List of image files in the root directory with the ".dicom" extension
        self.img_files = [file for file in os.listdir(self.root) if file.endswith('.dicom')]
        random.shuffle(self.img_files)

        # Calculate the size of the subset based on the specified percentage
        subset_size = int(len(self.img_files) * self.subset_percentage)
        # Select a random subset of image files
        self.img_files = self.img_files[:subset_size]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # Get the file path of the current image
        file_name = self.img_files[index]
        file_path = os.path.join(self.root, file_name)

        # Load the image using the provided loader function
        image_tensor = self.loader(file_path)

        # Find bounding box data for the current image index
        bbox_data = self.csv_data[self.csv_data.index == index]

        if bbox_data.empty:
            warnings.warn(f"No bounding box data found for image index: {index}")
            bounding_boxes = torch.zeros((0, 4), dtype=torch.float32)  # Empty bounding box tensor
        else:
            # Process bounding box data as needed (e.g., extract coordinates)
            bounding_boxes = bbox_data[['x_min', 'y_min', 'x_max', 'y_max']].values.astype(float)

        # Add artifacts to the image if specified
        if self.num_artifacts > 0:
            image_np = np.array(image_tensor)
            image_with_artifacts = add_artifacts(image_np, num_shapes=self.num_artifacts)
            image_tensor = Image.fromarray(image_with_artifacts)

        # Apply the specified transformation to the image if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, bounding_boxes, file_name, 0
        #return image_tensor, 0



if __name__ == "__main__":
    batch_size = 1
    LR_size = 64
    img_size = 256

    root = '/DATA2/VinDr-CXR/train'
    testroot = '/DATA2/VinDr-CXR/test'

    c1='/DATA2/VinDr-CXR/annotations/annotations_test.csv'
    c2='/DATA2/VinDr-CXR/annotations/annotations_train.csv'
    c3='/DATA2/VinDr-CXR/annotations/image_labels_test.csv'
    c4='/DATA2/VinDr-CXR/annotations/image_labels_train.csv'
    cc1=pd.read_csv(c1)
    #print(cc1.head(5))
    cc3=pd.read_csv(c3)
    #print(cc3.head(5))
    test_csv = pd.merge(cc1, cc3, on='image_id', how='inner')
    #print(test_csv.head(5))
    #print("Next")
    test_csv.to_csv('/home/dattatreyo/sr3_try/test_csv.csv', index=False)


    cc2=pd.read_csv(c2)
    #print(cc2.head(5))
    cc4=pd.read_csv(c4)
    #print(cc4.head(5))
    train_csv = pd.merge(cc1, cc3, on='image_id', how='inner')
    
    train_csv.to_csv('/home/dattatreyo/sr3_try/train_csv.csv', index=False)
    columns_to_replace = ['x_min', 'y_min', 'x_max', 'y_max']
    train_csv[columns_to_replace] = train_csv[columns_to_replace].fillna(0)
    test_csv[columns_to_replace] = test_csv[columns_to_replace].fillna(0)
    print(train_csv.head(5))
    train_csv.to_csv('modified_train.csv', index=False)
    test_csv.to_csv('modified_test.csv', index=False)

    source_dir_train = Path(root)
    source_dir_test = Path(testroot)

    imgtrain_files  = os.listdir(source_dir_train)
    train_files, val_files = train_test_split(imgtrain_files, test_size=0.2, random_state=42)
    imgtest_files =os.listdir(source_dir_test)

    # Define the transformations for the datasets
    transforms_ = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Create train and test datasets using the DICOMDataset class
    train_dataset = DICOMDataset(root=root, loader=load_dicom_as_tensor, csv_file='/home/dattatreyo/sr3_try/modified_train.csv', transform=transforms_, subset_percentage=0.025)
    test_dataset = DICOMDataset(root=testroot, loader=load_dicom_as_tensor, csv_file='/home/dattatreyo/sr3_try/modified_test.csv', transform=transforms_, subset_percentage=0.025)

    # Create DataLoaders for training and testing
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

    #print("Number of samples in the train loader:", len(dataloader.dataset))
    #print("Number of samples in the test loader:", len(test_loader.dataset))

    #imgs, _ = next(iter(dataloader))
    #print(imgs)
    
    data_item = next(iter(dataloader))
    #print(data_item[0])
    imgs = data_item[0]
    t = next(iter(test_loader))
    #print(t)
    fixed_imgs1 = copy.deepcopy(t)
    #print(fixed_imgs1)
    #t1=t[0]
    #print(t1)
    LR_imgs = transforms.Resize(img_size)(transforms.Resize(LR_size)(imgs))
    plt.figure(figsize=(15,10))
    plt.subplot(1,3,1)
    plt.axis("off")
    plt.title("Original Image")
    original=np.transpose(torchvision.utils.make_grid(imgs[:1], padding=1, normalize=True).cpu(),(1,2,0))
    plt.imshow(original)
    
    #plt.savefig(os.path.join("/home/dattatreyo/sr3_try/photo_output/try", "original_image.png"))

    plt.subplot(1,3,2)
    plt.axis("off")
    plt.title("Low-Resolution Image")
    low=np.transpose(torchvision.utils.make_grid(LR_imgs[:1], padding=1, normalize=True).cpu(),(1,2,0))
    plt.imshow(low)
    #plt.savefig(os.path.join("/home/dattatreyo/sr3_try/photo_output/try", "low_resolution_image.png"))

    plt.subplot(1,3,3)
    plt.axis("off")
    plt.title("High-Resolution Image")
    high=np.transpose(torchvision.utils.make_grid(imgs[:1], padding=1, normalize=True).cpu(),(1,2,0))
    plt.imshow(high)
    plt.show()
    #plt.savefig(os.path.join("/home/dattatreyo/sr3_try/photo_output/try", "high_resolution_image.png"))
    
    original1 = original.numpy()
    low1 = low.numpy()
    high1 = high.numpy()
    mse_value = mse(original1, high1)
    psnr_value = PSNR(original1, high1)
    ssim_value = SSIM(original1, high1)

    #print("MSE between high resolution image and original image: ", mse_value)
    #print("PSNR value between high resolution image and original image: ", psnr_value)
    #print("SSIM value between high resolution image and original input image: ", ssim_value)


    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    #print(device)
    schedule_opt = {'schedule':'linear', 'n_timestep':1000, 'linear_start':1e-4, 'linear_end':0.05}

    sr3 = SR3(device, img_size=img_size, LR_size=LR_size, loss_type='l1', 
                dataloader=dataloader, testloader=test_loader, schedule_opt=schedule_opt, 
                save_path='bb64_256.pt',load_path='bb64_256.pt' ,load=False, inner_channel=96, 
                norm_groups=16, channel_mults=(1, 2, 2, 2), dropout=0.2, res_blocks=2, lr=1e-5, distributed=True)
            

    print("START TEST")            
    sr3.train(epoch=250, verbose=1)
    #sr3.test(LR_imgs, save_path1)
    print("END TEST")