import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image


def calculate_metrics(img1, img2, lpips_fn):

    img1_np = img1.detach().cpu().numpy().transpose(0, 2, 3, 1)
    img2_np = img2.detach().cpu().numpy().transpose(0, 2, 3, 1)
    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)

    batch_psnr, batch_ssim = 0, 0
    for i in range(img1_np.shape[0]):
        """计算psnr"""
        max_i = 1.0   #默认最大值为1.0，图像已经归一化
        mse_i = np.mean((img1_np[i] - img2_np[i]) ** 2)
        if mse_i <= 1e-10:
            batch_psnr += 100.0
        else:
            batch_psnr += 10.0 * np.log10(max_i**2 / mse_i)
        # batch_psnr += psnr(img1_np[i], img2_np[i], data_range=1.0)
        batch_ssim += ssim(img1_np[i], img2_np[i], data_range=1.0, multichannel=True, channel_axis=2, win_size=7)

    avg_psnr = batch_psnr / img1_np.shape[0]
    avg_ssim = batch_ssim / img1_np.shape[0]

    img1_lpips = img1.clone() * 2 - 1
    img2_lpips = img2.clone() * 2 - 1

    with torch.no_grad():
        lpips_score = lpips_fn(img1_lpips, img2_lpips).mean().item()

    return avg_psnr, avg_ssim, lpips_score


def save_checkpoint(state, folder="checkpoints", filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, f"{folder}/{filename}")


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def save_some_examples(model, loader, epoch, folder, device):
    import os
    model.eval()
    rainy, clean, filename = next(iter(loader))
    rainy, clean = rainy.to(device), clean.to(device)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with torch.no_grad():
        derained = model(rainy)

    comparison = torch.cat([rainy, derained, clean], dim=0)
    save_image(comparison, f"{folder}/comparison_epoch_{epoch}.png", nrow=rainy.shape[0])
    model.train()