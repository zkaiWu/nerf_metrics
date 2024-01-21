
import torch
from torchmetrics.functional import structural_similarity_index_measure
import PIL
import numpy as np
import pyiqa


def psnr(img_src:torch.Tensor, img_dst:torch.Tensor):

    """
        img_src: data range [-1, 1]
        img_dst: data range [-1, 1]
    Returns:
        a float of psnr value
    """
    img_src = (img_src + 1.0) / 2.0
    img_dst = (img_dst + 1.0) / 2.0
    mse = torch.mean((img_src - img_dst) ** 2).cpu()
    psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]))
    return psnr.item() 


def ssim(img_src:torch.Tensor, img_dst:torch.Tensor, mask=None):
    """
    Args:
        img_src (torch.Tensor): data range [-1, 1], shape [3, H, W] 
        img_dst (torch.Tensor): data range [-1, 1], shape [3, H, W]  
        mask (_type_, optional): _description_. Defaults to None.
    Returns:
        _type_: _description_
    """

    # img_src = img_src.permute(1, 2, 0).contiguous()
    # img_dst = img_dst.permute(1, 2, 0).contiguous()
    img_src = img_src.unsqueeze(0)
    img_dst = img_dst.unsqueeze(0)
    img_src = (img_src + 1.0) / 2.0
    img_dst = (img_dst + 1.0) / 2.0


    if mask is not None:
        img_src = img_src * mask + 1.0 * (1 - mask)
        img_dst = img_dst * mask + 1.0 * (1 - mask)

    #convert to PIL.Image and save
    # img_src_temp = img_src.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
    # img_dst_temp = img_dst.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
    # img_src_temp = (img_src_temp * 255).astype(np.uint8)
    # img_dst_temp = (img_dst_temp * 255).astype(np.uint8)
    # PIL.Image.fromarray(img_src_temp).save('img_src.png')
    # PIL.Image.fromarray(img_dst_temp).save('img_dst.png')

    # img_src = img_src.cpu().numpy()
    # img_dst = img_dst.cpu().numpy()
    # ssim_val = compare_ssim(img_src, img_dst, 
    #                         channel_axis=2, data_range=img_src.max() - img_src.min())
    # ssim_val = compare_ssim(img_src, img_dst, 
    #                         channel_axis=2, data_range=1.0)
    ssim_val = structural_similarity_index_measure(img_src, img_dst, gaussian_kernel=False, kernel_size=11).detach().cpu().numpy()

    return ssim_val


def cal_lpips(img_src:torch.Tensor, img_dst:torch.Tensor, lpips_model, mask=None):

    """
    Args:
        img_src (torch.Tensor): data range [-1, 1], shape [3, H, W] 
        img_dst (torch.Tensor): data range [-1, 1], shape [3, H, W]  
        mask (_type_, optional): _description_. Defaults to None.
    Returns:
        _type_: _description_
    """

    if mask is not None:
        img_src = (img_src + 1.0) / 2.0
        img_dst = (img_dst + 1.0) / 2.0
        img_src = img_src * mask + 1.0 * (1 - mask)
        img_dst = img_dst * mask + 1.0 * (1 - mask) 
        img_src = img_src * 2.0 - 1.0
        img_dst = img_dst * 2.0 - 1.0


    img_src_temp = (img_src.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()) / 2.0 + 0.5
    img_dst_temp = (img_dst.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()) / 2.0 + 0.5
    img_src_temp = (img_src_temp * 255).astype(np.uint8)
    img_dst_temp = (img_dst_temp * 255).astype(np.uint8)
    PIL.Image.fromarray(img_src_temp).save('img_src_lpips.png')
    PIL.Image.fromarray(img_dst_temp).save('img_dst_lpips.png')

    img_src = img_src.to('cuda')
    img_dst = img_dst.to('cuda')
    lpips_val = lpips_model(img_src, img_dst).detach().cpu().numpy()
    return lpips_val