

import argparse
import os
import torch
import nerf_metrics
import glob
import PIL.Image as Image
import numpy as np
import lpips
import pyiqa
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Blender 64x64 to 256x256 using floyd')
    parser.add_argument('--input_image_path_1', required=True, type=str, help='input image directory')
    parser.add_argument('--input_image_path_2', required=True, type=str, help='input image directory')
    parser.add_argument('--gt_image_path', required=True, type=str, help='output image directory')
    parser.add_argument('--type', type=str, default='llff')

    args = parser.parse_args()
    return args


def evaluate():
    args = parse_args()
    lpips_model = lpips.LPIPS(net='vgg').cuda()

    liqe_model = pyiqa.create_metric('liqe', as_loss=False).cuda()
    maniqa_model = pyiqa.create_metric('maniqa', as_loss=False).cuda()
    nima_model = pyiqa.create_metric('nima', as_loss=False).cuda()

    img_1 = Image.open(args.input_image_path_1).convert('RGB').resize((256, 256))
    img_2 = Image.open(args.input_image_path_2).convert('RGB').resize((256, 256))
    gt_img = Image.open(args.gt_image_path)
    mask = None
    if gt_img.mode == 'RGBA':
        mask = gt_img.split()[3]
        white_background = Image.new('RGBA', gt_img.size, (255, 255, 255, 255))
        white_background.paste(gt_img, mask=mask)
        gt_img = white_background.convert('RGB')
        mask = mask.resize((64, 64)).resize((256, 256))
        gt_img = gt_img.convert('RGB').resize((256, 256))
        mask = mask.point(lambda i: 255 if i > 0 else 0)

    img_1 = torch.tensor(np.array(img_1)).permute(2, 0, 1).float() / 255.0
    img_2 = torch.tensor(np.array(img_2)).permute(2, 0, 1).float() / 255.0
    gt_img = torch.tensor(np.array(gt_img)).permute(2, 0, 1).float() / 255.0
    if mask is not None:
        mask = torch.tensor(np.array(mask)).unsqueeze(0).float() / 255.0
    if mask is not None:
        img_1 = img_1 * mask + 1.0 * (1 - mask)
        img_2 = img_2 * mask + 1.0 * (1 - mask)
        gt_img = gt_img * mask + 1.0 * (1 - mask)
    img_1 = img_1 * 2 - 1.0
    img_2 = img_2 * 2 - 1.0
    gt_img = gt_img * 2 - 1.0
    
    psnr_1 = nerf_metrics.psnr(img_1, gt_img)
    psnr_2 = nerf_metrics.psnr(img_2, gt_img)

    lpips_1 = nerf_metrics.cal_lpips(img_1, gt_img, lpips_model)
    lpips_2 = nerf_metrics.cal_lpips(img_2, gt_img, lpips_model)

    liqe_value_1 = liqe_model(args.input_image_path_1)
    liqe_value_2 = liqe_model(args.input_image_path_2)

    maniqa_value_1 = maniqa_model(args.input_image_path_1)
    maniqa_value_2 = maniqa_model(args.input_image_path_2)


    nima_value_1 = nima_model(args.input_image_path_1)
    nima_value_2 = nima_model(args.input_image_path_2)

    print('psnr_1: ', psnr_1)
    print('psnr_2: ', psnr_2)
    print('lpips_1: ', lpips_1)
    print('lpips_2: ', lpips_2)
    print('liqe_1: ', liqe_value_1)
    print('liqe_2: ', liqe_value_2)
    print('maniqa_1: ', maniqa_value_1)
    print('maniqa_2: ', maniqa_value_2)
    print('nima_1: ', nima_value_1)
    print('nima_2: ', nima_value_2)

    # if args.type == 'blender':
    #     input_img_path_list = sorted(glob.glob(os.path.join(args.input_dir, '*.png')), key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[0]))
    #     gt_img_path_list = sorted(glob.glob(os.path.join(args.gt_dir, '*.png')), key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
    # elif args.type == 'llff':
    #     input_img_path_list = sorted(glob.glob(os.path.join(args.input_dir, '*.png')))
    #     gt_img_path_list = sorted(glob.glob(os.path.join(args.gt_dir, '*.png')))



    # print('psnr: ', np.mean(psnr_list))
    # print('ssim: ', np.mean(ssim_list))
    # print('lpips: ', np.mean(lpips_list))
    # print('liqe', np.mean(liqe_list))
    # print('maniqa', np.mean(maniqa_list))
    # print('nima', np.mean(nima_list))
    # print('brisque', np.mean(brsique_list))
    # print('niqe', np.mean(niqe_list))
    # with open(os.path.join(args.input_dir, 'metrics.txt'), 'w') as f:
    #     f.write('psnr: {}\n'.format(np.mean(psnr_list)))
    #     f.write('ssim: {}\n'.format(np.mean(ssim_list)))
    #     f.write('lpips: {}\n'.format(np.mean(lpips_list)))
    #     f.write('liqe: {}\n'.format(np.mean(liqe_list)))
    #     f.write('maniqa: {}\n'.format(np.mean(maniqa_list)))
    #     f.write('nima: {}\n'.format(np.mean(nima_list)))
    #     f.write('brisque: {}\n'.format(np.mean(brsique_list)))
    #     f.write('niqe: {}\n'.format(np.mean(niqe_list)))


if __name__ == '__main__':
    evaluate()