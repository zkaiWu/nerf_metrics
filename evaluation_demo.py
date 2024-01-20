

import argparse
import os
import torch
import nerf_metrics
import glob
import PIL.Image as Image
import numpy as np
import lpips

def parse_args():
    parser = argparse.ArgumentParser(description='Blender 64x64 to 256x256 using floyd')
    parser.add_argument('--input_dir', required=True, type=str, help='input image directory')
    parser.add_argument('--gt_dir', required=True, type=str, help='output image directory')
    parser.add_argument('--hold_out', type=int, default=8)
    parser.add_argument('--metrics', type=str, nargs='+', default=['ssim', 'lpips', 'psnr'])

    args = parser.parse_args()
    return args


def evaluate():
    args = parse_args()
    
    input_img_path_list = sorted(glob.glob(os.path.join(args.input_dir, '*.png')))
    gt_img_path_list = sorted(glob.glob(os.path.join(args.gt_dir, '*.png')))
    if args.hold_out > 0:
        input_img_path_list = [input_img_path_list[i] for i in range(len(input_img_path_list)) if i % args.hold_out == 0]
        gt_img_path_list = [gt_img_path_list[i] for i in range(len(gt_img_path_list)) if i % args.hold_out == 0]
    else:
        input_img_path_list = [input_img_path_list[14]]
        gt_img_path_list = [gt_img_path_list[14]]
    
    print('input image number: ', len(input_img_path_list))
    print('gt image number: ', len(gt_img_path_list))

    if len(input_img_path_list) != len(gt_img_path_list):
        raise ValueError('input image number is not equal to gt image number : {} vs {}'.format(len(input_img_path_list), len(gt_img_path_list)))
    psnr_list = []
    ssim_list = []
    lpips_list = []
    lpips_model = lpips.LPIPS(net='vgg').cuda()
    for img_path, gt_img_path in zip(input_img_path_list, gt_img_path_list):
        img = Image.open(img_path).convert('RGB')
        gt_img = Image.open(gt_img_path).convert('RGB')
        img.save('test1.png')
        gt_img.save('test2.png')
        # import pdb; pdb.set_trace()

        img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        img = img * 2 - 1.0
        gt_img = torch.tensor(np.array(gt_img)).permute(2, 0, 1).float() / 255.0
        gt_img = gt_img * 2 - 1.0
        if 'psnr' in args.metrics:
            psnr_list.append(nerf_metrics.psnr(img, gt_img))
        if 'ssim' in args.metrics:
            ssim_list.append(nerf_metrics.ssim(img, gt_img))
        if 'lpips' in args.metrics:
            lpips_list.append(nerf_metrics.cal_lpips(img, gt_img, lpips_model))

    # # max_idx = np.argmax(ssim_list)
    # ssim_list = np.array(ssim_list)
    # psnr_list = np.array(psnr_list)
    # lpips_list = np.array(lpips_list)
    # indices = np.argpartition(ssim_list, -3)[-3:]
    # topk_values = [indices]
    # print('psnr: ', np.mean(psnr_list[topk_values]))
    # print('ssim: ', np.mean(ssim_list[topk_values]))
    # print('lpips: ', np.mean(lpips_list[topk_values]))
    print('psnr: ', np.mean(psnr_list))
    print('ssim: ', np.mean(ssim_list))
    print('lpips: ', np.mean(lpips_list))


if __name__ == '__main__':
    evaluate()