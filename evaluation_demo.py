

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
    parser.add_argument('--input_dir', required=True, type=str, help='input image directory')
    parser.add_argument('--gt_dir', required=True, type=str, help='output image directory')
    parser.add_argument('--hold_out', type=int, default=8)
    parser.add_argument('--metrics', type=str, nargs='+', default=['ssim', 'lpips', 'psnr', 'liqe', 'maniqa', 'nima', 'brisque', 'niqe'])
    parser.add_argument('--type', type=str, default='llff')
    parser.add_argument('--whitebg', action='store_true')
    parser.add_argument('--use_mask', action='store_true')

    args = parser.parse_args()
    return args


def evaluate():
    args = parse_args()
    

    if args.type == 'blender':
        input_img_path_list = sorted(glob.glob(os.path.join(args.input_dir, '*.png')), key=lambda x: int(os.path.basename(x).split('.')[0].split('-')[0]))
        gt_img_path_list = sorted(glob.glob(os.path.join(args.gt_dir, '*.png')), key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
    elif args.type == 'llff':
        input_img_path_list = sorted(glob.glob(os.path.join(args.input_dir, '*.png')), key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[0]))
        gt_img_path_list = sorted(glob.glob(os.path.join(args.gt_dir, '*.png')))

    # input_img_path_list = sorted(glob.glob(os.path.join(args.input_dir, '*_pred_fine.png')))
    # gt_img_path_list = sorted(glob.glob(os.path.join(args.gt_dir, '*.png')))

    if args.hold_out > 0:
        input_img_path_list = [input_img_path_list[i] for i in range(len(input_img_path_list)) if i % args.hold_out == 0]
        gt_img_path_list = [gt_img_path_list[i] for i in range(len(gt_img_path_list)) if i % args.hold_out == 0]
    else:
        input_img_path_list = [input_img_path_list[14]]
        gt_img_path_list = [gt_img_path_list[14]]

    print(input_img_path_list)
    print(gt_img_path_list)
    
    print('input image number: ', len(input_img_path_list))
    print('gt image number: ', len(gt_img_path_list))

    if len(input_img_path_list) != len(gt_img_path_list):
        raise ValueError('input image number is not equal to gt image number : {} vs {}'.format(len(input_img_path_list), len(gt_img_path_list)))
    psnr_list = []
    ssim_list = []
    lpips_list = []
    liqe_list = []
    maniqa_list = []
    nima_list = []
    brsique_list = []
    niqe_list = []
    lpips_model = lpips.LPIPS(net='vgg').cuda()
    liqe_model = pyiqa.create_metric('liqe', as_loss=False).cuda()
    maniqa = pyiqa.create_metric('maniqa', as_loss=False).cuda()
    nima = pyiqa.create_metric('nima', as_loss=False).cuda()
    brsique = pyiqa.create_metric('brisque', as_loss=False).cuda()
    niqe = pyiqa.create_metric('niqe', as_loss=False).cuda()
    for img_path, gt_img_path in tqdm(zip(input_img_path_list, gt_img_path_list)):
        img = Image.open(img_path).convert('RGB').resize((256, 256))
        gt_img = Image.open(gt_img_path).convert('RGB').resize((256, 256))
        mask = None
        print(f'gt mode : {gt_img.mode}')
        if gt_img.mode == 'RGBA':
            if args.use_mask:
                alpha = gt_img.split()[3]
                mask = alpha.point(lambda i: i > 0 and 255)
                mask = mask.resize((64, 64), Image.BICUBIC)
                mask = mask.resize((256, 256), Image.BICUBIC)
                # mask = mask.resize((512, 512), Image.BICUBIC)
                mask = mask.point(lambda i: 255 if i > 0 else 0)
                mask = torch.tensor(np.array(mask)).unsqueeze(0)
                # import pdb; pdb.set_trace()
                mask = mask / 255.0
                # mask = torch.tensor(np.array(mask)).permute(2, 0, 1).float() / 255.0
            if args.whitebg:
                white_bkgb = Image.new('RGB', gt_img.size, (255, 255, 255, 255))
                white_bkgb.paste(gt_img, mask=gt_img.split()[3])
                gt_img = white_bkgb.convert('RGB')
            else:
                gt_img = gt_img.convert('RGB')
        else:
            gt_img = gt_img.convert('RGB')
            if args.use_mask:
                raise ValueError('mask is not available when no alpha is provided')

        img.save('test1.png')
        gt_img.save('test2.png')
        # import pdb; pdb.set_trace()

        img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        gt_img = torch.tensor(np.array(gt_img)).permute(2, 0, 1).float() / 255.0

        img = img * 2 - 1.0
        gt_img = gt_img * 2 - 1.0
        if 'psnr' in args.metrics:
            psnr_list.append(nerf_metrics.psnr(img, gt_img))
        if 'ssim' in args.metrics:
            ssim_list.append(nerf_metrics.ssim(img, gt_img, mask))
        if 'lpips' in args.metrics:
            lpips_list.append(nerf_metrics.cal_lpips(img, gt_img, lpips_model, mask))
        if 'liqe' in args.metrics:
            if args.type == 'llff':
                img_temp = Image.open(img_path).resize((504, 378)).save('temp.png')
                liqe_list.append(liqe_model('temp.png').cpu().numpy())
            else:
                print('aaa')
                liqe_list.append(liqe_model(img_path).cpu().numpy())
        if 'maniqa' in args.metrics:
            maniqa_list.append(maniqa(img_path).cpu().numpy())
        if 'nima' in args.metrics:
            nima_list.append(nima(img_path).cpu().numpy())
        if 'brisque' in args.metrics:
            brsique_list.append(brsique(img_path).cpu().numpy())
        if 'niqe' in args.metrics:
            niqe_list.append(niqe(img_path).cpu().numpy())

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
    print('liqe', np.mean(liqe_list))
    print('maniqa', np.mean(maniqa_list))
    print('nima', np.mean(nima_list))
    print('brisque', np.mean(brsique_list))
    print('niqe', np.mean(niqe_list))
    with open(os.path.join(args.input_dir, 'metrics.txt'), 'w') as f:
        f.write('psnr: {}\n'.format(np.mean(psnr_list)))
        f.write('ssim: {}\n'.format(np.mean(ssim_list)))
        f.write('lpips: {}\n'.format(np.mean(lpips_list)))
        f.write('liqe: {}\n'.format(np.mean(liqe_list)))
        f.write('maniqa: {}\n'.format(np.mean(maniqa_list)))
        f.write('nima: {}\n'.format(np.mean(nima_list)))
        f.write('brisque: {}\n'.format(np.mean(brsique_list)))
        f.write('niqe: {}\n'.format(np.mean(niqe_list)))


if __name__ == '__main__':
    evaluate()