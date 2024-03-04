


# CUDA_VISIBLE_DEVICES=0 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/llff_fixhrgenerateplane_stage2_pgamma6.0_trainingset_504x376_lpipsgan_wogpc_wvd_wdpc_beta/fern/infer/004000/train_image \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/fern/images_8


CUDA_VISIBLE_DEVICES=7 python evaluation_demo.py \
    --input_dir /home/zhongkaiwu/proj/Mimic3D/out/llff_fixhrgenerateplane_stage2_pgamma6.0_trainingset_504x376_lpipsgan_wogpc_wvd_wdpc_nan_noise_gain8_wmse_betasampling_tv1/flower/infer/001000/train_image \
    --gt_dir /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/flower/images_8 \

# # CUDA_VISIBLE_DEVICES=0 python evaluation_demo.py \
# #     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/llff_fixhrgenerateplane_stage2_pgamma6.0_trainingset_252x188_lpipsgan_wogpc_wvd_wdpc_wmse_betasampling_motionblur_ks9_lpips1_allonekernel/flower/infer/004200/train_image \
# #     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/flower/images_8 \


# CUDA_VISIBLE_DEVICES=0 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/llff_fixhrgenerateplane_stage2_pgamma6.0_trainingset_504x376_lpipsgan_wogpc_wvd_wdpc_beta/fortress/infer/004600/train_image \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/fortress/images_8 \


# CUDA_VISIBLE_DEVICES=0 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/llff_fixhrgenerateplane_stage2_pgamma6.0_trainingset_504x376_lpipsgan_wogpc_wvd_wdpc_beta/horns/infer/008800/train_image \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/horns/images_8 \


# CUDA_VISIBLE_DEVICES=0 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/llff_fixhrgenerateplane_stage2_pgamma6.0_trainingset_504x376_lpipsgan_wogpc_wvd_wdpc_beta/leaves/infer/004000/train_image  \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/leaves/images_8 \


# CUDA_VISIBLE_DEVICES=0 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/llff_fixhrgenerateplane_stage2_pgamma6.0_trainingset_504x376_lpipsgan_wogpc_wvd_wdpc_beta/orchids_g3.0/infer/004800/train_image  \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/orchids/images_8 \

# CUDA_VISIBLE_DEVICES=0 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/llff_fixhrgenerateplane_stage2_pgamma6.0_trainingset_504x376_lpipsgan_wogpc_wvd_wdpc_beta/room_onlygan_g3.0/infer/004400/train_image  \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/room/images_8 \

# CUDA_VISIBLE_DEVICES=0 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/llff_fixhrgenerateplane_stage2_pgamma6.0_trainingset_504x376_lpipsgan_wogpc_wvd_wdpc_beta/trex/infer/004400/train_image \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/trex/images_8 \



# CUDA_VISIBLE_DEVICES=0 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_fixhrgeneratedplane_stage2_pgamma3.0_wogpc_wvd_wdpc_trainiset_lpipsgan/chair/infer/003000/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_alpha/chair/test \
#     --use_mask \
#     --type blender \

# CUDA_VISIBLE_DEVICES=3 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_drums_ablation_onlyhrgeneratetriplane_stage2_pgamma3.0_wmse_trainiset_lpipsgan_wvd_wdpc_wogpc/drums/infer/005200/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_alpha/drums/test \
#     --whitebg \
#     --type blender \


# CUDA_VISIBLE_DEVICES=1 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_ablation_coarserefine_stage2_pgamma0.5_wmse_trainiset_lpipsgan_wvd_wdpc_wogpc_degradation/ficus_lpips0.2_gan1.0_mse1.0/infer/000200/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_alpha/ficus/test \
#     --use_mask \
#     --type blender \

# CUDA_VISIBLE_DEVICES=1 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_ablation_coarserefine_stage2_pgamma0.5_wmse_trainiset_lpipsgan_wvd_wdpc_wogpc_degradation/hotdog_lpips0.2_gan1.0_mse1.0/infer/000200/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_alpha/hotdog/test \
#     --use_mask \
#     --type blender &\

# CUDA_VISIBLE_DEVICES=7 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_fixhrgeneratedplane_stage2_pgamma3.0_wogpc_wvd_wdpc_trainiset_lpipsgan/lego/infer/006800/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_alpha/lego/test \
#     --use_mask \
#     --type blender &\

# CUDA_VISIBLE_DEVICES=1 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_ablation_coarserefine_stage2_pgamma0.5_wmse_trainiset_lpipsgan_wvd_wdpc_wogpc_degradation/materials_lpips0.2_gan1.0_mse1.0/infer/000200/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_alpha/materials/test \
#     --use_mask \
#     --type blender &\

# CUDA_VISIBLE_DEVICES=3 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_ablation_coarserefine_stage2_pgamma0.5_wmse_trainiset_lpipsgan_wvd_wdpc_wogpc_degradation/mic_lpips0.2_gan1.0_mse1.0/infer/000200/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_alpha/mic/test \
#     --use_mask \
#     --type blender &\


# CUDA_VISIBLE_DEVICES=4 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_ablation_coarserefine_stage2_pgamma0.5_wmse_trainiset_lpipsgan_wvd_wdpc_wogpc_degradation/ship_lpips0.2_gan1.0_mse1.0/infer/000200/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_alpha/ship/test \
#     --use_mask \
#     --type blender &\

# CUDA_VISIBLE_DEVICES=1 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_256perframe/chair/infer/000000/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_whitebg/chair/test \
#     --type blender &\

# CUDA_VISIBLE_DEVICES=1 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_256perframe/drums/infer/000000/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_whitebg/drums/test \
#     --type blender &\


# CUDA_VISIBLE_DEVICES=2 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_256perframe/ficus/infer/test/eval_image \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_whitebg/ficus/test \
#     --type blender &\

# CUDA_VISIBLE_DEVICES=3 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_256perframe/hotdog/infer/000000/eval_image \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_whitebg/hotdog/test \
#     --type blender &\

# CUDA_VISIBLE_DEVICES=4 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_256perframe/lego/infer/000000/eval_image \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_whitebg/lego/test \
#     --type blender &\

# CUDA_VISIBLE_DEVICES=5 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_256perframe/materials/infer/000000/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_whitebg/materials/test \
#     --type blender &\

# CUDA_VISIBLE_DEVICES=6 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_256perframe/mic/infer/000000/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_whitebg/mic/test \
#     --type blender &\

# CUDA_VISIBLE_DEVICES=6 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_256perframe/ship/infer/000000/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_whitebg/ship/test \
#     --type blender &\