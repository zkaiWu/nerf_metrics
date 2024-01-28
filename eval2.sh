
CIDX=$1

CUDA_VISIBLE_DEVICES=$CIDX python evaluation_demo.py \
    --input_dir /home/zhongkaiwu/proj/Mimic3D/out/llff_fixhrgenerateplane_stage2_pgamma6.0_trainingset_504x376_lpipsgan_wogpc_wvd_wdpc_degradation_wmse_betasampling/fern/infer/003800/train_image \
    --gt_dir  /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/fern/images_8 \


CUDA_VISIBLE_DEVICES=$CIDX python evaluation_demo.py \
    --input_dir  \
    --gt_dir /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/flower/images_8 \

CUDA_VISIBLE_DEVICES=$CIDX python evaluation_demo.py \
    --input_dir /home/zhongkaiwu/proj/Mimic3D/out/llff_fixhrgenerateplane_stage2_pgamma6.0_trainingset_504x376_lpipsgan_wogpc_wvd_wdpc_degradation_wmse_betasampling/horns/infer/003400/train_image \
    --gt_dir /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/fortress/images_8 \


CUDA_VISIBLE_DEVICES=$CIDX python evaluation_demo.py \
    --input_dir /home/zhongkaiwu/proj/Mimic3D/out/llff_fixhrgenerateplane_stage2_pgamma6.0_trainingset_504x376_lpipsgan_wogpc_wvd_wdpc_degradation_wmse_betasampling/leaves_g6.0_tv1ckpt_lpips1.0/infer/008800/train_image \
    --gt_dir /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/horns/images_8 \


CUDA_VISIBLE_DEVICES=$CIDX python evaluation_demo.py \
    --input_dir /home/zhongkaiwu/proj/Mimic3D/out/llff_fixhrgenerateplane_stage2_pgamma6.0_trainingset_504x376_lpipsgan_wogpc_wvd_wdpc_beta/leaves/infer/004000/train_image \
    --gt_dir /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/leaves/images_8 \


CUDA_VISIBLE_DEVICES=$CIDX python evaluation_demo.py \
    --input_dir /home/zhongkaiwu/proj/Mimic3D/out/llff_fixhrgenerateplane_stage2_pgamma6.0_trainingset_504x376_lpipsgan_wogpc_wvd_wdpc_degradation_wmse_betasampling/orchids_g5.0_tv1ckpt/infer/004000/train_image  \
    --gt_dir /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/orchids/images_8 \


CUDA_VISIBLE_DEVICES=$CIDX python evaluation_demo.py \
    --input_dir /home/zhongkaiwu/proj/Mimic3D/out/llff_fixhrgenerateplane_stage2_pgamma6.0_trainingset_504x376_lpipsgan_wogpc_wvd_wdpc_degradation_wmse_betasampling/room/infer/004400/train_image  \
    --gt_dir /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/room/images_8 \

CUDA_VISIBLE_DEVICES=$CIDX python evaluation_demo.py \
    --input_dir /home/zhongkaiwu/proj/Mimic3D/out/llff_fixhrgenerateplane_stage2_pgamma6.0_trainingset_504x376_lpipsgan_wogpc_wvd_wdpc_degradation_wmse_betasampling/trex/infer/003400/train_image  \
    --gt_dir /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/trex/images_8 \

# CUDA_VISIBLE_DEVICES=0 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/NeRF-SR/checkpoints/nerf-sr-refine/fern \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/llff/nerf_llff_data_252x188_gt/trex/images_8 \



# CUDA_VISIBLE_DEVICES=0 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_fixhrgeneratedplane_stage2_pgamma3.0_wogpc_wvd_wdpc_trainiset_lpipsgan/chair/infer/003000/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_whitebg/chair/test \
#     --type blender &\

# CUDA_VISIBLE_DEVICES=1 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_fixhrgeneratedplane_stage2_pgamma3.0_wogpc_wvd_wdpc_trainiset_lpipsgan/drums/infer/007200/eval_image \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_whitebg/drums/test \
#     --type blender &\
#     --type blender &\


# CUDA_VISIBLE_DEVICES=2 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_fixhrgeneratedplane_stage2_pgamma3.0_wogpc_wvd_wdpc_trainiset_lpipsgan/ficus/infer/003400/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_whitebg/ficus/test \
#     --type blender &\

# CUDA_VISIBLE_DEVICES=3 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_fixhrgeneratedplane_stage2_pgamma3.0_wogpc_wvd_wdpc_trainiset_lpipsgan/hotdog/infer/004800/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_whitebg/hotdog/test \
#     --type blender &\

# CUDA_VISIBLE_DEVICES=4 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_fixhrgeneratedplane_stage2_pgamma3.0_wogpc_wvd_wdpc_trainiset_lpipsgan/lego/infer/006800/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_whitebg/lego/test \
#     --type blender &\

# CUDA_VISIBLE_DEVICES=5 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_fixhrgeneratedplane_stage2_pgamma3.0_wogpc_wvd_wdpc_trainiset_lpipsgan/materials/infer/003600/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_whitebg/materials/test \
#     --type blender &\

# CUDA_VISIBLE_DEVICES=6 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_fixhrgeneratedplane_stage2_pgamma3.0_wogpc_wvd_wdpc_trainiset_lpipsgan/mic/infer/003400/eval_image_res256 \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_whitebg/mic/test \
#     --type blender &\



# CUDA_VISIBLE_DEVICES=7 python evaluation_demo.py \
#     --input_dir /home/zhongkaiwu/proj/Mimic3D/out/blender_256perframe/drums/infer/000000/eval_image \
#     --gt_dir /home/zhongkaiwu/data/dreamfusion_data/blender_origin/nerf_synthetic_256_testset_whitebg/drums/test \
#     --type blender &\

# CUDA_VISIBLE_DEVICES=0 python evaluation_demo.py \
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