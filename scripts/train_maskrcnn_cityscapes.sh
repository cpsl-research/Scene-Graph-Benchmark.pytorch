#!/usr/bin/env bash

set -e 

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --master_port 10001 \
    --nproc_per_node=2 \
    tools/detector_pretrain_net.py \
    --config-file "configs/cityscapes/e2e_mask_rcnn_R_50_FPN_1x_from_COCO.yaml" \
    SOLVER.IMS_PER_BATCH 6 \
    TEST.IMS_PER_BATCH 4 \
    SOLVER.VAL_PERIOD 500 \
    SOLVER.CHECKPOINT_PERIOD 500 \
    MODEL.RELATION_ON False \
    SOLVER.PRE_VAL False

    # --config-file "configs/cityscapes/e2e_mask_rcnn_X_101_32_8_FPN_1x_from_COCO.yaml" \