#!/usr/bin/env bash

set -e 

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --master_port 10026 \
    --nproc_per_node=2 \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
    SOLVER.IMS_PER_BATCH 12 \
    TEST.IMS_PER_BATCH 2 \
    DTYPE "float16" \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.PRE_VAL False \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    GLOVE_DIR ./data/glove \
    MODEL.PRETRAINED_DETECTOR_CKPT ./models/pretrained_faster_rcnn/model_final.pth \
    OUTPUT_DIR ./models/motif-sgcls-exmp
