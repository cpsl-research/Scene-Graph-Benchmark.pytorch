#!/usr/bin/env bash

set -e

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --master_port 10027 \
    --nproc_per_node=1 \
    tools/relation_test_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    --n_max 1 \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
    MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none \
    MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum \
    MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs \
    TEST.IMS_PER_BATCH 1 \
    DTYPE "float16" \
    GLOVE_DIR ./data/glove \
    MODEL.PRETRAINED_DETECTOR_CKPT ./models/pretrained_faster_rcnn/model_final.pth \
    OUTPUT_DIR ./models/motif-sgdet-pretrain \
    TEST.CUSTUM_EVAL True \
    TEST.CUSTUM_PATH /home/spencer/Documents/Projects/Research/attack-perception/4-nsai-via-sgg/submodules/Scene-Graph-Benchmark.pytorch/data/MOT15/train/ADL-Rundle-6/img1 \
    DETECTED_SGG_DIR ./test_output



    # TEST.CUSTUM_PATH /home/spencer/Documents/Projects/Research/attack-perception/4-nsai-via-sgg/submodules/Scene-Graph-Benchmark.pytorch/data/vg/Images/VG_100K \
