#!/bin/bash
export PYTHONPATH=/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd /hpc_stor03/sjtu_home/zhengshun.xia/need/codes/emotion-recognition/emo2vec/scripts

dataset=IEMOCAP
export CUDA_VISIBLE_DEVICES=1
checkpoint=/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/emotion-recognition/emo2vec/scripts/premodel/base_libri.pt



for layer in {11..11}; do
    true_layer=$[$layer+1]
    if [ $dataset == 'IEMOCAP' ]; then
        python data2vec_speech_features.py  \
            /hpc_stor03/sjtu_home/zhengshun.xia/need/data/manifest \
            /hpc_stor03/sjtu_home/zhengshun.xia/need/codes/fairseq \
            --split=Session_all \
            --save-dir=/hpc_stor03/sjtu_home/zhengshun.xia/need/codes/emotion-recognition/emo2vec/data2vec/layer-${true_layer} \
            --checkpoint=$checkpoint \
            --layer=$layer 
done
