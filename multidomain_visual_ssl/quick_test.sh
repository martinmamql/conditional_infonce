#!/bin/bash
LR=$1
TEMP=$2
COND=$3
EPOCH=10
BS=960
DS=$4

if [ $COND == "uncond" ]
then
    COND_NAME="False"
    WEAK_NAME="False"
elif [ $COND == "cond" ]
then
    COND_NAME="True"
    WEAK_NAME="False"
elif [ $COND == "weak_cond" ]
then
    COND_NAME="True"
    WEAK_NAME="True"
fi
CKPT_PATH="/home/qianlim/checkpoint/multitask_cond_${COND_NAME}_weak_${WEAK_NAME}_simclr_bsz_${BS}_lr_${LR}_featdim_128_temp_${TEMP}_epoch_${EPOCH}_resnet50.pth"
echo $CKPT_PATH
#python -u main_multitask.py --data ~/dataset --checkpoint-dir ~/checkpoint --epochs ${EPOCH} --batch-size ${BS} --projector 2048-128 --learning-rate $LR --method simclr --temperature $TEMP --dist-address 58601 --$COND && 
python linear_for_small_datasets_multitask.py --data ~/dataset --model_path $CKPT_PATH --batch_size 512 --epochs 10 --dataset $DS
