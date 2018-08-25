#!/bin/bash
#SBATCH --job-name=NN-FINN
#SBATCH -o  %J-22-1.out -e  %J-22-1.err
#NC="-c 1"
#NP="-n 1"
#REQUEST="--mem=4000"
#SBATCH --gres=gpu:K20Xm:1
#SBATCH -t 600

module load gcc/latest
module load nvidia/7.5
module load cudnn/7.5-v5

python train.py --function train_net --data_path dataset/ --save_dir models --batch_size 64 --lr 0.0001 --log_after 10 --cuda 1 --device 0
##python run_model.py resnet_models/model-559.pt test.tif
##python model.py