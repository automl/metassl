#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J MSSL_D_Cifar10_SimSiam
#SBATCH -t 19:59:00

pip list

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

python -m metassl.baselines.execute_pt_and_ft --gpu 0 --valid_size 0.0 --seed 1 --pt_learning_rate 0.06 --trial $EXPERIMENT_NAME --exp_dir "/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10" --pretrained /work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/$EXPERIMENT_NAME/$EXPERIMENT_NAME

