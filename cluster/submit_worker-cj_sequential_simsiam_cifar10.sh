#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
##SBATCH -p testdlc_gpu-rtx2080
#SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J MSSL_W_Cifar10_SimSiam
#SBATCH -t 0-15:00 # time (D-HH:MM)
#SBATCH --array 0-500%8

pip list

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

python -m metassl.baselines.execute_pt_and_ft --gpu 0 --is_bohb_run --valid_size 0.1 --seed 1 --trial $EXPERIMENT_NAME --exp_dir "/work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10" --pretrained /work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10/$EXPERIMENT_NAME/$EXPERIMENT_NAME --n_iterations 250 --run_id "color_jitter_strengths" --configspace_mode 'color_jitter_strengths' --pt_learning_rate 0.06 --shutdown_workers --worker
