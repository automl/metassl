#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J MSSL_D_Cifar10_SimSiam
#SBATCH -t 10:00:00

pip list

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

python -m metassl.baselines.execute_pt_and_ft --gpu 0 \
       	--valid_size 0.1 \
	--seed 4 \
	--pt_learning_rate 0.06 \
       	--trial $EXPERIMENT_NAME \
	--exp_dir "/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10" \
	--pretrained /work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/$EXPERIMENT_NAME/$EXPERIMENT_NAME \
	--use_fix_aug_params --brightness_strength 1.0684649657099303 --contrast_strength 0.7830870112737627 --saturation_strength 0.06963168751907613 --hue_strength 0.08283567444533997

# --ft_brightness_strength 0.7269498100347377 --ft_contrast_strength 1.188342814448252 --ft_saturation_strength 0.490243242105925 --ft_hue_strength 0.16944211021863254
# --use_fix_aug_params_ft
# --do_weight_decay_annealing
