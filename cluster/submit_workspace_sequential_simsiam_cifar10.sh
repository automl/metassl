#!/bin/bash
#SBATCH -p mlhiwi_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J MSSL_D_Cifar10_SimSiam
#SBATCH -t 1:00:00

pip list

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

python -m metassl.baselines.execute_pt_and_ft --gpu 0 \
       	--valid_size 0.0 \
	--seed 1 \
	--pt_learning_rate 0.06 \
       	--trial $EXPERIMENT_NAME \
	--exp_dir "/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10" \
	--pretrained /work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10/weight_decay_annealing/104-0-0/weight_decay_annealing/weight_decay_annealing \
	--do_weight_decay_annealing
	# --pretrained /work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/$EXPERIMENT_NAME/$EXPERIMENT_NAME \

# --use_fix_aug_params --brightness_strength 0.4247967568716101 --contrast_strength 1.1108046612978293 --saturation_strength 0.0019149552189590801 --hue_strength 0.01511293198806661

# --ft_brightness_strength 0.7269498100347377 --ft_contrast_strength 1.188342814448252 --ft_saturation_strength 0.490243242105925 --ft_hue_strength 0.16944211021863254
# --use_fix_aug_params_ft
# --do_weight_decay_annealing
