#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J MSSL_D_Cifar10_SimSiam
#SBATCH -t 19:59:00

pip list

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

python -m metassl.baselines.execute_pt_and_ft --gpu 0 --valid_size 0.0 --seed 1 --pt_learning_rate 0.06 --trial $EXPERIMENT_NAME --exp_dir "/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10" --pretrained /work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/$EXPERIMENT_NAME/$EXPERIMENT_NAME --do_weight_decay_annealing

# --use_fix_aug_params --use_fix_aug_params_ft --brightness_strength 0.7115759465642293 --contrast_strength 0.4024641067193702 --saturation_strength 0.2845185592463405 --hue_strength 0.04998358654852737 --ft_brightness_strength 0.7269498100347377 --ft_contrast_strength 1.188342814448252 --ft_saturation_strength 0.490243242105925 --ft_hue_strength 0.16944211021863254

