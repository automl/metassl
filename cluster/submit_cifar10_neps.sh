#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J MSSL_C10_SimSiam
#SBATCH -t 20:00:00
#SBATCH --array 0-8%3

source activate metassl

python -m metassl.train_simsiam --config "metassl/default_metassl_config_cifar10.yaml" \
				--use_fixed_args \
				--data.dataset_percentage_usage 25 \
				--train.epochs 2 \
				--finetuning.epochs 2 \
				--expt.warmup_epochs 0 \
				--expt.seed 0 \
				--expt.save_model_frequency 50 \
				--expt.is_non_grad_based \
				--expt.multiprocessing_distributed \
				--model.arch "baseline_resnet" \
				--simsiam.use_baselines_loss \
				--neps.is_neps_run \
				--finetuning.valid_size 0.1 \
				--expt.expt_name $EXPERIMENT_NAME \
				--neps.config_space parameterized_cifar10_augmentation_with_solarize \
				# --neps.is_user_prior

