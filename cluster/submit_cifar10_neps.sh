#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J C10_Combined
#SBATCH -t 23:59:59
#SBATCH --array 0-199%10

source activate metassl

python -m metassl.train_simsiam --config "metassl/default_metassl_config_cifar10.yaml" \
				--use_fixed_args \
				--data.dataset_percentage_usage 100 \
				--train.epochs 800 \
				--finetuning.epochs 100 \
				--expt.warmup_epochs 0 \
				--expt.seed 0 \
				--expt.save_model_frequency 50 \
				--expt.is_non_grad_based \
				--expt.multiprocessing_distributed \
				--neps.is_neps_run \
				--finetuning.valid_size 0.2 \
				--expt.expt_name $EXPERIMENT_NAME \
				--neps.config_space combined \
				--neps.optimize_backbone_only \
				--neps.is_user_prior
