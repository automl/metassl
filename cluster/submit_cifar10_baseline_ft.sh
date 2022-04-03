#!/bin/bash
#SBATCH -p testdlc_gpu-rtx2080  # mlhiwidlc_gpu-rtx2080
#SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J MSSL_D_Cifar10_SimSiam_FT
#SBATCH -t 00:30:00

source activate new_metassl

python -m metassl.train_linear_classifier_simsiam --config "metassl/default_metassl_config_cifar10.yaml" \
						  --use_fixed_args \
						  --data.dataset_percentage_usage 100 \
						  --finetuning.epochs 100 \
						  --expt.warmup_epochs 0 \
						  --expt.seed $SEED \
						  --expt.save_model_frequency 10 \
						  --expt.is_non_grad_based \
						  --expt.multiprocessing_distributed \
						  --expt.expt_name $EXPERIMENT_NAME \
						  --expt.ssl_model_checkpoint_path "/work/dlclarge2/wagnerd-metassl-experiments/metassl/CIFAR10/$EXPERIMENT_NAME/checkpoint_0799.pth.tar"
