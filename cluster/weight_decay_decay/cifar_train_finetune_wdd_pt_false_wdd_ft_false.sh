#!/bin/bash
python metassl/train_simsiam.py --expt.expt_name $1 --train.epochs $2 --finetuning.epochs $3 --expt.warmup_epochs $4 --config $5 --expt.expt_mode "CIFAR10" --data.dataset "CIFAR10"  --use_fixed_args --expt.run_knn_val
python metassl/train_linear_classifier_simsiam.py --expt.expt_name $1 --train.epochs $2 --finetuning.epochs $3 --expt.warmup_epochs $4 --config $5 --expt.expt_mode "CIFAR10" --data.dataset "CIFAR10"  --use_fixed_args
