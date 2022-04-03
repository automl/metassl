#!/bin/bash
python metassl/train_simsiam.py --expt.expt_name $1 --train.epochs $2 --finetuning.epochs $3 --expt.warmup_epochs $4 --config $5 --data.dataset "CIFAR10"  --use_fixed_args --expt.run_knn_val --model.arch "our_resnet" --expt.multiprocessing_distributed
python metassl/train_linear_classifier_simsiam.py --expt.expt_name $1 --train.epochs $2 --finetuning.epochs $3 --expt.warmup_epochs $4 --config $5 --data.dataset "CIFAR10"  --use_fixed_args --model.arch "our_resnet" --expt.multiprocessing_distributed
