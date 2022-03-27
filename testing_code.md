## Diane (local)


#### CIFAR10 Pretraining
```
python -m metassl.train_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --use_fixed_args --data.dataset_percentage_usage 25 --train.epochs 5 --expt.warmup_epochs 0 --expt.seed 0 --expt.save_model_frequency 50 --expt.is_non_grad_based --expt.multiprocessing_distributed --model.arch "our_resnet" --expt.expt_name testing_cifar10_pretraining_1
```

#### CIFAR10 Finetuning
```
python -m metassl.train_linear_classifier_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --use_fixed_args --data.dataset_percentage_usage 25 --finetuning.epochs 5 --expt.warmup_epochs 0 --expt.seed 0 --expt.save_model_frequency 10 --expt.is_non_grad_based --expt.multiprocessing_distributed --model.arch our_resnet --expt.expt_name testing_cifar10_finetuning_1 --expt.ssl_model_checkpoint_path "experiments/CIFAR10/testing_cifar10_pretraining_1/checkpoint_0004.pth.tar"
```

#### CIFAR10 Alternating
```
python -m metassl.train_alternating_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --use_fixed_args --data.dataset_percentage_usage 25 --train.epochs 2 --finetuning.epochs 2 --expt.multiprocessing_distributed --expt.expt_name testing_cifar10_alternating_1
```





## Diane (cluster)


#### CIFAR10 Pretraining
```
python -m metassl.train_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --use_fixed_args --data.dataset_percentage_usage 25 --train.epochs 5 --expt.warmup_epochs 0 --expt.seed 0 --expt.save_model_frequency 50 --expt.is_non_grad_based --expt.multiprocessing_distributed --model.arch "our_resnet" --expt.expt_name testing_cifar10_pretraining_1
```

#### CIFAR10 Finetuning
```
python -m metassl.train_linear_classifier_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --use_fixed_args --data.dataset_percentage_usage 25 --finetuning.epochs 5 --expt.warmup_epochs 0 --expt.seed 0 --expt.save_model_frequency 10 --expt.is_non_grad_based --expt.multiprocessing_distributed --model.arch our_resnet --expt.expt_name testing_cifar10_finetuning_1 --expt.ssl_model_checkpoint_path "/work/dlclarge2/wagnerd-metassl-experiments/metassl/CIFAR10/testing_cifar10_pretraining_1/checkpoint_0004.pth.tar"
```

#### CIFAR10 Alternating
```
python -m metassl.train_alternating_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --use_fixed_args --data.dataset_percentage_usage 25 --train.epochs 2 --finetuning.epochs 2 --expt.multiprocessing_distributed --expt.expt_name testing_cifar10_alternating_1
```



#### ImageNet Pretraining
```
python -m metassl.train_simsiam --config "metassl/default_metassl_config_imagenet.yaml" --use_fixed_args --data.dataset_percentage_usage 0.1 --train.epochs 2 --train.batch_size 4 --expt.warmup_epochs 0 --expt.seed 0 --expt.save_model_frequency 50 --expt.is_non_grad_based --expt.expt_name testing_imagenet_pretraining_1
```

#### ImageNet Finetuning
```
python -m metassl.train_linear_classifier_simsiam --config "metassl/default_metassl_config_imagenet.yaml" --use_fixed_args --data.dataset_percentage_usage 0.1 --finetuning.epochs 2 --finetuning.batch_size 4 --expt.warmup_epochs 0 --expt.seed 0 --expt.save_model_frequency 10 --expt.is_non_grad_based --expt.expt_name testing_imagenet_finetuning_1 --expt.ssl_model_checkpoint_path "/work/dlclarge2/wagnerd-metassl-experiments/metassl/ImageNet/testing_imagenet_pretraining_1/checkpoint_0001.pth.tar"
```

#### ImageNet Alternating
```
python -m metassl.train_alternating_simsiam --config "metassl/default_metassl_config_imagenet.yaml" --use_fixed_args --data.dataset_percentage_usage 0.1 --train.batch_size 4 --finetuning.batch_size 4 --train.epochs 2 --finetuning.epochs 2 --expt.expt_name testing_imagenet_alternating_1
```





## Fábio


TODO?





## External user


TODO
