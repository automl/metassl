# List available just commands
@list:
  just --list

# Format python code in {metassl}
format:
  bash utils_scripts/format.sh

# Clean test experiments
delete_test_runs:
  bash utils_scripts/clean_test_experiments.sh

# ---------------------------------------------------------------------------------------
# SIMSIAM TESTS
# ---------------------------------------------------------------------------------------

# Run SimSiam local Tests on CIFAR10
@run_cifar10_local_tests:
  echo ""
  echo ""
  echo "------- TESTING CIFAR10 PRETRAINING -------"
  python -m metassl.train_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --use_fixed_args --expt.is_testing_mode --data.dataset_percentage_usage 25 --train.epochs 5 --expt.warmup_epochs 0 --expt.seed 0 --expt.save_model_frequency 50 --expt.is_non_grad_based --expt.multiprocessing_distributed --expt.expt_name testing_cifar10_pretraining_1
  echo ""
  echo ""
  echo "------- TESTING CIFAR10 FINETUNING -------"
  python -m metassl.train_linear_classifier_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --use_fixed_args --expt.is_testing_mode --data.dataset_percentage_usage 25 --finetuning.epochs 5 --expt.warmup_epochs 0 --expt.seed 0 --expt.save_model_frequency 10 --expt.is_non_grad_based --expt.multiprocessing_distributed --expt.expt_name testing_cifar10_finetuning_1 --expt.ssl_model_checkpoint_path "experiments/CIFAR10/testing_cifar10_pretraining_1/checkpoint_0004.pth.tar"
  echo ""
  echo ""
  echo "------- TESTING CIFAR10 ALTERNATING -------"
  python -m metassl.train_alternating_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --use_fixed_args --expt.is_testing_mode --data.dataset_percentage_usage 25 --train.epochs 2 --finetuning.epochs 2 --expt.multiprocessing_distributed --expt.expt_name testing_cifar10_alternating_1

# Run SimSiam local Tests on CIFAR100
@run_cifar100_local_tests:
  echo ""
  echo ""
  echo "------- TESTING CIFAR100 PRETRAINING -------"
  python -m metassl.train_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --data.dataset "CIFAR100" --use_fixed_args --expt.is_testing_mode --data.dataset_percentage_usage 25 --train.epochs 5 --expt.warmup_epochs 0 --expt.seed 0 --expt.save_model_frequency 50 --expt.is_non_grad_based --expt.multiprocessing_distributed --expt.expt_name testing_cifar100_pretraining_1
  echo ""
  echo ""
  echo "------- TESTING CIFAR100 FINETUNING -------"
  python -m metassl.train_linear_classifier_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --data.dataset "CIFAR100" --use_fixed_args --expt.is_testing_mode --data.dataset_percentage_usage 25 --finetuning.epochs 5 --expt.warmup_epochs 0 --expt.seed 0 --expt.save_model_frequency 10 --expt.is_non_grad_based --expt.multiprocessing_distributed --expt.expt_name testing_cifar100_finetuning_1 --expt.ssl_model_checkpoint_path "experiments/CIFAR100/testing_cifar100_pretraining_1/checkpoint_0004.pth.tar"
  echo ""
  echo ""
  echo "------- TESTING CIFAR100 ALTERNATING -------"
  python -m metassl.train_alternating_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --data.dataset "CIFAR100" --use_fixed_args --expt.is_testing_mode --data.dataset_percentage_usage 25 --train.epochs 2 --finetuning.epochs 2 --expt.multiprocessing_distributed --expt.expt_name testing_cifar100_alternating_1

# Run SimSiam local Tests on CIFAR10 and CIFAR10
@run_all_local_tests:
  echo ""
  echo ""
  echo "------- TESTING CIFAR10 PRETRAINING -------"
  python -m metassl.train_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --use_fixed_args --expt.is_testing_mode --data.dataset_percentage_usage 25 --train.epochs 5 --expt.warmup_epochs 0 --expt.seed 0 --expt.save_model_frequency 50 --expt.is_non_grad_based --expt.multiprocessing_distributed --expt.expt_name testing_cifar10_pretraining_1
  echo ""
  echo ""
  echo "------- TESTING CIFAR10 FINETUNING -------"
  python -m metassl.train_linear_classifier_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --use_fixed_args --expt.is_testing_mode --data.dataset_percentage_usage 25 --finetuning.epochs 5 --expt.warmup_epochs 0 --expt.seed 0 --expt.save_model_frequency 10 --expt.is_non_grad_based --expt.multiprocessing_distributed --expt.expt_name testing_cifar10_finetuning_1 --expt.ssl_model_checkpoint_path "experiments/CIFAR10/testing_cifar10_pretraining_1/checkpoint_0004.pth.tar"
  echo ""
  echo ""
  echo "------- TESTING CIFAR10 ALTERNATING -------"
  python -m metassl.train_alternating_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --use_fixed_args --expt.is_testing_mode --data.dataset_percentage_usage 25 --train.epochs 2 --finetuning.epochs 2 --expt.multiprocessing_distributed --expt.expt_name testing_cifar10_alternating_1
  echo ""
  echo ""
  echo "------- TESTING CIFAR100 PRETRAINING -------"
  python -m metassl.train_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --data.dataset "CIFAR100" --use_fixed_args --expt.is_testing_mode --data.dataset_percentage_usage 25 --train.epochs 5 --expt.warmup_epochs 0 --expt.seed 0 --expt.save_model_frequency 50 --expt.is_non_grad_based --expt.multiprocessing_distributed --expt.expt_name testing_cifar100_pretraining_1
  echo ""
  echo ""
  echo "------- TESTING CIFAR100 FINETUNING -------"
  python -m metassl.train_linear_classifier_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --data.dataset "CIFAR100" --use_fixed_args --expt.is_testing_mode --data.dataset_percentage_usage 25 --finetuning.epochs 5 --expt.warmup_epochs 0 --expt.seed 0 --expt.save_model_frequency 10 --expt.is_non_grad_based --expt.multiprocessing_distributed --expt.expt_name testing_cifar100_finetuning_1 --expt.ssl_model_checkpoint_path "experiments/CIFAR100/testing_cifar100_pretraining_1/checkpoint_0004.pth.tar"
  echo ""
  echo ""
  echo "------- TESTING CIFAR100 ALTERNATING -------"
  python -m metassl.train_alternating_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --data.dataset "CIFAR100" --use_fixed_args --expt.is_testing_mode --data.dataset_percentage_usage 25 --train.epochs 2 --finetuning.epochs 2 --expt.multiprocessing_distributed --expt.expt_name testing_cifar100_alternating_1

# Run SimSiam local Tests on CIFAR10
@run_local_neps_test:
  echo ""
  echo ""
  echo "------- TESTING CIFAR10 NEPS -------"
  python -m metassl.train_simsiam --config "metassl/default_metassl_config_cifar10.yaml" --use_fixed_args --expt.is_testing_mode --data.dataset_percentage_usage 25 --train.epochs 5 --finetuning.epochs 5 --expt.warmup_epochs 0 --expt.seed 0 --expt.save_model_frequency 50 --expt.is_non_grad_based --expt.multiprocessing_distributed --neps.is_neps_run --neps.config_space hierarchical_nas --expt.expt_name testing_cifar10_neps_1

# ---------------------------------------------------------------------------------------
# SIMSIAM ON CIFAR10
# ---------------------------------------------------------------------------------------

# Submit SimSiam baseline on CIFAR10 - PT
@cifar10_baseline_pt EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/metassl/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/metassl/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/metassl/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_cifar10_baseline_pt.sh

# Submit SimSiam baseline on CIFAR10 - FT
@cifar10_baseline_ft EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/metassl/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/metassl/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/metassl/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster/submit_cifar10_baseline_ft.sh

# Submit NEPS
@neps EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/metassl/CIFAR10/NEPS/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu05 --output=/work/dlclarge2/wagnerd-metassl-experiments/metassl/CIFAR10/NEPS/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/metassl/CIFAR10/NEPS/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_cifar10_neps.sh

# ---------------------------------------------------------------------------------------
# SIMSIAM ON IMAGENET
# ---------------------------------------------------------------------------------------

# Submit SimSiam baseline on IMAGENET - PT
@imagenet_baseline_pt EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/metassl/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/metassl/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/metassl/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_imagenet_baseline_pt.sh

# Submit SimSiam baseline on IMAGENET - FT
@imagenet_baseline_ft EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/metassl/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/metassl/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/metassl/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_imagenet_baseline_ft.sh

# ---------------------------------------------------------------------------------------
# SIMSIAM ON CIFAR10 WITH BOHB
# ---------------------------------------------------------------------------------------

# Submit master to train alternating SimSiam on CIFAR10 with BOHB
@simsiam_cifar10_master EXPERIMENT_NAME:
  #!/usr/bin/env bash
  python -m metassl.train_alternating_simsiam --expt.expt_name {{EXPERIMENT_NAME}} --config "metassl/default_metassl_config_cifar10.yaml" --expt.expt_mode "CIFAR10_BOHB" --expt.is_non_grad_based --use_fixed_args --finetuning.valid_size 0.1 --train.epochs 450 --finetuning.epochs 450 --expt.warmup_epochs 0 --data.dataset_percentage_usage 25 --bohb.configspace_mode "probability_augment" --expt.data_augmentation_mode "probability_augment" --finetuning.data_augmentation "none" --bohb.max_budget 450 --bohb.min_budget 450 --bohb.n_iterations 70 --bohb.nic_name "enp1s0" --bohb.run_id "probability_augment_pt-only"

# Submit worker to train alternating SimSiam on CIFAR10 with BOHB
@simsiam_cifar10_worker EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_cifar10_bohb_worker.sh

# ---------------------------------------------------------------------------------------
# SIMSIAM ON IMAGENET
# ---------------------------------------------------------------------------------------

# Submit alternating SimSiam on IMAGENET with default setting to reproduce results
@simsiam_imagenet_default EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --nodelist=dlcgpu14 --output=/work/dlclarge2/wagnerd-metassl_experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_imagenet_default.sh

# Submit alternating SimSiam on IMAGENET with some experimental settings
@simsiam_imagenet_workspace EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl_experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_imagenet_workspace.sh

# ---------------------------------------------------------------------------------------
# SIMSIAM ON IMAGENET WITH BOHB
# ---------------------------------------------------------------------------------------

# Submit master to train alternating SimSiam on IMAGENET with BOHB
@simsiam_imagenet_master EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_imagenet_bohb_master.sh

# Submit worker to train alternating SimSiam on IMAGENET with BOHB
@simsiam_imagenet_worker EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_imagenet_bohb_worker.sh
