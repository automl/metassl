# List available just commands
@list:
  just --list

# ---------------------------------------------------------------------------------------
# BASELINES - SIMSIAM ON CIFAR10
# ---------------------------------------------------------------------------------------

# Submit Baseline for SimSiam on CIFAR10
@baseline EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_baseline_sequential_simsiam_cifar10.sh

# Submit workspace (baseline) experiment with SimSiam on CIFAR10
@workspace EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_workspace_sequential_simsiam_cifar10.sh


# Submit TrivialAugment for SimSiam on CIFAR10
@trivial EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_trivial_sequential_simsiam_cifar10.sh


# Submit SmartSamplingAugment for SimSiam on CIFAR10
@smartsampling EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_smartsampling_sequential_simsiam_cifar10.sh

# ---------------------------------------------------------------------------------------
# BASELINES - SIMSIAM ON CIFAR10 WITH BOHB
# ---------------------------------------------------------------------------------------

# Start master for weight decay configspace on the login node
@login-master-wd EXPERIMENT_NAME:
  #!/usr/bin/env bash
  python3 -m metassl.baselines.execute_pt_and_ft --gpu 0 --is_bohb_run --valid_size 0.1 --seed 1 --trial {{EXPERIMENT_NAME}} --exp_dir "/work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10" --pretrained /work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/{{EXPERIMENT_NAME}} --n_iterations 250 --run_id "weight_decay_annealing" --configspace_mode 'weight_decay_annealing' --do_weight_decay_annealing --pt_learning_rate 0.06 --shutdown_workers --nic_name "enp1s0"

# Submit worker for weight decay annealing configspace to train SimSiam on CIFAR10 with BOHB
@worker-wd EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_worker-wd_sequential_simsiam_cifar10.sh


# Start master for color jitter configspace on the login node
@login-master-cj EXPERIMENT_NAME:
  #!/usr/bin/env bash
  python3 -m metassl.baselines.execute_pt_and_ft --gpu 0 --is_bohb_run --valid_size 0.1 --seed 0 --trial {{EXPERIMENT_NAME}} --exp_dir "/work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10" --pretrained /work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/{{EXPERIMENT_NAME}} --n_iterations 100 --run_id "color_jitter_strengths" --configspace_mode 'color_jitter_strengths' --pt_learning_rate 0.06 --shutdown_workers --nic_name "enp1s0"

# Submit worker for color jitter configspace to train SimSiam on CIFAR10 with BOHB
@worker-cj EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_worker-cj_sequential_simsiam_cifar10.sh


# Start master for jointly lr + color jitter configspace on the login node
@login-master-lr-cj EXPERIMENT_NAME:
  #!/usr/bin/env bash
  python3 -m metassl.baselines.execute_pt_and_ft --gpu 0 --is_bohb_run --valid_size 0.1 --seed 0 --trial {{EXPERIMENT_NAME}} --exp_dir "/work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10" --pretrained /work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/{{EXPERIMENT_NAME}} --n_iterations 100 --run_id "lr_color_jitter_strengths" --configspace_mode 'lr_color_jitter_strengths' --pt_learning_rate 0.06 --shutdown_workers --nic_name "enp1s0"

# Submit worker for jointly lr + color jitter configspace to train SimSiam on CIFAR10 with BOHB
@worker-lr-cj EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_worker-lr-cj_sequential_simsiam_cifar10.sh







# ---------------------------------------------------------------------------------------
# SIMSIAM ON CIFAR10
# ---------------------------------------------------------------------------------------

# Submit alternating SimSiam on CIFAR10 with default setting to reproduce results
#@simsiam_cifar10_default EXPERIMENT_NAME:
#  #!/usr/bin/env bash
#  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
#  sbatch --exclude=dlcgpu42 --nodelist=dlcgpu14 --output=/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_cifar10_default.sh

# Submit alternating SimSiam on CIFAR10 with some experimental settings
#@simsiam_cifar10_workspace EXPERIMENT_NAME:
#  #!/usr/bin/env bash
#  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
#  sbatch --exclude=dlcgpu42 --nodelist=dlcgpu25 --output=/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_cifar10_workspace.sh

# Submit alternating SimSiam on CIFAR10 with some experimental settings on Bosch queue
#@simsiam_cifar10_bosch_workspace EXPERIMENT_NAME:
#  #!/usr/bin/env bash
#  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
#  sbatch --nodelist=mlgpu08 --output=/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_cifar10_bosch_workspace.sh

# Submit non-alternating SimSiam on CIFAR10 with some experimental settings on Bosch queue
#@non-alternating EXPERIMENT_NAME:
#  #!/usr/bin/env bash
#  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
#  sbatch --output=/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_simsiam_cifar10_workspace.sh

# ---------------------------------------------------------------------------------------
# SIMSIAM ON CIFAR10 WITH BOHB
# ---------------------------------------------------------------------------------------

# Submit master to train alternating SimSiam on CIFAR10 with BOHB
#@simsiam_cifar10_master EXPERIMENT_NAME:
#  #!/usr/bin/env bash
#  python -m metassl.train_alternating_simsiam --expt.expt_name {{EXPERIMENT_NAME}} --config "metassl/default_metassl_config_cifar10.yaml" --expt.expt_mode "CIFAR10_BOHB" --expt.is_non_grad_based --use_fixed_args --finetuning.valid_size 0.1 --train.epochs 450 --finetuning.epochs 450 --expt.warmup_epochs 0 --data.dataset_percentage_usage 25 --bohb.configspace_mode "probability_augment" --expt.data_augmentation_mode "probability_augment" --finetuning.data_augmentation "none" --bohb.max_budget 450 --bohb.min_budget 450 --bohb.n_iterations 70 --bohb.nic_name "enp1s0" --bohb.run_id "probability_augment_pt-only"

# Submit worker to train alternating SimSiam on CIFAR10 with BOHB
#@simsiam_cifar10_worker EXPERIMENT_NAME:
#  #!/usr/bin/env bash
#  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
#  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_cifar10_bohb_worker.sh

# ---------------------------------------------------------------------------------------
# SIMSIAM ON IMAGENET
# ---------------------------------------------------------------------------------------

# Submit alternating SimSiam on IMAGENET with default setting to reproduce results
#@simsiam_imagenet_default EXPERIMENT_NAME:
#  #!/usr/bin/env bash
#  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
#  sbatch --exclude=dlcgpu42 --nodelist=dlcgpu14 --output=/work/dlclarge2/wagnerd-metassl-experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_imagenet_default.sh

# Submit alternating SimSiam on IMAGENET with some experimental settings
#@simsiam_imagenet_workspace EXPERIMENT_NAME:
#  #!/usr/bin/env bash
#  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
#  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl-experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_imagenet_workspace.sh

# ---------------------------------------------------------------------------------------
# SIMSIAM ON IMAGENET WITH BOHB
# ---------------------------------------------------------------------------------------

# Submit master to train alternating SimSiam on IMAGENET with BOHB
#@simsiam_imagenet_master EXPERIMENT_NAME:
#  #!/usr/bin/env bash
#  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
#  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl-experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_imagenet_bohb_master.sh

# Submit worker to train alternating SimSiam on IMAGENET with BOHB
#@simsiam_imagenet_worker EXPERIMENT_NAME:
#  #!/usr/bin/env bash
#  mkdir -p /work/dlclarge2/wagnerd-metassl-experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
#  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl-experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl-experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_imagenet_bohb_worker.sh
