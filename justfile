# List available receipes
@list:
  just --list


# ----------------------------------------------------------------------------------------------------------------------
# SIMSIAM + CIFAR10
# ----------------------------------------------------------------------------------------------------------------------

# Submit BOHB master for SimSiam on CIFAR10
@bohb-master-cifar10 EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_cifar10_bohb_master.sh

# Submit BOHB worker for SimSiam on CIFAR10
@bohb-worker-cifar10 EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_cifar10_bohb_worker.sh
