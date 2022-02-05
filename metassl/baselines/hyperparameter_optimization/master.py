import logging
import os
from os import path
import random
import time

from pathlib import Path

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import numpy as np

from hpbandster.optimizers import BOHB as BOHB

from metassl.baselines.hyperparameter_optimization.configspaces import \
    get_cifar10_simsiam_augment_configspace, get_color_jitter_strengths_configspace, \
    get_double_color_jitter_strengths_configspace, get_lr_color_jitter_strengths_configspace, \
    get_rand_augment_configspace, get_probability_augment_configspace, get_double_probability_augment_configspace, \
    get_weight_decay_annealing_configspace, get_lr_cifar10_simsiam_augment_configspace
from metassl.baselines.hyperparameter_optimization.worker import HPOWorker
from metassl.baselines.hyperparameter_optimization.dispatcher import add_shutdown_worker_to_register_result


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def rmdir(directory):
    """ Checks whether a given directory already exists. If so, deltete it!"""
    directory = Path(directory)
    if os.path.exists(directory):
        for item in directory.iterdir():
            if item.is_dir():
                rmdir(item)
            else:
                item.unlink()
        directory.rmdir()


def run_worker(args, trial_dir):
    time.sleep(20)  # short artificial delay to make sure the nameserver is already running
    host = hpns.nic_name_to_host(args.nic_name)
    print(f"host:{host=}")
    w = HPOWorker(args=args, trial_dir=trial_dir, run_id=args.run_id, host=host)
    w.load_nameserver_credentials(working_directory=trial_dir)
    w.run(background=False)


def run_master(args, trial_dir):
    # Test experiments (whose expt_name are 'test') will always get overwritten
    if args.trial == "test" and os.path.exists(trial_dir):
        rmdir(trial_dir)

    # NameServer
    ns = hpns.NameServer(
        run_id=args.run_id,
        working_directory=trial_dir,
        nic_name=args.nic_name,
        port=args.port,
    )
    ns_host, ns_port = ns.start()
    print(f"{ns_host=}, {ns_host=}")

    # Start a background worker for the master node
    w = HPOWorker(
        args=args,
        trial_dir=trial_dir,
        run_id=args.run_id,
        host=ns_host,
        nameserver=ns_host,
        nameserver_port=ns_port,
    )
    if args.run_background_worker:
        w.run(background=True)
    else:
        pass

    # Select a configspace based on configspace_mode
    if args.configspace_mode == "cifar10_simsiam_augment":
        configspace = get_cifar10_simsiam_augment_configspace()
    elif args.configspace_mode == "lr_cifar10_simsiam_augment":
        configspace = get_lr_cifar10_simsiam_augment_configspace()
    elif args.configspace_mode == "color_jitter_strengths":
        configspace = get_color_jitter_strengths_configspace()
    elif args.configspace_mode == "double_color_jitter_strengths":
        configspace = get_double_color_jitter_strengths_configspace()
    elif args.configspace_mode == "lr_color_jitter_strengths":
        configspace = get_lr_color_jitter_strengths_configspace()
    elif args.configspace_mode == "rand_augment":
        configspace = get_rand_augment_configspace()
    elif args.configspace_mode == "probability_augment":
        configspace = get_probability_augment_configspace()
    elif args.configspace_mode == "double_probability_augment":
        configspace = get_double_probability_augment_configspace()
    elif args.configspace_mode == "weight_decay_annealing":
        configspace = get_weight_decay_annealing_configspace()
    else:
        raise ValueError(f"Configspace {args.configspace_mode} is not implemented yet!")

    # Warmstarting
    if args.warmstarting:
        previous_run = hpres.logged_results_to_HBS_result(args.bohb.warmstarting_dir)
    else:
        previous_run = None

    # Create an optimizer
    result_logger = hpres.json_result_logger(directory=trial_dir, overwrite=False)
    optimizer = BOHB(
        configspace=configspace,
        run_id=args.run_id,
        host=ns_host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        eta=args.eta,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        result_logger=result_logger,
        previous_result=previous_run,
    )

    # Overwrite the register results of the dispatcher to shutdown workers once they are finished if not in testing env
    if args.shutdown_workers:
        add_shutdown_worker_to_register_result(optimizer.dispatcher)
    else:
        pass

    try:
        optimizer.run(n_iterations=args.n_iterations)
    finally:
        optimizer.shutdown(shutdown_workers=True)
        ns.shutdown()


def start_bohb_master(args):
    set_seeds(args.seed)
    pil_logger = logging.getLogger("PIL")
    pil_logger.setLevel(logging.INFO)

    trial_dir = path.join(args.exp_dir, args.trial)

    if args.worker:
        run_worker(args, trial_dir)
    else:
        run_master(args, trial_dir)
