import logging
import os
import random
import time

from pathlib import Path

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import numpy as np

from hpbandster.optimizers import BOHB as BOHB

from metassl.hyperparameter_optimization.configspaces import get_imagenet_probability_augment_configspace
from metassl.hyperparameter_optimization.configspaces import get_cifar10_probability_augment_configspace
from metassl.hyperparameter_optimization.worker import HPOWorker
from metassl.hyperparameter_optimization.dispatcher import add_shutdown_worker_to_register_result


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


def run_worker(args, yaml_config, expt_sub_dir):
    time.sleep(20)  # short artificial delay to make sure the nameserver is already running
    host = hpns.nic_name_to_host(args.bohb.nic_name)
    print(f"host:{host=}")
    w = HPOWorker(args=args, yaml_config=yaml_config, expt_sub_dir=expt_sub_dir, run_id=args.bohb.run_id, host=host)
    w.load_nameserver_credentials(working_directory=expt_sub_dir)
    w.run(background=False)
    # time.sleep(300)


def run_master(args, yaml_config, expt_sub_dir):
    # Test experiments (whose expt_name are 'test') will always get overwritten
    if args.expt.expt_name == "test" and os.path.exists(expt_sub_dir):
        rmdir(expt_sub_dir)

    # NameServer
    ns = hpns.NameServer(
        run_id=args.bohb.run_id,
        working_directory=expt_sub_dir,
        nic_name=args.bohb.nic_name,
        port=args.bohb.port,
    )
    ns_host, ns_port = ns.start()
    print(f"{ns_host=}, {ns_host=}")

    # Start a background worker for the master node
    w = HPOWorker(
        args=args,
        yaml_config=yaml_config,
        expt_sub_dir=expt_sub_dir,
        run_id=args.bohb.run_id,
        host=ns_host,
        nameserver=ns_host,
        nameserver_port=ns_port,
    )
    # w.run(background=True)

    # Select a configspace based on configspace_mode
    if args.bohb.configspace_mode == "imagenet_probability_augment":
        configspace = get_imagenet_probability_augment_configspace()
    elif args.bohb.configspace_mode == "cifar10_probability_augment":
        configspace = get_cifar10_probability_augment_configspace()
    else:
        raise ValueError(f"Configspace {args.bohb.configspace_mode} is not implemented yet!")

    # Warmstarting
    if args.bohb.warmstarting:
        previous_run = hpres.logged_results_to_HBS_result(args.bohb.warmstarting_dir)
    else:
        previous_run = None

    # Create an optimizer
    result_logger = hpres.json_result_logger(directory=expt_sub_dir, overwrite=False)
    optimizer = BOHB(
        configspace=configspace,
        run_id=args.bohb.run_id,
        host=ns_host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        eta=args.bohb.eta,
        min_budget=args.bohb.min_budget,
        max_budget=args.bohb.max_budget,
        result_logger=result_logger,
        previous_result=previous_run,
    )
    
    # Overwrite the register results of the dispatcher to shutdown workers once they are finished
    add_shutdown_worker_to_register_result(optimizer.dispatcher)
    
    try:
        optimizer.run(n_iterations=args.bohb.n_iterations)
    finally:
        optimizer.shutdown(shutdown_workers=True)
        ns.shutdown()


def start_bohb_master(args, yaml_config, expt_sub_dir):
    set_seeds(args.bohb.seed)
    pil_logger = logging.getLogger("PIL")
    pil_logger.setLevel(logging.INFO)

    if args.bohb.worker:
        run_worker(args, yaml_config, expt_sub_dir)
    else:
        run_master(args, yaml_config, expt_sub_dir)
