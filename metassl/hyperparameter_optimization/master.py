import logging
import os
import random
import time

from os.path import dirname
from os.path import join as path_join
from pathlib import Path

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import numpy as np

from hpbandster.optimizers import BOHB as BOHB

from metassl.hyperparameter_optimization.configspaces import get_test_configspace
from metassl.hyperparameter_optimization.configspaces import get_data_augmentation_configspace
from metassl.hyperparameter_optimization.worker import HPOWorker


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
    time.sleep(5)  # short artificial delay to make sure the nameserver is already running
    host = hpns.nic_name_to_host(args.nic_name)
    print(f"host:{host=}")
    w = HPOWorker(args=args, yaml_config=yaml_config, expt_sub_dir=expt_sub_dir, run_id=args.run_id, host=host)
    w.load_nameserver_credentials(working_directory=args.bohb.log_path)
    w.run(background=False)


def run_master(args, yaml_config, expt_sub_dir):
    # Test experiments (whose expt_name are 'test') will always get overwritten
    if args.expt.expt_name == "test" and os.path.exists(args.bohb.log_path):
        rmdir(args.bohb.log_path)

    # NameServer
    ns = hpns.NameServer(
        run_id=args.bohb.run_id,
        working_directory=args.bohb.log_path,
        nic_name=args.bohb.nic_name,
        port=40600,
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
    w.run(background=True)

    # Select a configspace based on configspace_mode
    if args.bohb.configspace_mode == "test":
        configspace = get_test_configspace()
    elif args.bohb.configspace_mode == "data_augmentation":
        configspace = get_data_augmentation_configspace()
    else:
        raise ValueError(f"Configspace {args.bohb.configspace_mode} is not implemented yet!")

    # Warmstarting
    if args.bohb.warmstarting:
        previous_run = hpres.logged_results_to_HBS_result(args.bohb.warmstarting_dir)
    else:
        previous_run = None

    # Create an optimizer
    result_logger = hpres.json_result_logger(directory=args.bohb.log_path, overwrite=False)
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

    try:
        optimizer.run(n_iterations=args.bohb.n_iterations)
    finally:
        optimizer.shutdown(shutdown_workers=True)
        ns.shutdown()


def start_bohb_master(args, yaml_config, expt_sub_dir):
    # TODO: Refactor saving directory!
    # Save test experiments (whose expt_name are 'test') in a separate directory called 'tests'
    if args.expt.expt_name == "test":
        args.bohb.log_path = path_join(
            dirname(__file__),
            "..",
            "..",
            "experiments",
            "tests",
            args.data.dataset,
        )
    else:
        args.bohb.log_path = path_join(
            dirname(__file__),
            "..",
            "..",
            "experiments",
            args.data.dataset,
            args.expt.expt_name,
        )

    set_seeds(args.bohb.seed)
    pil_logger = logging.getLogger("PIL")
    pil_logger.setLevel(logging.INFO)

    if args.bohb.worker:
        run_worker(args, yaml_config, expt_sub_dir)
    else:
        run_master(args, yaml_config, expt_sub_dir)
