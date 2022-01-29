import os

from hpbandster.core.worker import Worker

from metassl.baselines.pretraining import main as pt_main
from metassl.baselines.finetuning import main as ft_main

class HPOWorker(Worker):
    def __init__(self, args, trial_dir, **kwargs):
        self.args = args
        self.trial_dir = trial_dir
        super().__init__(**kwargs)

    def compute(self, config_id, config, budget, working_directory, *args, **kwargs):
        bohb_infos = {'bohb_config_id': config_id, 'bohb_config': config, 'bohb_budget': budget}
        # RUN PRETRAINING
        trial_dir = pt_main(self.args, trial_dir=self.trial_dir, bohb_infos=bohb_infos)
        # RUN FINETUNING + GET VALIDATION METRIC
        val_metric, test_metric = ft_main(self.args, trial_dir=trial_dir, bohb_infos=bohb_infos)
        return {
            "loss": -1 * val_metric,
            "info": {"test/metric": test_metric},
        }  # remember: HpBandSter always minimizes!
