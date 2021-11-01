from hpbandster.core.worker import Worker

from metassl.train_alternating_simsiam import main


class HPOWorker(Worker):
    def __init__(self, args, yaml_config, expt_sub_dir, **kwargs):
        self.args = args
        self.yaml_config = yaml_config
        self.expt_sub_dir = expt_sub_dir
        super().__init__(**kwargs)

    def compute(self, config_id, config, budget, working_directory, *args, **kwargs):
        val_metric, test_metric = main(config=self.yaml_config, expt_dir=self.expt_sub_dir, bohb_config_id=config_id, bohb_config=config, bohb_budget=budget)
        return {
            "loss": -1 * val_metric,
            "info": {"test/metric": test_metric},
        }  # remember: HpBandSter always minimizes!
