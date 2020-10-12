from pytorch_lightning.loggers import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

from .backend import MBEBackend

class PostgresLogger(LightningLoggerBase):
    def __init__(self, 
        experiment_name = "Default Experiment",
        experiment_version = 1

    ):
        super(PostgresLogger, self).__init__()
        self.B = MBEBackend()
        self._name = experiment_name
        self._version = experiment_version

    @rank_zero_only
    def log_hyperparams(self, params):
        return None

    @rank_zero_only
    def log_metrics(self, metrics, step):
        return None

    @rank_zero_only
    def finalize(self, status):
        return None

    @property
    @rank_zero_experiment
    def experiment(self):
        return self.B

    @property
    @rank_zero_only
    def name(self):
        return self._name

    @property
    @rank_zero_only
    def version(self):
        return self._version
