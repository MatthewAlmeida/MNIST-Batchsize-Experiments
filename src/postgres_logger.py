import os
import warnings
from pathlib import Path

from datetime import datetime
from dotenv import (
    find_dotenv,
    load_dotenv
)

from pytorch_lightning.loggers import (
    LightningLoggerBase, CSVLogger
)
from pytorch_lightning.utilities import rank_zero_only

from sqlalchemy import (
    create_engine, Column, String, Integer,
    Date
)
from sqlalchemy.orm import sessionmaker

from schema import (
    MBEBase, Analysis, Updates
)

class PostgresLogger(LightningLoggerBase):
    def __init__(self, 
        experiment_name = "Default Experiment",
        experiment_version = 1,
        start_time = datetime.now(),
        epochs = 100,
    ):
        super(PostgresLogger, self).__init__()
    
        self._name = experiment_name
        self._version = experiment_version
        self._start_time = start_time
        self._epochs = epochs
        self._failover = False

        # Set up database connection

        load_dotenv(find_dotenv())
        self.pg_user = os.environ["POSTGRES_USERNAME"]
        self.pg_password = os.environ["POSTGRES_PASSWORD"]
        self.pg_database = os.environ["POSTGRES_DATABASE"]
        self.pg_host = os.environ["POSTGRES_HOST"]

        self.connection_string = (
            f"postgresql+psycopg2://"
            f"{self.pg_user}:{self.pg_password}@{self.pg_host}:5432/{self.pg_database}"
        )
        
        # Set up ORM
        self.db_engine = create_engine(self.connection_string)

        # Here, the logger tries to hit the database for the
        # first time. If it's not up, we failover into a CSV 
        # logger and save the results so nothing is lost.
        try:
            MBEBase.metadata.create_all(self.db_engine)

            # Initialize session
            self.Session = sessionmaker(bind=self.db_engine)

            # Create analysis object for this training run
            self.analysis = Analysis(
                name = self._name,
                start_time = self._start_time,
                epochs = self._epochs
            )
        except:
            warnings.warn(
                f"Unable to establish database connection. Failing "
                f"over to CSV logging."
            )
            self._failover = True
            log_dir = Path(f"/{os.getenv('LOG_DIR') or 'logs'}")
            self.csv_logger = CSVLogger(log_dir, self._name, self._version)

    @rank_zero_only
    def log_hyperparams(self, params):
        if self._failover:
            # Database unavailable, log to CSV.
            self.csv_logger.log_hyperparams(params)
        else:
            # Use "get" method so that any key misses
            # will return None and leave those records
            # Null.
            self.analysis.learning_rate = params.get("Learning rate")
            self.analysis.batch_size = params.get("Batch size")
            self.analysis.train_fold_size = params.get("Train fold size")

    @rank_zero_only
    def log_metrics(self, metrics, step):
        
        return None

    @rank_zero_only
    def finalize(self, status):
        # Close all open connections
        if not self.csv_logger:
            self.Session.close_all()

    @property
    @rank_zero_only
    def experiment(self):
        return None

    @property
    @rank_zero_only
    def name(self):
        return self._name

    @property
    @rank_zero_only
    def version(self):
        return self._version
