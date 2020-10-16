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

#-------------------------------------------------------------------------------

def csv_failover(func):
    """
    Implementing CSV failover would result in a block of code
    at the start of each function to check if failover has occurred
    and call the CSV logger instead of writing to the database. We
    do that with a slightly sneaky decorator instead.
    """
    def wrapper(*args):
        PostgresLogger_instance = args[0] # the PostgresLogger object will be 
                                          # the first argument.
        if PostgresLogger_instance._failover:
            # Get the CSV logger object, then call the method of
            # the CSV logger with the same name as the wrapped
            # function.
            getattr(
                PostgresLogger_instance.csv_logger, func.__name__
            )(*args[1:])
        else:
            try:
                func(*args)
            except Exception as err: # We could be more careful here, but we're failing over
                    # if anything at all goes wrong
                warnings.warn(
                    f"Unable to establish database connection due to {}. "
                    f"Failing over to CSV logging."
                )

                # TODO: SETTER FOR FAILOVER!!
                PostgresLogger_instance._failover = True
                log_dir = Path(f"/{os.getenv('LOG_DIR') or 'logs'}")
                PostgresLogger_instance.csv_logger = CSVLogger(
                    log_dir, PostgresLogger_instance.name, 
                    PostgresLogger_instance._version
                )

    return wrapper

#-------------------------------------------------------------------------------

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

        # Here, the logger tries to hit the database for the
        # first time. If it's not up, we failover into a CSV 
        # logger and save the results so nothing is lost.
        try:
            # Set up ORM
            self.db_engine = create_engine(self.connection_string)

            MBEBase.metadata.create_all(self.db_engine)

            # Initialize session
            self.Session = sessionmaker(bind=self.db_engine)

            # Create analysis object for this training run
            self.analysis = Analysis(
                name = self._name,
                start_time = self._start_time,
                epochs = self._epochs
            )
        except Exception as err:
            warnings.warn(
                f"Unable to establish database connection due to error: {err}. "
                f"Failing over to CSV logging."
            )
            self._failover = True
            log_dir = Path(f"/{os.getenv('LOG_DIR') or 'logs'}")
            self.csv_logger = CSVLogger(log_dir, self._name, self._version)

    @rank_zero_only
    @csv_failover
    def log_hyperparams(self, params):
            # Use "get" method so that any key misses
            # will return None and leave those records
            # Null.
            self.analysis.learning_rate = params.get("Learning rate")
            self.analysis.batch_size = params.get("Batch size")
            self.analysis.train_fold_size = params.get("Train fold size")

    @rank_zero_only
    @csv_failover
    def log_metrics(self, metrics, step):
        
        return None

    @rank_zero_only
    @csv_failover
    def finalize(self, status):
        # Close all open connections
        self.Session.close_all()

    @rank_zero_only
    @property
    @csv_failover
    def experiment(self):
        return None

    @rank_zero_only
    @property
    def name(self):
        return self._name

    @rank_zero_only
    @property
    def version(self):
        return self._version

    @property
    @rank_zero_only
    def failover(self):
        return self._failover

    @failover.setter
    def failover(self, value):
        self.failover=bool(value)
