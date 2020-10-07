import os

from dotenv import (
    find_dotenv,
    load_dotenv
)
from sqlalchemy import (
    create_engine, Column, String, Integer,
    Date
)
from sqlalchemy.orm import sessionmaker

from .schema import (
    MBEBase, Analysis, Updates
)

class MBEBackend(object):
    def __init__(self):
        load_dotenv(find_dotenv())
        self.pg_user = os.environ["POSTGRES_USERNAME"]
        self.pg_password = os.environ["POSTGRES_PASSWORD"]
        self.pg_database = os.environ["POSTGRES_DATABASE"]
        self.pg_host = os.environ["POSTGRES_HOST"]

        self.connection_string = (
            f"postgresql+psycopg2://"
            f"{self.pg_user}:{self.pg_password}@{self.pg_host}:5432/{self.pg_database}"
        )
        
        self.db_engine = create_engine(self.connection_string)
        MBEBase.metadata.create_all(self.db_engine)

        self.Session = sessionmaker(bind=self.db_engine)
