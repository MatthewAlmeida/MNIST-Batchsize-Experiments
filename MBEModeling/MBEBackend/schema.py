from datetime import datetime

from sqlalchemy import (
    create_engine, Column, String, Integer, Float,
    Date, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (
    relationship
)
from sqlalchemy.dialects.postgresql import JSONB

MBEBase = declarative_base()

class Analysis(MBEBase):
    """
    This class defines a Table to hold Analysis 
    objects, the core objects representing a 
    single training process. This table is one-to-one
    with Trainset and one-to-many with Updates.
    """
    __tablename__ = "analysis"

    aid = Column(Integer, primary_key = True)
    name = Column(String)
    start_time = Column(Date)
    learning_rate = Column(Float)
    batch_size = Column(Integer)
    epochs = Column(Integer)
    train_fold_size = Column(Integer)
    test_result = Column(Float)

    # Each training set is stored as a JSON file of 
    # example indices, each of length train_fold_size.
    training_set = Column(JSONB)

    # Declare relationships among objects
    updates = relationship("updates", backref="analysis")

    def __init__(self, 
        name: str, start_time: datetime,
        learning_rate: float, batch_size: int,
        epochs: int, train_fold_size:int, test_result: float,
        training_set: JSONB
    ) -> None:
        self.name = name
        self.start_time = start_time
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_fold_size = train_fold_size
        self.test_result = test_result
        self.training_set = training_set

class Updates(MBEBase):
    __tablename__ = "updates"
    uid = Column(Integer, primary_key=True)
    aid = Column(Integer, ForeignKey("analysis.aid"), primary_key=True)
    epoch = Column(Integer)
    train_acc = Column(Float)
    train_loss = Column(Float)
    gradient_inner_product = Column(Float)

    def __init__(self, 
        epoch: int, train_acc: float,
        train_loss: float, gradient_inner_product: float
    ) -> None:
        self.epoch = epoch
        self.train_acc = train_acc
        self.train_loss = train_loss
        self.gradient_inner_product = gradient_inner_product
