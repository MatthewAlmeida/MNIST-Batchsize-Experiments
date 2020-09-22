from sqlalchemy import (
    create_engine, Column, String, Integer,
    Date, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import ForeignKey

Base = declarative_base()

class Analysis(Base):
    __tablename__ = "analysis"
    aid = Column(Integer, primary_key = True)
    start_time = Column(Date)
    name = Column(String)
    split = relationship("crossvalsplit", backref="analysis")


class CrossValSplit(Base):
    __tablename__ = "crossvalsplit"
    cvid = Column(Integer, primary_key = True)
    aid = Column(Integer, ForeignKey("analysis.aid"))
    split = Column(String)


