from tensorboard.summary.writer.record_writer import RecordWriter as RecordWriter
from torch._vendor.packaging.version import Version as Version

from .writer import (
    FileWriter as FileWriter,
    SummaryWriter as SummaryWriter,
)

__all__ = ["FileWriter", "RecordWriter", "SummaryWriter"]
