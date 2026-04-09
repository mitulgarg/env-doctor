"""SQLAlchemy ORM models for the dashboard database."""
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base


class Machine(Base):
    __tablename__ = "machines"

    id = Column(String(36), primary_key=True)  # UUID from machine-id
    hostname = Column(String, nullable=False)
    platform = Column(String, nullable=True)
    python_version = Column(String, nullable=True)
    latest_status = Column(String, nullable=True)  # "pass"/"warning"/"fail"
    latest_snapshot_id = Column(Integer, nullable=True)
    first_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    snapshots = relationship(
        "Snapshot",
        back_populates="machine",
        order_by="desc(Snapshot.timestamp)",
    )

    def to_dict(self):
        return {
            "id": self.id,
            "hostname": self.hostname,
            "platform": self.platform,
            "python_version": self.python_version,
            "latest_status": self.latest_status,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }


class Snapshot(Base):
    __tablename__ = "snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(String(36), ForeignKey("machines.id"), nullable=False)
    status = Column(String, nullable=False)  # "pass"/"warning"/"fail"
    timestamp = Column(DateTime, nullable=False)
    report_json = Column(Text, nullable=False)  # Full check output as JSON blob

    # Denormalized fields for fast list/sort/filter
    summary_driver = Column(String, nullable=True)
    summary_cuda = Column(String, nullable=True)
    summary_issues_count = Column(Integer, nullable=True)
    gpu_name = Column(String, nullable=True)
    driver_version = Column(String, nullable=True)
    cuda_version = Column(String, nullable=True)
    torch_version = Column(String, nullable=True)
    is_heartbeat = Column(Boolean, default=False)

    machine = relationship("Machine", back_populates="snapshots")

    def to_dict(self, include_report_json: bool = False):
        d = {
            "id": self.id,
            "machine_id": self.machine_id,
            "status": self.status,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "gpu_name": self.gpu_name,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "torch_version": self.torch_version,
            "is_heartbeat": self.is_heartbeat,
        }
        if include_report_json:
            d["report_json"] = self.report_json
        return d
