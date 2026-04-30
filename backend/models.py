from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Column, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from .database import Base

JSON_TYPE = JSON().with_variant(JSONB, "postgresql")


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    patients = relationship(
        "Patient", back_populates="doctor", cascade="all, delete-orphan"
    )


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False, index=True)

    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=True, index=True)
    password_hash = Column(String(255), nullable=True)
    dob = Column(Date, nullable=True)
    gender = Column(String(20), nullable=True)
    patient_notes = Column(Text, nullable=True)

    doctor = relationship("Doctor", back_populates="patients")
    scans = relationship("Scan", back_populates="patient", cascade="all, delete-orphan")


class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)

    scan_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    image_path = Column(String(512), nullable=True)

    primary_diagnosis = Column(String(255), nullable=False)
    severity_level = Column(String(255), nullable=True)
    model_confidence = Column(Float, nullable=True)
    composite_score = Column(Float, nullable=True)

    clinical_features_json = Column(JSON_TYPE, nullable=True)

    # Generated narratives (dual LLM reporting)
    clinical_report = Column(Text, nullable=True)
    patient_report = Column(Text, nullable=True)

    patient = relationship("Patient", back_populates="scans")

