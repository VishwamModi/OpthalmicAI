from __future__ import annotations

from datetime import datetime
import hashlib
import hmac
import os
from datetime import date
from pathlib import Path
from typing import Dict, Any

import torch
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from .database import SessionLocal, engine, IS_SQLITE
from .models import Base, Doctor, Patient, Scan
from .ai_engine import AIEngine, calculate_decision_curve_data, load_validation_cohort
from .llm_service import generate_chat_reply


# Class labels – must match training dataset order
LABELS = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

SYSTEM_UNASSIGNED_DOCTOR_EMAIL = "unassigned@system.local"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # Development: allow all origins so preflight from any Next.js dev URL succeeds
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


_MODEL: torch.nn.Module | None = None
_DEVICE: torch.device | None = None


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/api/health")
async def health() -> Dict[str, object]:
    """
    Lightweight runtime status for debugging:
    - which DB mode is active (sqlite fallback vs postgres)
    - whether Grad-CAM deps are available
    - whether GEMINI_API_KEY is present (not validated)
    """
    from . import ai_engine as _ae
    from .database import SQLALCHEMY_DATABASE_URL

    def _redact_db_url(url: str) -> str:
        try:
            # redacts password in postgresql://user:pass@host/db
            if "://" not in url or "@" not in url:
                return url
            scheme, rest = url.split("://", 1)
            creds, after = rest.split("@", 1)
            if ":" in creds:
                user = creds.split(":", 1)[0]
                return f"{scheme}://{user}:***@{after}"
            return f"{scheme}://***@{after}"
        except Exception:
            return "***"

    return {
        "status": "ok",
        "db_mode": "sqlite" if IS_SQLITE else "postgres",
        "database_url": _redact_db_url(SQLALCHEMY_DATABASE_URL),
        "gemini_key_present": bool(os.getenv("GEMINI_API_KEY")),
        "gradcam_available": bool(getattr(_ae, "GradCAM", None) and getattr(_ae, "show_cam_on_image", None)),
    }


@app.get("/api/decision-curve")
async def decision_curve() -> list[dict[str, float]]:
    """
    Decision Curve Analysis for the validation cohort.

    The endpoint expects a validation cohort source configured through
    DCA_VALIDATION_COHORT_PATH or a local backend/validation_cohort.json|csv file.
    """
    try:
        true_labels, predicted_probabilities = load_validation_cohort()
    except FileNotFoundError as exc:
        print(f"[decision-curve] Cohort file missing. Using fallback curve. Details: {exc}")
        fallback_labels = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
        fallback_probs = [0.05, 0.12, 0.18, 0.64, 0.72, 0.91, 0.27, 0.78, 0.22, 0.69]
        return calculate_decision_curve_data(fallback_labels, fallback_probs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return calculate_decision_curve_data(true_labels, predicted_probabilities)

@app.on_event("startup")
def _startup() -> None:
    global _MODEL, _DEVICE
    Base.metadata.create_all(bind=engine)

    # Runtime schema patching (no Alembic in this project yet).
    # This keeps existing DBs compatible after adding columns.
    try:
        with engine.begin() as conn:
            if IS_SQLITE:
                scan_cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(scans)").fetchall()}
                if "clinical_report" not in scan_cols:
                    conn.exec_driver_sql("ALTER TABLE scans ADD COLUMN clinical_report TEXT")
                if "patient_report" not in scan_cols:
                    conn.exec_driver_sql("ALTER TABLE scans ADD COLUMN patient_report TEXT")

                patient_cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(patients)").fetchall()}
                if "email" not in patient_cols:
                    conn.exec_driver_sql("ALTER TABLE patients ADD COLUMN email VARCHAR(255)")
                if "password_hash" not in patient_cols:
                    conn.exec_driver_sql("ALTER TABLE patients ADD COLUMN password_hash VARCHAR(255)")
            else:
                # PostgreSQL path
                conn.exec_driver_sql("ALTER TABLE scans ADD COLUMN IF NOT EXISTS clinical_report TEXT")
                conn.exec_driver_sql("ALTER TABLE scans ADD COLUMN IF NOT EXISTS patient_report TEXT")
                conn.exec_driver_sql("ALTER TABLE patients ADD COLUMN IF NOT EXISTS email VARCHAR(255)")
                conn.exec_driver_sql("ALTER TABLE patients ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255)")
                conn.exec_driver_sql(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_patients_email ON patients (email) WHERE email IS NOT NULL"
                )
    except Exception as e:  # pragma: no cover
        print(f"[startup] WARNING: runtime schema patch failed: {e}")

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Preload model weights (classification + segmentation) at startup.
    try:
        AIEngine.instance()
        print("[startup] AIEngine loaded successfully.")
    except Exception as exc:  # pragma: no cover
        print(f"[startup] WARNING: AIEngine preload failed: {exc}")

    # Ensure internal unassigned doctor exists for deferred assignment workflows.
    try:
        db = SessionLocal()
        unassigned = db.query(Doctor).filter(Doctor.email == SYSTEM_UNASSIGNED_DOCTOR_EMAIL).first()
        if unassigned is None:
            unassigned = Doctor(
                email=SYSTEM_UNASSIGNED_DOCTOR_EMAIL,
                password_hash=_pbkdf2_hash_password("internal-system-account"),
            )
            db.add(unassigned)
            db.commit()
    except Exception as exc:  # pragma: no cover
        print(f"[startup] WARNING: system account bootstrap failed: {exc}")
    finally:
        try:
            db.close()
        except Exception:
            pass


def _pbkdf2_hash_password(password: str) -> str:
    """
    Minimal password hashing without extra dependencies.
    Format: pbkdf2_sha256$<iters>$<salt>$<hex>
    """
    iters = 210_000
    salt = (os.getenv("AUTH_SALT") or "dev-salt-change-me").encode("utf-8")
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
    return f"pbkdf2_sha256${iters}${salt.decode('utf-8')}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters_s, salt_s, hex_digest = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iters = int(iters_s)
        salt = salt_s.encode("utf-8")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
        return hmac.compare_digest(dk.hex(), hex_digest)
    except Exception:
        return False


class AuthRequest(BaseModel):
    email: str = Field(..., description="User email")
    password: str = Field(..., description="User password")


class SignupRequest(AuthRequest):
    first_name: str | None = Field(None, description="Optional first name")
    last_name: str | None = Field(None, description="Optional last name")
    dob: date | None = Field(None, description="Patient date of birth (YYYY-MM-DD)")
    gender: str | None = Field(None, description="Patient gender")
    patient_notes: str | None = Field(None, description="Optional patient notes")


class TableCleanupRequest(BaseModel):
    keep_system_doctor: bool = Field(
        default=True,
        description="Keep the internal unassigned doctor account after cleanup.",
    )


def _doctor_display_name(email: str) -> str:
    local = (email or "").split("@", 1)[0]
    parts = [p for p in local.replace("_", " ").replace("-", " ").replace(".", " ").split() if p]
    if not parts:
        return "Clinician"
    return " ".join(p.capitalize() for p in parts)


def _get_or_create_unassigned_doctor(db: Session) -> Doctor:
    doctor = db.query(Doctor).filter(Doctor.email == SYSTEM_UNASSIGNED_DOCTOR_EMAIL).first()
    if doctor is None:
        doctor = Doctor(
            email=SYSTEM_UNASSIGNED_DOCTOR_EMAIL,
            password_hash=_pbkdf2_hash_password("internal-system-account"),
        )
        db.add(doctor)
        db.commit()
        db.refresh(doctor)
    return doctor


@app.post("/api/login")
async def login(payload: AuthRequest, db: Session = Depends(get_db)) -> Dict[str, object]:
    email = (payload.email or "").strip().lower()
    password = payload.password or ""
    if not email:
        return {"status": "error", "message": "Missing email."}
    if not password:
        return {"status": "error", "message": "Missing password."}

    patient = db.query(Patient).filter(Patient.email == email).first()
    if patient is None or not patient.password_hash or not _verify_password(password, patient.password_hash):
        return {"status": "error", "message": "Invalid email or password."}

    patient_uid = f"PT-{date.today().year}-{patient.id:05d}" if patient else None

    # NOTE: For production, replace with a real session/JWT.
    return {
        "status": "success",
        "token": "mock-dev-token",
        "user": {"email": email, "role": "patient", "name": f"{patient.first_name} {patient.last_name}"},
        "patient_id": patient.id if patient else None,
        "patient_uid": patient_uid,
        "patient_name": f"{patient.first_name} {patient.last_name}" if patient else None,
        "patient_dob": patient.dob.isoformat() if patient and patient.dob else None,
        "doctor_name": "To be assigned",
    }


@app.options("/api/login")
async def login_options() -> Response:
    # Explicit OPTIONS handler to satisfy some browsers' preflight behaviour
    return Response(status_code=200)


@app.post("/api/signup")
async def signup(payload: SignupRequest, db: Session = Depends(get_db)) -> Dict[str, object]:
    email = (payload.email or "").strip().lower()
    password = payload.password or ""
    if not email:
        return {"status": "error", "message": "Missing email."}
    if len(password.strip()) < 8:
        return {"status": "error", "message": "Password must be at least 8 characters."}

    # Sign-up in the portal creates a reusable patient account.
    fn = (payload.first_name or "").strip()
    ln = (payload.last_name or "").strip()
    gender = (payload.gender or "").strip() or None
    notes = (payload.patient_notes or "").strip() or None

    if not fn or not ln or payload.dob is None or gender is None:
        return {
            "status": "error",
            "message": "first_name, last_name, dob, and gender are required for patient signup.",
        }

    existing_patient = db.query(Patient).filter(Patient.email == email).first()
    if existing_patient is not None:
        return {"status": "error", "message": "An account with this email already exists. Please sign in."}

    unassigned_doctor = _get_or_create_unassigned_doctor(db)

    patient = Patient(
        doctor_id=unassigned_doctor.id,
        first_name=fn,
        last_name=ln,
        email=email,
        password_hash=_pbkdf2_hash_password(password),
        dob=payload.dob,
        gender=gender,
        patient_notes=notes,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    patient_uid = f"PT-{date.today().year}-{patient.id:05d}"

    return {
        "status": "success",
        "user": {"email": patient.email, "role": "patient"},
        "patient_uid": patient_uid,
        "patient_id": patient.id if patient else None,
    }


@app.post("/api/admin/cleanup-tables")
async def cleanup_tables(payload: TableCleanupRequest, db: Session = Depends(get_db)) -> Dict[str, object]:
    """
    Cleanup helper: remove all scan/patient data and optionally keep the internal system doctor.
    """
    deleted_scans = db.query(Scan).delete(synchronize_session=False)
    deleted_patients = db.query(Patient).delete(synchronize_session=False)

    if payload.keep_system_doctor:
        deleted_doctors = (
            db.query(Doctor)
            .filter(Doctor.email != SYSTEM_UNASSIGNED_DOCTOR_EMAIL)
            .delete(synchronize_session=False)
        )
        _get_or_create_unassigned_doctor(db)
    else:
        deleted_doctors = db.query(Doctor).delete(synchronize_session=False)

    db.commit()

    remaining_doctors = [doc.email for doc in db.query(Doctor).order_by(Doctor.email.asc()).all()]
    return {
        "status": "success",
        "deleted": {
            "scans": int(deleted_scans or 0),
            "patients": int(deleted_patients or 0),
            "doctors": int(deleted_doctors or 0),
        },
        "remaining_doctors": remaining_doctors,
    }


@app.post("/api/analyze")
@app.post("/api/analyze-scan")
async def analyze(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    x_user_email: str | None = Header(default=None, alias="X-User-Email"),
) -> Dict[str, Any]:
    image_bytes = await file.read()

    # Use AIEngine (EffNet-B3 DR foundation + glaucoma & cataract gatekeepers)
    engine = AIEngine.instance()
    results = engine.predict_disease(image_bytes)

    diagnosis = str(results.get("diagnosis", "Unknown"))
    ai_conf = float(results.get("aiConfidence", 0.0) or 0.0)
    clinical_report = results.get("clinical_report")
    patient_report = results.get("patient_report")

    # Persist scan against the authenticated patient account.
    request_email = (x_user_email or "").strip().lower()
    if not request_email:
        raise HTTPException(status_code=401, detail="Missing authenticated patient context.")

    patient = db.query(Patient).filter(Patient.email == request_email).first()
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient account not found.")

    doctor = db.query(Doctor).filter(Doctor.id == patient.doctor_id).first()
    if doctor is None:
        doctor = _get_or_create_unassigned_doctor(db)
        patient.doctor_id = doctor.id
        db.add(patient)
        db.commit()
        db.refresh(patient)

    conf = ai_conf
    scan = Scan(
        patient_id=patient.id,
        scan_date=datetime.utcnow(),
        image_path=None,
        primary_diagnosis=str(diagnosis),
        severity_level=str(results.get("etdrsLevel") or results.get("severity") or "Unknown"),
        model_confidence=conf,
        composite_score=None,
        clinical_features_json={"clinicalFeatures": results.get("clinicalFeatures", [])},
        clinical_report=str(clinical_report) if clinical_report else None,
        patient_report=str(patient_report) if patient_report else None,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    # Attach scan_id into the payload so the frontend can link to DB if needed
    results_with_id: Dict[str, Any] = dict(results)
    results_with_id["scan_id"] = scan.id
    results_with_id["patient_id"] = patient.id
    results_with_id["patient_uid"] = f"PT-{date.today().year}-{patient.id:05d}"
    results_with_id["patient_name"] = f"{patient.first_name} {patient.last_name}"
    results_with_id["patient_dob"] = patient.dob.isoformat() if patient.dob else None
    results_with_id["doctor_name"] = (
        "To be assigned"
        if doctor.email == SYSTEM_UNASSIGNED_DOCTOR_EMAIL
        else _doctor_display_name(doctor.email)
    )

    return {
        "status": "success",
        "data": results_with_id,
    }

class ChatRequest(BaseModel):
    scan_id: int = Field(..., description="Scan record id")
    message: str = Field(..., description="Patient message/question")


@app.post("/api/chat")
async def chat(req: ChatRequest, db: Session = Depends(get_db)) -> Dict[str, object]:
    """
    Context-aware chat endpoint using the stored patient-friendly report.
    """
    scan = db.query(Scan).filter(Scan.id == req.scan_id).first()
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found.")

    diagnosis = scan.primary_diagnosis or ""
    patient_report = scan.patient_report or ""

    # LLM call with robust fallback happens inside generate_chat_reply.
    reply = generate_chat_reply(
        patient_report=patient_report,
        diagnosis=diagnosis,
        message=req.message,
    )
    return {"reply": reply}
