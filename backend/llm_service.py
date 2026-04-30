from __future__ import annotations

import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load local secrets for development (e.g. backend/api_keys.env)
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent
load_dotenv(_BASE_DIR / ".env", override=True)
load_dotenv(_BASE_DIR / "api_keys.env", override=True)
load_dotenv(_BASE_DIR / "api_key.env", override=True)


FALLBACK_REPORT = ""

FALLBACK_PATIENT = ""


def _format_features(features: List[Dict[str, Any]]) -> str:
    rows = []
    for f in features:
        name = f.get("name", "Feature")
        value = f.get("value", "N/A")
        unit = f.get("unit", "")
        status = f.get("status", "unknown")
        # Keep it as compact bullets so the model doesn't hallucinate structure.
        rows.append(f"- {name}: {value}{(' ' + unit) if unit else ''} (status: {status})")
    return "\n".join(rows) if rows else "- No feature data provided."


def _deterministic_reports(
    dr_stage: int,
    dr_score: float,
    clinical_features: List[Dict[str, Any]],
) -> Dict[str, str]:
    top_features = []
    for f in (clinical_features or [])[:4]:
        name = f.get("name", "Feature")
        value = f.get("value", "N/A")
        unit = f.get("unit", "")
        status = str(f.get("status", "unknown"))
        top_features.append(f"{name}: {value}{(' ' + unit) if unit else ''} ({status})")

    feature_line = "; ".join(top_features) if top_features else "No structured feature values were provided."

    stage_label = {
        0: "No DR",
        1: "Mild NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "Proliferative DR",
    }.get(int(dr_stage), f"Stage {dr_stage}")

    clinical_report = (
        f"Objective findings are consistent with diabetic retinopathy stage {dr_stage} ({stage_label}), "
        f"with model regression score {dr_score:.2f}. "
        f"Key extracted signals include: {feature_line}.\n\n"
        f"Clinical assessment indicates a research-grade AI estimate aligned with the above stage. "
        "Interpretation should be correlated with full ophthalmic examination and retinal history.\n\n"
        "Recommended plan: continue clinician-directed follow-up, compare with prior imaging when available, "
        "and review progression risk based on systemic diabetic control and visual symptoms."
    )

    patient_report = (
        f"Your scan suggests diabetic retinopathy stage {dr_stage} ({stage_label}). "
        "This is an AI estimate used to support your doctor, not replace them.\n\n"
        "Some image findings were detected and used to calculate this result. "
        "Your doctor will combine these findings with your eye exam and medical history.\n\n"
        "Please follow your eye specialist's follow-up plan, manage blood sugar and blood pressure, "
        "and seek care sooner if vision gets worse or changes suddenly."
    )

    return {"clinical_report": clinical_report, "patient_report": patient_report}


def _gemini_generate_text(system_prompt: str, user_prompt: str) -> str:
    """
    Small wrapper around google-genai with robust error handling.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY.")

    from google import genai

    client = genai.Client(api_key=api_key)
    # Keep prompts deterministic: embed "system" instructions into the user text.
    prompt = f"{system_prompt}\n\n{user_prompt}".strip()
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return ((getattr(response, "text", None) or "")).strip()


def generate_dual_clinical_reports(
    dr_stage: int,
    dr_score: float,
    clinical_features: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Dual-LLM reporting:
    - clinical_report: 3-paragraph expert ophthalmologist chart note (medical terminology)
    - patient_report: 6th-grade reading level explanation (empathetic, actionable)

    Returns:
      {"clinical_report": str, "patient_report": str}
    """
    try:
        clinical_system_prompt = (
            "You are an Expert Ophthalmologist acting as a clinical charting physician.\n"
            "Write a medical chart note in EXACTLY 3 paragraphs.\n"
            "Do NOT use markdown headers. Do NOT use bullet lists.\n"
            "Paragraph 1 must describe Objective Findings.\n"
            "Paragraph 2 must contain Clinical Assessment.\n"
            "Paragraph 3 must contain Recommended Plan.\n"
            "Use only the provided DR stage, DR regression score, and clinical features."
        )

        patient_system_prompt = (
            "You are an empathetic, clear medical communicator for patients.\n"
            "Write an explanation in EXACTLY 3 short paragraphs.\n"
            "Do NOT use markdown headers. Do NOT use bullet lists. Do NOT use bolds.\n"
            "Use a 6th-grade reading level.\n"
            "Explain: what the diagnosis means, why the measured findings matter, and what the patient should do next.\n"
            "Use only the provided DR stage, DR regression score, and clinical features."
        )

        features_text = _format_features(clinical_features)
        user_prompt = (
            f"DR Stage (0-4): {dr_stage}\n"
            f"DR Regression Score: {dr_score:.2f}\n"
            "Clinical Features:\n"
            f"{features_text}\n"
        )

        clinical_text = (_gemini_generate_text(clinical_system_prompt, user_prompt) or "").strip()
        patient_text = (_gemini_generate_text(patient_system_prompt, user_prompt) or "").strip()

        fallback = _deterministic_reports(dr_stage, dr_score, clinical_features)
        return {
            "clinical_report": clinical_text or fallback["clinical_report"],
            "patient_report": patient_text or fallback["patient_report"],
        }
    except Exception:
        # If API key is missing or call fails: return deterministic, non-empty reports.
        return _deterministic_reports(dr_stage, dr_score, clinical_features)


def generate_chat_reply(patient_report: str, diagnosis: str, message: str) -> str:
    """
    Context-aware chatbot for patients.
    """
    try:
        system_prompt = "You are a helpful AI assistant for an eye clinic."

        user_prompt = (
            "A patient is asking about their recent eye scan.\n"
            f"Context: Diagnosis: {diagnosis}\nPatient report: {patient_report}\n"
            f"The patient asks: {message}\n"
            "Answer clearly, concisely, and empathetically."
        )

        text = _gemini_generate_text(system_prompt, user_prompt)
        return text or "Unable to generate a response right now."
    except Exception:
        return _deterministic_chat_reply(patient_report=patient_report, diagnosis=diagnosis, message=message)


def _deterministic_chat_reply(patient_report: str, diagnosis: str, message: str) -> str:
    """
    Offline-safe fallback so chat still behaves like an assistant if the LLM API is unreachable.
    """
    msg = (message or "").strip().lower()
    diagnosis_text = (diagnosis or "the current scan finding").strip()

    if not msg:
        return (
            "I can help with your scan summary. You can ask what the diagnosis means, "
            "whether this may be urgent, and what next steps to follow."
        )

    if any(k in msg for k in ["diagnosis", "mean", "what is", "what does"]):
        return (
            f"Your result suggests {diagnosis_text}. This means changes were seen in the retina image that can occur with diabetes. "
            "This tool provides a supportive AI estimate, and your eye specialist should confirm it with clinical examination."
        )

    if any(k in msg for k in ["urgent", "emergency", "immediately", "asap", "serious"]):
        return (
            "If you notice sudden vision loss, severe eye pain, flashes, or many new floaters, seek urgent eye care now. "
            "If symptoms are stable, schedule timely follow-up with your ophthalmologist and bring this report."
        )

    if any(k in msg for k in ["next", "do now", "what should i do", "plan", "treatment", "follow"]):
        return (
            "Next steps: 1) book specialist follow-up, 2) compare with prior retinal images if available, "
            "3) optimize blood sugar, blood pressure, and lipid control, 4) return sooner if vision changes."
        )

    if any(k in msg for k in ["summary", "summarize", "bullet", "overview"]):
        clean_report = re.sub(r"\s+", " ", (patient_report or "").strip())
        if clean_report:
            return (
                f"Short summary: {clean_report[:600]}"
                + ("..." if len(clean_report) > 600 else "")
            )
        return (
            f"Summary: The scan indicates {diagnosis_text}. Please review the AI report sections and confirm management with your clinician."
        )

    return (
        f"I can help explain your result ({diagnosis_text}) in simple language. "
        "Try asking: 'What does this diagnosis mean?', 'What should I do next?', or 'Is this urgent?'."
    )

