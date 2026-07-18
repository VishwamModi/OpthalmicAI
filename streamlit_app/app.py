from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import timm
import torch
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "Models" / "OphthalmicAI_Final_EffNetB3.pt"
DEVICE = torch.device("cpu")
CLASS_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]
RECOMMENDATIONS = [
    "Continue routine screening according to the applicable clinical protocol.",
    "Repeat retinal assessment and optimize diabetes risk-factor control.",
    "Arrange ophthalmology follow-up for a complete retinal assessment.",
    "Prompt referral to a retina specialist is recommended.",
    "Urgent retina-specialist evaluation is recommended.",
]
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
torch.set_num_threads(2)


@st.cache_resource(show_spinner="Loading the EfficientNet-B3 checkpoint…")
def load_model() -> torch.nn.Module:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH}")

    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=1)
    saved = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state = saved.get("state_dict", saved) if isinstance(saved, dict) else saved
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)
    return model.to(DEVICE).eval()


def prepare_image(image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    rgb_original = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb_original, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (300, 300), interpolation=cv2.INTER_AREA)

    circle = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(circle, (150, 150), int(150 * 0.95), 255, -1)
    bgr = cv2.bitwise_and(bgr, bgr, mask=circle)

    green = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(bgr[:, :, 1])
    processed_rgb = cv2.merge([green, green, green])
    tensor = torch.from_numpy(processed_rgb.transpose(2, 0, 1)).float() / 255.0
    tensor = NORMALIZE(tensor).unsqueeze(0).to(DEVICE)
    return tensor, processed_rgb


def ordinal_grade(score: float) -> int:
    return int(np.digitize(score, [0.5, 1.5, 2.5, 3.5]).item())


def calculate_gradcam(model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    activations = None
    gradients = None

    def forward_hook(_module, _inputs, output):
        nonlocal activations
        activations = output

        def save_gradient(gradient):
            nonlocal gradients
            gradients = gradient

        output.register_hook(save_gradient)

    handle = model.conv_head.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        output = model(tensor).view(-1)[0]
        output.backward()
        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM did not capture feature maps.")
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activations).sum(dim=1))[0]
        heatmap -= heatmap.min()
        heatmap /= heatmap.max().clamp_min(1e-8)
        heatmap = torch.nn.functional.interpolate(
            heatmap[None, None],
            size=(300, 300),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return heatmap.detach().cpu().numpy()
    finally:
        handle.remove()


def analyze(image: Image.Image) -> tuple[float, int, float, np.ndarray]:
    model = load_model()
    tensor, processed = prepare_image(image)
    heatmap = calculate_gradcam(model, tensor)
    with torch.inference_mode():
        raw_score = float(model(tensor).view(-1)[0].item())

    grade = ordinal_grade(raw_score)
    proximity = max(0.0, 1.0 - abs(np.clip(raw_score, 0, 4) - grade)) * 100
    color_heatmap = cv2.applyColorMap(
        np.uint8(np.clip(heatmap, 0, 1) * 255),
        cv2.COLORMAP_JET,
    )
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(
        np.clip(0.58 * processed.astype(np.float32) + 0.42 * color_heatmap, 0, 255)
    )
    return raw_score, grade, proximity, overlay


st.set_page_config(
    page_title="OphthalmicAI DR Severity",
    page_icon="👁️",
    layout="wide",
)
st.title("OphthalmicAI — Diabetic Retinopathy Severity")
st.write(
    "Upload a retinal fundus image for an EfficientNet-B3 ordinal DR grade "
    "and Grad-CAM visualization."
)
st.video("https://youtu.be/opcj0hZPnxU")

uploaded = st.file_uploader(
    "Fundus image",
    type=["png", "jpg", "jpeg"],
    help="Use de-identified retinal fundus images only.",
)

if uploaded is not None:
    input_image = Image.open(uploaded).convert("RGB")
    left, right = st.columns(2)
    with left:
        st.image(input_image, caption="Uploaded fundus image", use_container_width=True)

    if st.button("Analyze image", type="primary", use_container_width=True):
        try:
            with st.spinner("Running CPU inference and Grad-CAM…"):
                raw_score, grade, proximity, overlay = analyze(input_image)
            with right:
                st.image(overlay, caption="Grad-CAM", use_container_width=True)
            st.subheader(CLASS_NAMES[grade])
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("DR grade", f"{grade}/4")
            metric_b.metric("Raw ordinal score", f"{raw_score:.3f}")
            metric_c.metric("Ordinal proximity", f"{proximity:.1f}%")
            st.caption("Ordinal proximity is uncalibrated and is not a probability.")
            st.info(RECOMMENDATIONS[grade])
        except Exception as error:
            st.error(f"Analysis failed: {error}")

st.warning(
    "Research and educational demonstration only. This software is not a "
    "medical device and must not replace evaluation by a qualified clinician."
)
