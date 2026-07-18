from __future__ import annotations

from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms


MODEL_PATH = Path(__file__).resolve().parent / "OphthalmicAI_Final_EffNetB3.pt"
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


def load_checkpoint(model: torch.nn.Module, path: Path) -> torch.nn.Module:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Upload the final DR checkpoint to the Space root."
        )
    saved = torch.load(path, map_location="cpu", weights_only=False)
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


MODEL = load_checkpoint(
    timm.create_model("efficientnet_b3", pretrained=False, num_classes=1),
    MODEL_PATH,
)


def prepare_image(image: Image.Image) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
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
    return tensor, rgb_original, processed_rgb


def ordinal_grade(score: float) -> int:
    return int(np.digitize(score, [0.5, 1.5, 2.5, 3.5]).item())


def gradcam(model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
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
            raise RuntimeError("Grad-CAM hooks did not capture the feature maps.")
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


def analyze(image: Image.Image | None):
    if image is None:
        raise gr.Error("Upload a retinal fundus image first.")

    tensor, _original, processed = prepare_image(image)
    heatmap = gradcam(MODEL, tensor)
    with torch.no_grad():
        raw_score = float(MODEL(tensor).view(-1)[0].item())

    grade = ordinal_grade(raw_score)
    color_heatmap = cv2.applyColorMap(
        np.uint8(np.clip(heatmap, 0, 1) * 255),
        cv2.COLORMAP_JET,
    )
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(
        np.clip(0.58 * processed.astype(np.float32) + 0.42 * color_heatmap, 0, 255)
    )

    proximity = max(0.0, 1.0 - abs(np.clip(raw_score, 0, 4) - grade)) * 100
    result = (
        f"## {CLASS_NAMES[grade]}\n\n"
        f"- **DR grade:** {grade}/4\n"
        f"- **Raw ordinal score:** {raw_score:.3f}\n"
        f"- **Ordinal proximity:** {proximity:.1f}% *(uncalibrated; not a probability)*\n"
        f"- **Suggested next step:** {RECOMMENDATIONS[grade]}\n\n"
        "This output is for research and educational use and must not replace "
        "evaluation by a qualified clinician."
    )
    return result, overlay


with gr.Blocks(title="OphthalmicAI DR Severity") as demo:
    gr.Markdown(
        "# OphthalmicAI — DR Severity\n"
        "Upload a retinal fundus image to obtain an EfficientNet-B3 ordinal "
        "DR grade and Grad-CAM visualization.\n\n"
        "[Watch the updated project walkthrough](https://youtu.be/opcj0hZPnxU)"
    )
    with gr.Row():
        image_input = gr.Image(type="pil", label="Fundus image")
        gradcam_output = gr.Image(type="numpy", label="Grad-CAM")
    result_output = gr.Markdown()
    analyze_button = gr.Button("Analyze image", variant="primary")
    analyze_button.click(
        analyze,
        inputs=image_input,
        outputs=[result_output, gradcam_output],
    )


if __name__ == "__main__":
    demo.launch()
