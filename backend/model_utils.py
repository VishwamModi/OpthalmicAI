from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None  # type: ignore

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:  # pragma: no cover
    GradCAM = None  # type: ignore
    show_cam_on_image = None  # type: ignore
    ClassifierOutputTarget = None  # type: ignore


LABELS = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_timm() -> None:
    if timm is None:
        raise RuntimeError("Missing dependency: timm. Install with `pip install timm`.")


def _ensure_gradcam() -> None:
    # We can fall back to a lightweight internal Grad-CAM implementation.
    return


def _gradcam_fallback(model, x: torch.Tensor, target_layer, class_idx: int) -> np.ndarray:
    """
    Dependency-free Grad-CAM fallback.
    Returns a (H,W) float array in [0,1] at input resolution.
    """
    activations = None
    gradients = None

    def fwd_hook(_m, _inp, out):
        nonlocal activations
        activations = out

        def _save_grad(g):
            nonlocal gradients
            gradients = g

        out.register_hook(_save_grad)

    handle = target_layer.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        score = logits[:, class_idx].sum()
        score.backward()

        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM fallback did not capture activations/gradients.")

        # (N,C,H,W)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam[0].detach().float().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        cam_t = torch.from_numpy(cam)[None, None, ...]
        cam_up = torch.nn.functional.interpolate(
            cam_t,
            size=(int(x.shape[2]), int(x.shape[3])),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return cam_up.detach().cpu().numpy()
    finally:
        handle.remove()


def standardize_fundus_image(image_bytes: bytes, out_size: int) -> Image.Image:
    """
    OpenCV-based fundus standardization to reduce domain shift:
    - Decode BGR
    - Auto-crop largest external contour (removes black borders)
    - Resize to out_size x out_size
    - Ben Graham lighting normalization
    - Return RGB PIL
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image bytes.")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = int(0.03 * max(w, h))
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, bgr.shape[1])
        y1 = min(y + h + pad, bgr.shape[0])
        bgr = bgr[y0:y1, x0:x1]

    bgr = cv2.resize(bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)
    sigma = out_size / 10.0
    ben = cv2.addWeighted(bgr, 4.0, cv2.GaussianBlur(bgr, (0, 0), sigma), -4.0, 128.0)

    rgb = cv2.cvtColor(ben, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _tensor_from_pil(img: Image.Image) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )
    return tfm(img).unsqueeze(0)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            v = ckpt.get(k)
            if isinstance(v, dict):
                return v
    return ckpt


def _strip_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        for p in ("module.", "model."):
            if nk.startswith(p):
                nk = nk[len(p) :]
                break
        cleaned[nk] = v
    return cleaned


def load_model(path: str | Path):
    """
    Load EfficientNet-B3 (timm) from `EfficientNetB3.pt`.
    The provided checkpoint uses timm-style keys like `conv_stem.*`, `blocks.*`, `classifier.*`.
    """
    _ensure_timm()
    path = Path(path)
    print(f"Loading EfficientNet-B3 weights from: {path}")

    ckpt = torch.load(path, map_location="cpu")
    state = _extract_state_dict(ckpt)
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format for EfficientNetB3.pt")
    state = _strip_prefixes(state)

    # Infer num_classes from classifier weights
    if "classifier.weight" not in state:
        raise ValueError("Checkpoint missing `classifier.weight`; cannot infer num_classes.")
    num_classes = int(state["classifier.weight"].shape[0])

    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def analyze_eye_image(image_bytes: bytes, model) -> Dict[str, object]:
    """
    End-to-end inference:
    - Standardize fundus (auto-crop + Ben Graham), resize to model input
    - Softmax probabilities
    - Grad-CAM overlay on standardized image
    """
    # Determine model input size (default to 300 for EfficientNet-B3)
    input_size = 300
    cfg = getattr(model, "default_cfg", None)
    if isinstance(cfg, dict):
        sz = cfg.get("input_size")
        if isinstance(sz, (tuple, list)) and len(sz) == 3:
            input_size = int(sz[1])

    standardized = standardize_fundus_image(image_bytes, out_size=input_size)
    x = _tensor_from_pil(standardized).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    # Map to labels (fallback to generic labels if mismatch)
    labels = LABELS if len(LABELS) == len(probs) else [f"Class {i}" for i in range(len(probs))]
    confidence_scores = {lab: float(p) for lab, p in zip(labels, probs.tolist())}
    top_idx = int(probs.argmax())
    diagnosis = labels[top_idx]

    # Grad-CAM: for timm EfficientNet, `conv_head` is the last conv feature layer
    target_layer = getattr(model, "conv_head", None)
    if target_layer is None:
        # Fallback: try last block's project conv if present
        try:
            target_layer = model.blocks[-1][-1].conv_pwl  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Could not locate a suitable target layer for Grad-CAM.") from e

    if GradCAM is not None and show_cam_on_image is not None and ClassifierOutputTarget is not None:
        cam = GradCAM(
            model=model, target_layers=[target_layer], use_cuda=DEVICE.type == "cuda"
        )  # type: ignore[arg-type]
        grayscale_cam = cam(
            input_tensor=x,
            targets=[ClassifierOutputTarget(top_idx)],  # type: ignore[call-arg]
        )[0]
    else:
        grayscale_cam = _gradcam_fallback(model=model, x=x, target_layer=target_layer, class_idx=top_idx)

    rgb_np = np.array(standardized).astype(np.float32) / 255.0
    if show_cam_on_image is not None:
        overlay = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)  # type: ignore[call-arg]
    else:
        # Minimal overlay if pytorch-grad-cam isn't installed
        heat = cv2.applyColorMap((grayscale_cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        overlay = np.clip(rgb_np * 0.55 + heat * 0.45, 0, 1)
        overlay = (overlay * 255).astype(np.uint8)

    out = Image.fromarray(overlay)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "diagnosis": diagnosis,
        "confidence_scores": confidence_scores,
        "heatmap_base64": heatmap_b64,
    }

