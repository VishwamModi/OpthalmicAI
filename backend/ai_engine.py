from __future__ import annotations

import csv
import json
import math
import os
import base64
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torch.cuda.amp import autocast
from torchvision import transforms

try:
    import segmentation_models_pytorch as smp
except ImportError:  # pragma: no cover
    smp = None  # type: ignore

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    # Guard against a known edge case where GradCAM's __del__ assumes
    # activations_and_grads exists even if initialization failed.
    try:  # pragma: no cover
        from pytorch_grad_cam.base_cam import BaseCAM

        def _safe_basecam_del(self) -> None:  # type: ignore[override]
            try:
                aag = getattr(self, "activations_and_grads", None)
                if aag is not None:
                    aag.release()
            except Exception:
                pass

        BaseCAM.__del__ = _safe_basecam_del  # type: ignore[assignment]
    except Exception:  # pragma: no cover
        pass
except ImportError:  # pragma: no cover
    GradCAM = None  # type: ignore
    show_cam_on_image = None  # type: ignore
    ClassifierOutputTarget = None  # type: ignore


IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEG_INPUT_SIZE = 512


def _env_enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


ENABLE_SEGMENTATION = _env_enabled("ENABLE_SEGMENTATION", True)
ENABLE_GRADCAM = _env_enabled("ENABLE_GRADCAM", True)

SEGMENTATION_SPECS: dict[str, dict[str, Any]] = {
    "OD": {
        "arch": "UnetPlusPlus",
        "encoder": "resnet34",
        "decoder_attention_type": "scse",
        "weight": "best_od_model_fp16.pth",
        "threshold": 0.5,
    },
    "EX": {
        "arch": "UnetPlusPlus",
        "encoder": "efficientnet-b2",
        "decoder_attention_type": "scse",
        "weight": "best_ex_model_fp16.pth",
        "threshold": 0.4,
    },
    "SE": {
        "arch": "UnetPlusPlus",
        "encoder": "efficientnet-b2",
        "decoder_attention_type": "scse",
        "weight": "best_se_model_fp16.pth",
        "threshold": 0.4,
    },
    "MA": {
        "arch": "UnetPlusPlus",
        "encoder": "efficientnet-b4",
        "decoder_attention_type": "scse",
        "weight": "best_ma_model_fp16.pth",
        "threshold": 0.3,
    },
    "HE": {
        "arch": "UnetPlusPlus",
        "encoder": "efficientnet-b4",
        "decoder_attention_type": "scse",
        "weight": "best_he_model_fp16.pth",
        "threshold": 0.3,
    },
}

SEGMENTATION_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "OD": (80, 120, 255),   # blue
    "EX": (255, 193, 7),    # amber
    "SE": (156, 39, 176),   # purple
    "MA": (244, 67, 54),    # red
    "HE": (121, 85, 72),    # brown
}


def _coerce_binary_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0.5 else 0

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "positive", "pos", "dr", "referable", "diabetic retinopathy"}:
        return 1
    if normalized in {"0", "false", "no", "n", "negative", "neg", "normal", "non-referable", "nonreferable", "no dr"}:
        return 0
    raise ValueError(f"Unsupported cohort label value: {value!r}")


def _coerce_probability(value: Any) -> float:
    probability = float(value)
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"Predicted probability must be between 0 and 1, got {value!r}")
    return probability


def _thresholds(step: float = 0.05) -> list[float]:
    if step <= 0 or step > 1:
        raise ValueError("Threshold step must be between 0 and 1.")

    intervals = int(round(1.0 / step))
    thresholds = [round(index * step, 2) for index in range(intervals)]
    if not thresholds or thresholds[-1] != 1.0:
        thresholds.append(1.0)
    return thresholds


def _parse_cohort_payload(payload: Any) -> tuple[list[int], list[float]]:
    if isinstance(payload, dict):
        if {"true_labels", "predicted_probabilities"}.issubset(payload):
            labels = payload["true_labels"]
            probabilities = payload["predicted_probabilities"]
        elif "data" in payload and isinstance(payload["data"], list):
            labels = []
            probabilities = []
            for row in payload["data"]:
                if not isinstance(row, dict):
                    raise ValueError("Each validation cohort row must be an object.")
                labels.append(
                    row.get("true_label", row.get("label", row.get("y_true", row.get("target"))))
                )
                probabilities.append(
                    row.get(
                        "predicted_probability",
                        row.get("probability", row.get("y_prob", row.get("score"))),
                    )
                )
            return [
                _coerce_binary_label(value) for value in labels
            ], [
                _coerce_probability(value) for value in probabilities
            ]
        else:
            raise ValueError("Unsupported JSON cohort format.")
    elif isinstance(payload, list):
        labels = []
        probabilities = []
        for row in payload:
            if isinstance(row, dict):
                labels.append(row.get("true_label", row.get("label", row.get("y_true", row.get("target")))))
                probabilities.append(
                    row.get(
                        "predicted_probability",
                        row.get("probability", row.get("y_prob", row.get("score"))),
                    )
                )
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                labels.append(row[0])
                probabilities.append(row[1])
            else:
                raise ValueError("Unsupported validation cohort row format.")
        return [
            _coerce_binary_label(value) for value in labels
        ], [
            _coerce_probability(value) for value in probabilities
        ]

    raise ValueError("Unsupported validation cohort payload format.")


def load_validation_cohort(source_path: str | Path | None = None) -> tuple[list[int], list[float]]:
    """Load validation labels/probabilities from JSON or CSV."""
    candidate_paths: list[Path] = []

    if source_path is not None:
        candidate_paths.append(Path(source_path))

    env_path = os.getenv("DCA_VALIDATION_COHORT_PATH")
    if env_path:
        candidate_paths.append(Path(env_path))

    base_dir = Path(__file__).resolve().parent
    candidate_paths.extend(
        [
            base_dir / "validation_cohort.json",
            base_dir / "validation_cohort.csv",
        ]
    )

    for path in candidate_paths:
        if not path.exists():
            continue

        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            labels, probabilities = _parse_cohort_payload(payload)
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            if not rows:
                raise ValueError(f"Validation cohort CSV is empty: {path}")
            labels, probabilities = _parse_cohort_payload(rows)
        else:
            continue

        if len(labels) != len(probabilities):
            raise ValueError("Validation cohort labels and probabilities must be the same length.")
        if not labels:
            raise ValueError("Validation cohort must contain at least one sample.")
        return labels, probabilities

    raise FileNotFoundError(
        "Validation cohort not found. Set DCA_VALIDATION_COHORT_PATH to a JSON or CSV file with true labels and predicted probabilities."
    )


def calculate_decision_curve_data(
    true_labels: Iterable[int | bool | float | str],
    predicted_probabilities: Iterable[float | int | str],
    threshold_step: float = 0.05,
) -> list[dict[str, float]]:
    labels = [_coerce_binary_label(label) for label in true_labels]
    probabilities = [_coerce_probability(probability) for probability in predicted_probabilities]

    if len(labels) != len(probabilities):
        raise ValueError("True labels and predicted probabilities must have the same length.")
    if not labels:
        raise ValueError("At least one cohort sample is required for DCA.")

    cohort_size = float(len(labels))
    prevalence = sum(labels) / cohort_size
    results: list[dict[str, float]] = []

    for threshold in _thresholds(threshold_step):
        if threshold >= 1.0:
            model_net_benefit = 0.0
            treat_all_net_benefit = 0.0
        else:
            threshold_weight = threshold / (1.0 - threshold)
            predicted_positive = [int(probability >= threshold) for probability in probabilities]
            true_positive = sum(1 for label, prediction in zip(labels, predicted_positive) if label == 1 and prediction == 1)
            false_positive = sum(1 for label, prediction in zip(labels, predicted_positive) if label == 0 and prediction == 1)
            model_net_benefit = (true_positive / cohort_size) - (false_positive / cohort_size) * threshold_weight
            treat_all_net_benefit = prevalence - (1.0 - prevalence) * threshold_weight

        results.append(
            {
                "threshold": round(threshold, 2),
                "model": round(model_net_benefit, 4),
                "treatAll": round(treat_all_net_benefit, 4),
                "treatNone": 0.0,
            }
        )

    return results


def is_valid_retina_scan(cv2_img: np.ndarray) -> bool:
    """
    Heuristic validation that the input is a retinal fundus image:
    - Convert to grayscale
    - Threshold
    - Find largest contour
    - Require that contour area is large and roughly circular
    """
    if cv2_img is None or cv2_img.size == 0:
        return False

    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    h, w = gray.shape[:2]
    img_area = float(h * w)

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 0.15 * img_area:
        # Fundus disc is usually a dominant circle in the frame
        return False

    perimeter = cv2.arcLength(largest, True)
    if perimeter <= 0:
        return False

    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    # Ideal circle has circularity ~1. Real-world fundus often ~0.7–0.95.
    return circularity > 0.6


def _preprocess_fundus(image_bytes: bytes) -> torch.Tensor:
    """
    EXACT preprocessing replication (pixel-for-pixel intent) from training:
    1) Decode bytes to OpenCV BGR
    2) Validate is an eye (is_valid_retina_scan)
    3) Resize to 300x300
    4) Circular mask:
       center=(150,150), radius=int(150*0.95)
    5) Extract Green channel, enhance via CLAHE
    6) Triplicate enhanced green channel:
       img = cv2.merge([g_enhanced, g_enhanced, g_enhanced])
    7) Convert BGR -> RGB, then PIL
    8) transforms.ToTensor + Normalize(ImageNet)
    9) Add batch dim, move to device
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode input image bytes.")

    if not is_valid_retina_scan(bgr):
        raise ValueError("Uploaded image does not appear to be a retinal fundus scan.")

    bgr = cv2.resize(bgr, (300, 300), interpolation=cv2.INTER_AREA)

    # Fixed mask geometry as specified in your training pipeline.
    center = (150, 150)
    radius = int(150 * 0.95)
    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    bgr = cv2.bitwise_and(bgr, bgr, mask=mask)

    # Extract green channel (OpenCV BGR channel order => index 1).
    g = bgr[:, :, 1]

    # CLAHE enhancement on the green channel.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_enhanced = clahe.apply(g)

    # Triplicate enhanced green channel to build 3-channel image.
    img = cv2.merge([g_enhanced, g_enhanced, g_enhanced])

    # Convert BGR->RGB before PIL.
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    x = tfm(pil).unsqueeze(0)
    return x.to(DEVICE)


def _encode_png_base64(rgb_uint8: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode PNG.")
    import base64

    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _encode_mask_png_base64(mask_uint8: np.ndarray) -> str:
    mask_u8 = (mask_uint8.astype(np.uint8) * 255)
    ok, buf = cv2.imencode(".png", mask_u8)
    if not ok:
        raise RuntimeError("Failed to encode mask PNG.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _clean_binary_mask(mask: np.ndarray, min_area: int = 6) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary)
    out = np.zeros_like(binary)
    for label in range(1, num_labels):
        area = int(np.sum(labels == label))
        if area >= min_area:
            out[labels == label] = 1
    return out


def _preprocess_segmentation_image(image_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode input image bytes.")

    if not is_valid_retina_scan(bgr):
        raise ValueError("Uploaded image does not appear to be a retinal fundus scan.")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (SEG_INPUT_SIZE, SEG_INPUT_SIZE), interpolation=cv2.INTER_AREA)

    pil = Image.fromarray(rgb)
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )
    x = tfm(pil).unsqueeze(0).to(DEVICE)
    return x, rgb


def _largest_component_centroid_and_diameter(mask: np.ndarray) -> tuple[tuple[float, float] | None, float]:
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None, 0.0

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None, 0.0

    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])
    diameter = math.sqrt((4.0 * area) / math.pi)
    return (cx, cy), diameter


def extract_clinical_features(masks: dict[str, np.ndarray]) -> dict[str, Any]:
    ma = masks.get("MA")
    he = masks.get("HE")
    ex = masks.get("EX")
    se = masks.get("SE")
    od = masks.get("OD")

    if ma is None:
        ma = np.zeros((SEG_INPUT_SIZE, SEG_INPUT_SIZE), dtype=np.uint8)
    if he is None:
        he = np.zeros((SEG_INPUT_SIZE, SEG_INPUT_SIZE), dtype=np.uint8)
    if ex is None:
        ex = np.zeros((SEG_INPUT_SIZE, SEG_INPUT_SIZE), dtype=np.uint8)
    if se is None:
        se = np.zeros((SEG_INPUT_SIZE, SEG_INPUT_SIZE), dtype=np.uint8)
    if od is None:
        od = np.zeros((SEG_INPUT_SIZE, SEG_INPUT_SIZE), dtype=np.uint8)

    ma = ma.astype(np.uint8)
    he = he.astype(np.uint8)
    ex = ex.astype(np.uint8)
    se = se.astype(np.uint8)
    od = od.astype(np.uint8)

    ma = _clean_binary_mask(ma, min_area=4)
    he = _clean_binary_mask(he, min_area=8)
    ex = _clean_binary_mask(ex, min_area=8)
    se = _clean_binary_mask(se, min_area=8)
    od = _clean_binary_mask(od, min_area=80)

    ma_components = cv2.connectedComponents(ma)[0] - 1
    he_components = cv2.connectedComponents(he)[0] - 1
    ex_area_pixels = int(np.count_nonzero(ex))
    se_area_pixels = int(np.count_nonzero(se))

    od_center, od_diameter = _largest_component_centroid_and_diameter(od)
    dme_risk_flag = False
    macula_centers: list[tuple[int, int]] = []
    macula_radius = 0

    if od_center and od_diameter > 0:
        cx, cy = od_center
        offset = 2.5 * od_diameter
        macula_radius = max(12, int(0.65 * od_diameter))
        left_center = (int(round(cx - offset)), int(round(cy)))
        right_center = (int(round(cx + offset)), int(round(cy)))
        macula_centers = [left_center, right_center]

        ex_points = np.column_stack(np.where(ex > 0))  # (y, x)
        if ex_points.size > 0:
            for my, mx in [(left_center[1], left_center[0]), (right_center[1], right_center[0])]:
                d2 = (ex_points[:, 1] - mx) ** 2 + (ex_points[:, 0] - my) ** 2
                if np.any(d2 <= (macula_radius ** 2)):
                    dme_risk_flag = True
                    break

    return {
        "ma_count": int(max(0, ma_components)),
        "he_count": int(max(0, he_components)),
        "ex_area_pixels": int(ex_area_pixels),
        "se_area_pixels": int(se_area_pixels),
        "dme_risk_flag": bool(dme_risk_flag),
        "od_center": od_center,
        "od_diameter": float(od_diameter),
        "macula_centers": macula_centers,
        "macula_radius": int(macula_radius),
    }


def _gradcam_overlay_b64(model: torch.nn.Module, x: torch.Tensor) -> str | None:
    """
    Generate Grad-CAM overlay on the preprocessed 300x300 image.
    Returns base64 PNG string, or None if grad-cam is unavailable.
    """
    if not ENABLE_GRADCAM:
        return None

    try:
        if GradCAM is None or show_cam_on_image is None:
            return None

        # Prefer timm EfficientNet conv_head; otherwise fall back to last Conv2d.
        target_layer = getattr(model, "conv_head", None)
        if target_layer is None:
            last_conv: torch.nn.Module | None = None
            for m in model.modules():
                if isinstance(m, torch.nn.Conv2d):
                    last_conv = m
            target_layer = last_conv
        if target_layer is None:
            return None

        # Convert tensor back to 0..1 RGB image for overlay.
        img = x[0].detach().float().cpu()
        mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
        std = torch.tensor(IMAGENET_STD)[:, None, None]
        img = (img * std + mean).clamp(0, 1)
        rgb_np = img.permute(1, 2, 0).numpy().astype(np.float32)

        # grad-cam API changed across versions; some accept use_cuda, others don't.
        import inspect

        try:
            sig = inspect.signature(GradCAM.__init__)  # type: ignore[misc]
            if "use_cuda" in sig.parameters:
                cam = GradCAM(  # type: ignore[arg-type]
                    model=model, target_layers=[target_layer], use_cuda=DEVICE.type == "cuda"
                )
            else:
                cam = GradCAM(model=model, target_layers=[target_layer])  # type: ignore[arg-type]
        except Exception:
            cam = GradCAM(model=model, target_layers=[target_layer])  # type: ignore[arg-type]

        # Regression output: use a simple target that maximizes the scalar prediction.
        class _RegressionTarget:
            def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
                # model_output shape: [B, 1] or [B]
                if model_output.ndim == 2:
                    return model_output[:, 0]
                return model_output

        grayscale_cam = cam(input_tensor=x, targets=[_RegressionTarget()])[0]  # type: ignore[call-arg]
        overlay = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)
        return _encode_png_base64(overlay)
    except Exception as e:  # pragma: no cover
        # Don't crash inference if Grad-CAM fails; log once for diagnosis.
        try:
            print(f"[AIEngine] Grad-CAM generation failed: {type(e).__name__}: {e}")
        except Exception:
            pass
        return None


def _build_effnet_b3() -> torch.nn.Module:
    """Create an EfficientNet-B3 backbone with a single regression/logit output."""
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=1)
    return model


class AIEngine:
    """
    Singleton service that owns all three models and exposes a high-level
    `predict_disease(image_bytes)` API that returns a dict shaped like MOCK_RESULTS.
    """

    _instance: "AIEngine | None" = None

    def __init__(
        self,
        dr_path: Path,
        segmentation_dir: Path,
    ) -> None:
        if ENABLE_SEGMENTATION and smp is None:
            raise ImportError(
                "segmentation-models-pytorch is required for segmentation inference. Install it in backend requirements."
            )

        self.device = DEVICE
        torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))

        self.dr_model = _build_effnet_b3()
        self.segmentation_models: dict[str, torch.nn.Module] = {}

        self._load_weights(self.dr_model, dr_path)
        if ENABLE_SEGMENTATION:
            self._load_segmentation_models(segmentation_dir)

        self.dr_model.to(self.device).eval()

    @classmethod
    def instance(cls) -> "AIEngine":
        if cls._instance is None:
            base = Path(__file__).resolve().parent
            project_root = base.parent

            dr_candidates = [
                base / "OphthalmicAI_Final_EffNetB3.pt",
                project_root / "Models" / "OphthalmicAI_Final_EffNetB3.pt",
                project_root.parent / "Models" / "OphthalmicAI_Final_EffNetB3.pt",
            ]
            dr_path = next((p for p in dr_candidates if p.exists()), dr_candidates[0])

            segmentation_candidates = [
                project_root / "Models" / "segmentation",
                project_root.parent / "Models" / "segmentation",
            ]
            segmentation_dir = next((p for p in segmentation_candidates if p.exists()), segmentation_candidates[0])

            cls._instance = cls(dr_path=dr_path, segmentation_dir=segmentation_dir)
        return cls._instance

    def _load_segmentation_models(self, segmentation_dir: Path) -> None:
        if not segmentation_dir.exists():
            raise FileNotFoundError(f"Segmentation directory not found: {segmentation_dir}")

        for key, spec in SEGMENTATION_SPECS.items():
            arch = str(spec["arch"])
            encoder = str(spec["encoder"])
            decoder_attention_type = spec.get("decoder_attention_type")
            weight_name = str(spec["weight"])
            weight_path = segmentation_dir / weight_name

            if not weight_path.exists():
                raise FileNotFoundError(f"Missing segmentation weights for {key}: {weight_path}")

            model_ctor = getattr(smp, arch, None)
            if model_ctor is None:
                raise ValueError(f"Unsupported segmentation architecture: {arch}")

            model_kwargs: dict[str, Any] = {
                "encoder_name": encoder,
                "encoder_weights": None,
                "in_channels": 3,
                "classes": 1,
            }
            if decoder_attention_type:
                model_kwargs["decoder_attention_type"] = decoder_attention_type

            model = model_ctor(**model_kwargs)
            self._load_weights(model, weight_path)
            model.to(self.device).eval()
            self.segmentation_models[key] = model

    @staticmethod
    def _load_weights(model: torch.nn.Module, path: Path) -> None:
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        cleaned: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            nk = k
            for prefix in ("module.", "model."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix) :]
                    break
            cleaned[nk] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            preview = ", ".join(missing[:5])
            print(f"[AIEngine] Missing keys for {path.name}: {len(missing)} (e.g. {preview})")
        if unexpected:
            preview = ", ".join(unexpected[:5])
            print(f"[AIEngine] Unexpected keys for {path.name}: {len(unexpected)} (e.g. {preview})")

    def _run_dr_model(self, x: torch.Tensor) -> float:
        """Run the single DR regression model and return the raw float output."""
        amp_ctx = (
            autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), amp_ctx:
            dr_out = self.dr_model(x)  # regression scalar
        return float(dr_out.view(-1)[0].cpu().item())

    @staticmethod
    def _dr_stage(score: float) -> int:
        """Map DR regression score to 0–4 stages using [0.5, 1.5, 2.5, 3.5]."""
        if score < 0.5:
            return 0
        if score < 1.5:
            return 1
        if score < 2.5:
            return 2
        if score < 3.5:
            return 3
        return 4

    @staticmethod
    def _dr_stage_label(stage: int) -> str:
        return [
            "No DR",
            "Mild NPDR",
            "Moderate NPDR",
            "Severe NPDR",
            "Proliferative DR",
        ][stage]

    def _build_recommendation(
        self,
        dr_stage: int,
    ) -> str:
        if dr_stage == 0:
            return "No referable diabetic retinopathy. Continue routine annual screening per clinical protocol."
        if dr_stage == 1:
            return "Mild NPDR. Recommend repeat imaging in 6–12 months and optimize glycemic control."
        if dr_stage == 2:
            return "Moderate NPDR. Recommend ophthalmology follow-up within 4–8 weeks and consider OCT for macular assessment."
        if dr_stage == 3:
            return "Severe NPDR. Recommend timely referral to a retina specialist within 4–6 weeks."
        return "Proliferative DR suspected. Recommend urgent retina evaluation and treatment planning as indicated."

    def _run_segmentation_models(self, image_bytes: bytes) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        x_seg, rgb_512 = _preprocess_segmentation_image(image_bytes)
        masks: dict[str, np.ndarray] = {}

        amp_ctx = (
            autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )

        with torch.no_grad(), amp_ctx:
            for key, model in self.segmentation_models.items():
                output = model(x_seg)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                probs = torch.sigmoid(output)
                threshold = float(SEGMENTATION_SPECS[key]["threshold"])
                mask = (probs >= threshold).detach().cpu().numpy()[0, 0].astype(np.uint8)
                min_area = 80 if key == "OD" else 6
                masks[key] = _clean_binary_mask(mask, min_area=min_area)

        # Build professional visual outputs
        combined = rgb_512.copy().astype(np.float32)
        mask_images_b64: dict[str, str] = {}
        feature_images_b64: dict[str, str] = {}

        for key, mask in masks.items():
            color = np.array(SEGMENTATION_COLORS_RGB.get(key, (255, 255, 255)), dtype=np.float32)
            alpha = 0.35 if key != "OD" else 0.2
            idx = mask > 0
            if np.any(idx):
                combined[idx] = (1 - alpha) * combined[idx] + alpha * color
            mask_images_b64[key] = _encode_mask_png_base64(mask)

            color_mask = np.zeros_like(rgb_512, dtype=np.uint8)
            color_mask[mask > 0] = SEGMENTATION_COLORS_RGB.get(key, (255, 255, 255))
            feature_images_b64[key.lower()] = _encode_png_base64(color_mask)

        segmentation_images = {
            "original_base64": _encode_png_base64(rgb_512),
            "combined_overlay_base64": _encode_png_base64(combined.clip(0, 255).astype(np.uint8)),
            "masks_base64": mask_images_b64,
            "feature_images_base64": feature_images_b64,
        }
        return masks, segmentation_images

    @staticmethod
    def _status_for_count(count: int, warning: int, elevated: int) -> str:
        if count <= 0:
            return "normal"
        if count <= warning:
            return "warning"
        if count <= elevated:
            return "elevated"
        return "critical"

    @staticmethod
    def _status_for_area(area_pixels: int, warning: int, elevated: int) -> str:
        if area_pixels <= warning:
            return "normal"
        if area_pixels <= elevated:
            return "warning"
        if area_pixels <= elevated * 2:
            return "elevated"
        return "critical"

    def _build_clinical_features_from_masks(self, extracted: dict[str, Any]) -> list[dict[str, Any]]:
        ma_count = int(extracted.get("ma_count", 0))
        he_count = int(extracted.get("he_count", 0))
        ex_area = int(extracted.get("ex_area_pixels", 0))
        se_area = int(extracted.get("se_area_pixels", 0))
        dme_risk = bool(extracted.get("dme_risk_flag", False))

        ma_status = self._status_for_count(ma_count, warning=5, elevated=20)
        he_status = self._status_for_count(he_count, warning=3, elevated=12)
        ex_status = self._status_for_area(ex_area, warning=300, elevated=1800)
        se_status = self._status_for_area(se_area, warning=250, elevated=1400)
        dme_status = "critical" if dme_risk else "normal"

        def score(value: float) -> float:
            return float(max(0.01, min(0.99, value)))

        return [
            {
                "name": "Microaneurysms",
                "value": ma_count,
                "unit": "components",
                "reference": "0 (normal)",
                "status": ma_status,
                "score": round(score(ma_count / 30.0), 2),
                "mask_key": "ma",
            },
            {
                "name": "Hemorrhages",
                "value": he_count,
                "unit": "components",
                "reference": "0 (normal)",
                "status": he_status,
                "score": round(score(he_count / 20.0), 2),
                "mask_key": "he",
            },
            {
                "name": "Hard Exudate Area",
                "value": ex_area,
                "unit": "pixels",
                "reference": "<300 px",
                "status": ex_status,
                "score": round(score(ex_area / 5000.0), 2),
                "mask_key": "ex",
            },
            {
                "name": "Soft Exudate Area",
                "value": se_area,
                "unit": "pixels",
                "reference": "<250 px",
                "status": se_status,
                "score": round(score(se_area / 4500.0), 2),
                "mask_key": "se",
            },
            {
                "name": "DME Risk",
                "value": "High" if dme_risk else "Low",
                "unit": "flag",
                "reference": "No exudates near estimated macula",
                "status": dme_status,
                "score": 0.95 if dme_risk else 0.1,
                "mask_key": "od",
            },
        ]

    def generate_reports(
        self,
        dr_stage: int,
        dr_score: float,
        clinical_features: list[dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Dual-report generation (clinical vs patient-friendly).
        Uses Gemini via google-generativeai; failures return safe fallbacks.
        """
        from .llm_service import generate_dual_clinical_reports

        return generate_dual_clinical_reports(
            dr_stage=dr_stage,
            dr_score=dr_score,
            clinical_features=clinical_features,
        )

    def predict_disease(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Single-model DR regression inference.
        Returns a JSON-serializable payload tailored for the DR-focused UI.
        """
        x = _preprocess_fundus(image_bytes)
        gradcam_b64 = _gradcam_overlay_b64(self.dr_model, x)
        raw_score = self._run_dr_model(x)
        dr_score = round(raw_score, 2)

        dr_stage = self._dr_stage(raw_score)  # 0..4
        max_severity = 4

        diagnosis = "Diabetic Retinopathy" if dr_stage > 0 else "No Referable DR"
        etdrsLevel = self._dr_stage_label(dr_stage)

        # The checkpoint is an ordinal regressor, not a calibrated probabilistic
        # classifier. Expose deterministic proximity to the selected integer
        # grade for UI context, and label it explicitly as uncalibrated.
        clipped_score = max(0.0, min(float(max_severity), raw_score))
        ordinal_distance = abs(clipped_score - float(dr_stage))
        ai_confidence = round(100.0 * max(0.0, 1.0 - ordinal_distance), 1)

        recommendation = self._build_recommendation(dr_stage)

        # Lesion segmentation is optional because the free Render instance has
        # 512 MB RAM. Research and local deployments can enable it explicitly.
        if ENABLE_SEGMENTATION:
            masks, segmentation_images = self._run_segmentation_models(image_bytes)
            extracted = extract_clinical_features(masks)
            clinical_features = self._build_clinical_features_from_masks(extracted)
        else:
            segmentation_images = {}
            extracted = {}
            clinical_features = []

        severity_stages = [
            {
                "level": 0,
                "label": "No DR",
                "description": "No apparent diabetic retinopathy detected.",
                "current": dr_stage == 0,
            },
            {
                "level": 1,
                "label": "Mild NPDR",
                "description": "Microaneurysms only.",
                "current": dr_stage == 1,
            },
            {
                "level": 2,
                "label": "Moderate NPDR",
                "description": "More than microaneurysms but less than severe NPDR.",
                "current": dr_stage == 2,
            },
            {
                "level": 3,
                "label": "Severe NPDR",
                "description": "Extensive intraretinal hemorrhages and venous beading.",
                "current": dr_stage == 3,
            },
            {
                "level": 4,
                "label": "Proliferative DR",
                "description": "Neovascularization or vitreous / pre-retinal hemorrhage.",
                "current": dr_stage == 4,
            },
        ]

        # Dual LLM reporting (clinical + patient-friendly)
        dual = self.generate_reports(
            dr_stage=dr_stage, dr_score=dr_score, clinical_features=clinical_features
        )
        clinical_report = str(dual.get("clinical_report") or "").strip()
        patient_report = str(dual.get("patient_report") or "").strip()
        if not clinical_report:
            clinical_report = (
                f"Diagnosis is {diagnosis} ({etdrsLevel}) with model score {dr_score:.2f}. "
                f"Recommendation: {recommendation}"
            )
        if not patient_report:
            patient_report = (
                f"Your scan suggests {diagnosis} ({etdrsLevel}). "
                "Please follow your doctor’s advice and attend regular follow-up visits."
            )

        dca_curve_data: list[dict[str, float]] = []
        try:
            true_labels, predicted_probabilities = load_validation_cohort()
            dca_curve_data = calculate_decision_curve_data(true_labels, predicted_probabilities)
        except Exception:
            # DCA is optional at inference time; the dedicated endpoint handles validation-cohort requests.
            dca_curve_data = []

        icd10_by_stage = {
            0: "Z13.5",
            1: "E11.3291",
            2: "E11.3391",
            3: "E11.3491",
            4: "E11.3591",
        }

        return {
            "diagnosis": diagnosis,
            "etdrsLevel": etdrsLevel,
            "icd10": icd10_by_stage.get(dr_stage, "E11.319"),
            "severity": dr_stage,
            "maxSeverity": max_severity,
            "dr_score": dr_score,
            "aiConfidence": ai_confidence,
            "confidenceType": "ordinal_proximity_uncalibrated",
            "modelVersion": "Opthalmic v1.0 (EfficientNet-B3)",
            "recommendation": recommendation,
            "clinicalFeatures": clinical_features,
            "clinical_report": clinical_report,
            "patient_report": patient_report,
            # Backward-compatible alias for any UI still rendering `generatedReport`.
            "generatedReport": clinical_report,
            "gradcam_base64": gradcam_b64,
            "segmentation_images": segmentation_images,
            "clinical_features": extracted,
            "dcaCurveData": dca_curve_data,
            "severityStages": severity_stages,
        }
