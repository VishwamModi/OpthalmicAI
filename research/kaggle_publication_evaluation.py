# %% [markdown]
# # OphthalmicAI - Publication-grade evaluation and figure export
#
# Run this file as a Kaggle notebook (File -> Import Notebook, or paste the
# cells separated by `# %%`). It evaluates the saved DR, glaucoma, cataract,
# and lesion-segmentation checkpoints without inventing metrics.
#
# Important: use an untouched test set for the journal. If a split was used
# for checkpoint selection, label it "internal validation", not "test".

# %%
# Kaggle setup. Internet must be enabled for pip installation.
!pip -q install timm segmentation-models-pytorch grad-cam openpyxl

# %%
from __future__ import annotations

import json
import hashlib
import math
import os
import random
import time
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score, jaccard_score, matthews_corrcoef,
    mean_absolute_error, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    GradCAM = None
    show_cam_on_image = None

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT = Path("/kaggle/working/publication_results")
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
EXAMPLES = OUTPUT / "examples"
for folder in (OUTPUT, FIGURES, TABLES, EXAMPLES):
    folder.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -------------------------- EDIT THESE PATHS --------------------------
DR_CSV = Path("/kaggle/input/aptos2019-blindness-detection/train.csv")
DR_IMAGE_DIR = Path("/kaggle/input/aptos2019-blindness-detection/train_images")
DR_MODEL = Path("/kaggle/input/YOUR-MODEL-DATASET/OphthalmicAI_Final_EffNetB3.pt")

EYE_DISEASE_DIR = Path("/kaggle/input/eye-diseases-classification/dataset")
GLAUCOMA_MODEL = Path("/kaggle/input/YOUR-MODEL-DATASET/Glaucoma_EffNetB3.pt")
CATARACT_MODEL = Path("/kaggle/input/YOUR-MODEL-DATASET/Cataract_EffNetB3.pt")

SEGMENTATION_MODEL_DIR = Path("/kaggle/input/YOUR-MODEL-DATASET/segmentation")

# Optional evidence files. Leave as None when unavailable.
# Training history columns: model, phase, epoch, train_loss, val_loss and any
# recorded metrics such as qwk, auc, dice, iou, learning_rate.
TRAINING_HISTORY_CSV = None  # Path("/kaggle/input/YOUR-HISTORY/training_history.csv")
# DME predictions columns: y_true, probability.
DME_PREDICTIONS_CSV = None  # Path("/kaggle/input/YOUR-DME/dme_predictions.csv")
# Optional DR metadata columns are evaluated when present.
SUBGROUP_COLUMNS = ["sex", "age_group", "camera", "site"]

# Each CSV must contain: image_path, mask_path. Paths may be absolute Kaggle
# paths or relative to /kaggle/input. Create one CSV per biomarker.
SEGMENTATION_MANIFESTS = {
    "OD": Path("/kaggle/input/YOUR-MANIFESTS/od_test_manifest.csv"),
    "EX": Path("/kaggle/input/YOUR-MANIFESTS/ex_test_manifest.csv"),
    "SE": Path("/kaggle/input/YOUR-MANIFESTS/se_test_manifest.csv"),
    "MA": Path("/kaggle/input/YOUR-MANIFESTS/ma_test_manifest.csv"),
    "HE": Path("/kaggle/input/YOUR-MANIFESTS/he_test_manifest.csv"),
}

# `validation` reproduces the original APTOS 85/15 split. For the journal,
# prefer `external_test` and point DR_CSV to a CSV with id_code + diagnosis.
DR_EVALUATION_LABEL = "internal_validation"
DR_TEST_SIZE = 0.15
BATCH_SIZE = 32
NUM_WORKERS = 2
N_BOOTSTRAP = 1000
# --------------------------------------------------------------------

print("Device:", DEVICE)
print("Outputs:", OUTPUT)
model_metadata_rows = []

# %% [markdown]
# ## Shared utilities

# %%
def save_figure(name: str):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def load_state(model: torch.nn.Module, path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(path.name, "| missing:", len(missing), "| unexpected:", len(unexpected))
    return model


def checkpoint_metadata(name, model, path, inference_seconds=None, n_images=None):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    row = {
        "model": name,
        "checkpoint": str(path),
        "sha256": digest.hexdigest(),
        "size_mb": path.stat().st_size / (1024 ** 2),
        "parameters_total": total,
        "parameters_trainable": trainable,
        "device": str(DEVICE),
    }
    if inference_seconds is not None and n_images:
        row["inference_total_seconds"] = inference_seconds
        row["images_evaluated"] = n_images
        row["milliseconds_per_image"] = 1000 * inference_seconds / n_images
        row["images_per_second"] = n_images / inference_seconds
    model_metadata_rows.append(row)


def bootstrap_ci(y_true, y_pred, metric, n=N_BOOTSTRAP, seed=SEED):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            value = float(metric(y_true[idx], y_pred[idx]))
            if np.isfinite(value):
                values.append(value)
        except Exception:
            pass
    if not values:
        return np.nan, np.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def specificity_per_class(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        rows.append({
            "class_id": label,
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
            "npv": tn / (tn + fn) if tn + fn else np.nan,
            "support": int(tp + fn),
        })
    return pd.DataFrame(rows)


def decision_curve(y_true, probability, name):
    y_true, probability = np.asarray(y_true), np.asarray(probability)
    thresholds = np.arange(0.05, 0.96, 0.05)
    prevalence = y_true.mean()
    records = []
    for threshold in thresholds:
        pred = probability >= threshold
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        weight = threshold / (1 - threshold)
        records.append({
            "threshold": threshold,
            "model": tp / len(y_true) - fp / len(y_true) * weight,
            "treat_all": prevalence - (1 - prevalence) * weight,
            "treat_none": 0.0,
        })
    frame = pd.DataFrame(records)
    frame.to_csv(TABLES / f"{name}_decision_curve.csv", index=False)
    plt.figure(figsize=(7, 5))
    plt.plot(frame.threshold, frame.model, lw=2, label="Model")
    plt.plot(frame.threshold, frame.treat_all, "--", label="Treat all")
    plt.plot(frame.threshold, frame.treat_none, ":", label="Treat none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(f"Decision curve - {name.replace('_', ' ').title()}")
    plt.legend()
    save_figure(f"{name}_decision_curve.png")

# %% [markdown]
# ## DR severity evaluation
#
# The DR checkpoint is an ordinal regression model with one scalar output.
# Therefore, honest ROC curves are generated for clinically meaningful binary
# endpoints (Any DR, Referable DR, and Vision-threatening DR). A five-class
# one-vs-rest ROC would require five calibrated class probabilities and must
# not be fabricated from the scalar output.

# %%
class DRDataset(Dataset):
    def __init__(self, frame, image_dir):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_id = str(row["id_code"])
        path = self.image_dir / (image_id if Path(image_id).suffix else image_id + ".png")
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(path)
        bgr = cv2.resize(bgr, (300, 300))
        mask = np.zeros((300, 300), np.uint8)
        cv2.circle(mask, (150, 150), int(150 * 0.95), 255, -1)
        bgr = cv2.bitwise_and(bgr, bgr, mask=mask)
        green = cv2.createCLAHE(2.0, (8, 8)).apply(bgr[:, :, 1])
        rgb = cv2.cvtColor(cv2.merge([green, green, green]), cv2.COLOR_BGR2RGB)
        return self.transform(rgb), int(row["diagnosis"]), str(path)


def dr_grade(scores):
    return np.digitize(np.asarray(scores), [0.5, 1.5, 2.5, 3.5]).astype(int)


dr_frame = pd.read_csv(DR_CSV)
required = {"id_code", "diagnosis"}
assert required.issubset(dr_frame.columns), f"DR CSV needs columns {required}"

if DR_EVALUATION_LABEL == "internal_validation":
    _, dr_eval = train_test_split(
        dr_frame, test_size=DR_TEST_SIZE, stratify=dr_frame.diagnosis,
        random_state=SEED
    )
else:
    dr_eval = dr_frame.copy()
dr_eval = dr_eval.reset_index(drop=True)

distribution = (
    dr_eval.diagnosis.value_counts().reindex(range(5), fill_value=0)
    .rename_axis("grade").reset_index(name="count")
)
distribution["class_name"] = CLASS_NAMES
distribution["proportion"] = distribution["count"] / distribution["count"].sum()
distribution.to_csv(TABLES / "dr_evaluation_class_distribution.csv", index=False)
plt.figure(figsize=(8, 4.5))
sns.barplot(data=distribution, x="class_name", y="count", color="#3b82f6")
plt.xlabel("DR grade")
plt.ylabel("Images")
plt.title(f"DR class distribution - {DR_EVALUATION_LABEL.replace('_', ' ')}")
plt.xticks(rotation=20)
save_figure("dr_evaluation_class_distribution.png")

dr_model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=1)
dr_model = load_state(dr_model, DR_MODEL).to(DEVICE).eval()
dr_loader = DataLoader(
    DRDataset(dr_eval, DR_IMAGE_DIR), batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
)

dr_true, dr_score, dr_path = [], [], []
dr_inference_started = time.perf_counter()
with torch.no_grad():
    for images, labels, paths in tqdm(dr_loader, desc="DR inference"):
        output = dr_model(images.to(DEVICE)).view(-1).cpu().numpy()
        dr_score.extend(output.tolist())
        dr_true.extend(labels.numpy().tolist())
        dr_path.extend(paths)
dr_inference_seconds = time.perf_counter() - dr_inference_started
checkpoint_metadata("DR severity", dr_model, DR_MODEL, dr_inference_seconds, len(dr_true))

dr_true = np.asarray(dr_true, int)
dr_score = np.asarray(dr_score, float)
dr_pred = dr_grade(dr_score)

dr_predictions = pd.DataFrame({
    "image_path": dr_path, "y_true": dr_true, "raw_ordinal_score": dr_score,
    "y_pred": dr_pred, "absolute_error": np.abs(dr_true - dr_pred),
    "evaluation_set": DR_EVALUATION_LABEL,
})
for column in ["patient_id", *SUBGROUP_COLUMNS]:
    if column in dr_eval.columns:
        dr_predictions[column] = dr_eval[column].to_numpy()
dr_predictions.to_csv(TABLES / "dr_image_level_predictions.csv", index=False)

dr_summary = {
    "n": len(dr_true),
    "accuracy": accuracy_score(dr_true, dr_pred),
    "balanced_accuracy": balanced_accuracy_score(dr_true, dr_pred),
    "macro_f1": f1_score(dr_true, dr_pred, average="macro"),
    "weighted_f1": f1_score(dr_true, dr_pred, average="weighted"),
    "qwk": cohen_kappa_score(dr_true, dr_pred, weights="quadratic"),
    "mae_grades": mean_absolute_error(dr_true, dr_pred),
    "within_one_grade_accuracy": float(np.mean(np.abs(dr_true - dr_pred) <= 1)),
    "mcc": matthews_corrcoef(dr_true, dr_pred),
}
for metric_name, function in {
    "accuracy": accuracy_score,
    "macro_f1": lambda a, b: f1_score(a, b, average="macro"),
    "qwk": lambda a, b: cohen_kappa_score(a, b, weights="quadratic"),
}.items():
    low, high = bootstrap_ci(dr_true, dr_pred, function)
    dr_summary[f"{metric_name}_ci95_low"] = low
    dr_summary[f"{metric_name}_ci95_high"] = high

(TABLES / "dr_metrics_summary.json").write_text(json.dumps(dr_summary, indent=2))
pd.DataFrame(classification_report(
    dr_true, dr_pred, labels=range(5), target_names=CLASS_NAMES,
    output_dict=True, zero_division=0
)).T.to_csv(TABLES / "dr_classification_report.csv")
dr_class_specific = specificity_per_class(dr_true, dr_pred, range(5))
dr_class_specific["class_name"] = CLASS_NAMES
dr_class_specific.to_csv(TABLES / "dr_sensitivity_specificity.csv", index=False)

cm = confusion_matrix(dr_true, dr_pred, labels=range(5))
pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(TABLES / "dr_confusion_matrix_counts.csv")
for normalize, suffix, fmt in [(None, "counts", "d"), ("true", "row_normalized", ".2f")]:
    cm_plot = confusion_matrix(dr_true, dr_pred, labels=range(5), normalize=normalize)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted grade")
    plt.ylabel("True grade")
    plt.title(f"DR severity confusion matrix - {suffix.replace('_', ' ')}")
    save_figure(f"dr_confusion_matrix_{suffix}.png")

report_plot = pd.DataFrame(classification_report(
    dr_true, dr_pred, labels=range(5), target_names=CLASS_NAMES,
    output_dict=True, zero_division=0
)).T.loc[CLASS_NAMES, ["precision", "recall", "f1-score"]]
report_plot.plot(kind="bar", figsize=(10, 5), ylim=(0, 1), rot=20)
plt.ylabel("Score")
plt.title("DR per-grade classification performance")
plt.legend(loc="lower right")
save_figure("dr_per_grade_metrics.png")

plt.figure(figsize=(7, 5))
errors = dr_pred - dr_true
sns.countplot(x=errors, color="#3b82f6")
plt.xlabel("Predicted grade - true grade")
plt.ylabel("Number of images")
plt.title("DR ordinal error distribution")
save_figure("dr_ordinal_error_distribution.png")

dr_predictions.sort_values(
    ["absolute_error", "raw_ordinal_score"], ascending=[False, False]
).head(50).to_csv(TABLES / "dr_largest_errors_for_review.csv", index=False)

subgroup_rows = []
for column in SUBGROUP_COLUMNS:
    if column not in dr_predictions.columns:
        continue
    for group, part in dr_predictions.groupby(column, dropna=False):
        if len(part) < 20:
            continue
        subgroup_rows.append({
            "variable": column, "group": str(group), "n": len(part),
            "accuracy": accuracy_score(part.y_true, part.y_pred),
            "macro_f1": f1_score(part.y_true, part.y_pred, average="macro", zero_division=0),
            "qwk": cohen_kappa_score(part.y_true, part.y_pred, weights="quadratic"),
            "mae_grades": mean_absolute_error(part.y_true, part.y_pred),
        })
if subgroup_rows:
    pd.DataFrame(subgroup_rows).to_csv(TABLES / "dr_subgroup_metrics.csv", index=False)

# %%
endpoint_rows = []
plt.figure(figsize=(7, 6))
for name, cutoff in [("Any DR", 1), ("Referable DR", 2), ("Vision-threatening DR", 3)]:
    binary_true = (dr_true >= cutoff).astype(int)
    fpr, tpr, thresholds = roc_curve(binary_true, dr_score)
    auc_value = roc_auc_score(binary_true, dr_score)
    low, high = bootstrap_ci(binary_true, dr_score, roc_auc_score)
    endpoint_rows.append({"endpoint": name, "cutoff": cutoff, "auc": auc_value,
                          "auc_ci95_low": low, "auc_ci95_high": high,
                          "prevalence": binary_true.mean()})
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
        TABLES / f"dr_{name.lower().replace(' ', '_').replace('-', '_')}_roc_points.csv",
        index=False
    )
    plt.plot(fpr, tpr, lw=2, label=f"{name}: AUC={auc_value:.3f}")
plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
plt.xlabel("False-positive rate")
plt.ylabel("True-positive rate")
plt.title("DR clinical-endpoint ROC curves")
plt.legend()
save_figure("dr_clinical_endpoint_roc.png")
pd.DataFrame(endpoint_rows).to_csv(TABLES / "dr_clinical_endpoint_auc.csv", index=False)

plt.figure(figsize=(7, 6))
for name, cutoff in [("Any DR", 1), ("Referable DR", 2), ("Vision-threatening DR", 3)]:
    binary_true = (dr_true >= cutoff).astype(int)
    precision, recall, thresholds = precision_recall_curve(binary_true, dr_score)
    ap = average_precision_score(binary_true, dr_score)
    pd.DataFrame({
        "precision": precision[:-1], "recall": recall[:-1], "threshold": thresholds
    }).to_csv(TABLES / f"dr_{name.lower().replace(' ', '_').replace('-', '_')}_pr_points.csv", index=False)
    plt.plot(recall, precision, lw=2, label=f"{name}: AP={ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("DR clinical-endpoint precision-recall curves")
plt.legend()
save_figure("dr_clinical_endpoint_precision_recall.png")

# Referable-DR calibration and decision curve. Convert the ordinal score to a
# monotonic probability for visualization; report this as an uncalibrated
# sigmoid mapping unless probability calibration is fitted on a separate set.
referable_true = (dr_true >= 2).astype(int)
referable_probability = 1 / (1 + np.exp(-(dr_score - 1.5)))
prob_true, prob_pred = calibration_curve(referable_true, referable_probability, n_bins=10)
pd.DataFrame({"mean_predicted_probability": prob_pred, "observed_fraction": prob_true}).to_csv(
    TABLES / "dr_referable_calibration_points.csv", index=False
)
plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, "o-", label="Model")
plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed referable-DR fraction")
plt.title(f"Referable-DR reliability diagram (Brier={brier_score_loss(referable_true, referable_probability):.3f})")
plt.legend()
save_figure("dr_referable_calibration.png")
decision_curve(referable_true, referable_probability, "dr_referable")

# %% [markdown]
# ## DR Grad-CAM and representative output gallery

# %%
class RegressionTarget:
    def __call__(self, model_output):
        return model_output[:, 0] if model_output.ndim == 2 else model_output


def find_last_conv(model):
    layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    if not layers:
        raise RuntimeError("No Conv2d layer found")
    return layers[-1]


if GradCAM is not None:
    target_layer = getattr(dr_model, "conv_head", find_last_conv(dr_model))
    cam = GradCAM(model=dr_model, target_layers=[target_layer])
    gallery_rows = []
    chosen = []
    for grade in range(5):
        correct = np.where((dr_true == grade) & (dr_pred == grade))[0]
        errors_grade = np.where((dr_true == grade) & (dr_pred != grade))[0]
        if len(correct):
            chosen.append((int(correct[0]), "correct"))
        if len(errors_grade):
            chosen.append((int(errors_grade[0]), "error"))

    fig, axes = plt.subplots(len(chosen), 3, figsize=(12, 4 * len(chosen)))
    if len(chosen) == 1:
        axes = np.asarray([axes])
    transform_tensor = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    for row, (index, case_type) in enumerate(chosen):
        original_bgr = cv2.imread(dr_path[index])
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(original_bgr, (300, 300))
        circle = np.zeros((300, 300), np.uint8)
        cv2.circle(circle, (150, 150), int(150 * 0.95), 255, -1)
        resized = cv2.bitwise_and(resized, resized, mask=circle)
        green = cv2.createCLAHE(2.0, (8, 8)).apply(resized[:, :, 1])
        processed = cv2.cvtColor(cv2.merge([green, green, green]), cv2.COLOR_BGR2RGB)
        tensor = transform_tensor(processed).unsqueeze(0).to(DEVICE)
        grayscale_cam = cam(input_tensor=tensor, targets=[RegressionTarget()])[0]
        overlay = show_cam_on_image(processed.astype(np.float32) / 255, grayscale_cam, use_rgb=True)
        axes[row, 0].imshow(original_rgb)
        axes[row, 0].set_title("Original")
        axes[row, 1].imshow(processed, cmap="gray")
        axes[row, 1].set_title("Model input")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(
            f"Grad-CAM | true={CLASS_NAMES[dr_true[index]]}\n"
            f"pred={CLASS_NAMES[dr_pred[index]]} ({case_type})"
        )
        for axis in axes[row]:
            axis.axis("off")
        gallery_rows.append({
            "image_path": dr_path[index], "case_type": case_type,
            "y_true": int(dr_true[index]), "y_pred": int(dr_pred[index]),
            "raw_score": float(dr_score[index])
        })
    pd.DataFrame(gallery_rows).to_csv(TABLES / "dr_gradcam_gallery_index.csv", index=False)
    save_figure("dr_gradcam_gallery.png")
else:
    warnings.warn("Grad-CAM unavailable; install grad-cam and rerun this cell.")

# %% [markdown]
# ## Glaucoma and cataract transfer-learning evaluation

# %%
class BinaryEyeDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = list(paths), list(labels)
        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        bgr = cv2.imread(self.paths[index])
        if bgr is None:
            raise FileNotFoundError(self.paths[index])
        bgr = cv2.resize(bgr, (300, 300))
        circle = np.zeros((300, 300), np.uint8)
        cv2.circle(circle, (150, 150), int(150 * 0.95), 255, -1)
        bgr = cv2.bitwise_and(bgr, bgr, mask=circle)
        green = cv2.createCLAHE(2.0, (8, 8)).apply(bgr[:, :, 1])
        rgb = cv2.cvtColor(cv2.merge([green, green, green]), cv2.COLOR_BGR2RGB)
        return self.transform(rgb), int(self.labels[index]), self.paths[index]


def evaluate_binary_disease(disease, checkpoint):
    normal = sorted(str(p) for p in (EYE_DISEASE_DIR / "normal").glob("*.*"))
    positive = sorted(str(p) for p in (EYE_DISEASE_DIR / disease.lower()).glob("*.*"))
    paths = normal + positive
    labels = [0] * len(normal) + [1] * len(positive)
    _, eval_paths, _, eval_labels = train_test_split(
        paths, labels, test_size=0.15, stratify=labels, random_state=SEED
    )
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=1)
    model = load_state(model, checkpoint).to(DEVICE).eval()
    loader = DataLoader(BinaryEyeDataset(eval_paths, eval_labels), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)
    truth, probability, image_paths = [], [], []
    inference_started = time.perf_counter()
    with torch.no_grad():
        for images, labels_batch, paths_batch in tqdm(loader, desc=disease):
            probability.extend(torch.sigmoid(model(images.to(DEVICE))).view(-1).cpu().numpy())
            truth.extend(labels_batch.numpy())
            image_paths.extend(paths_batch)
    inference_seconds = time.perf_counter() - inference_started
    checkpoint_metadata(disease, model, checkpoint, inference_seconds, len(truth))
    truth, probability = np.asarray(truth, int), np.asarray(probability, float)
    prediction = (probability >= 0.5).astype(int)
    prefix = disease.lower()
    pd.DataFrame({
        "image_path": image_paths, "y_true": truth, "probability": probability,
        "y_pred": prediction, "evaluation_set": "original_internal_validation"
    }).to_csv(TABLES / f"{prefix}_image_level_predictions.csv", index=False)
    summary = {
        "n": len(truth), "auc_roc": roc_auc_score(truth, probability),
        "average_precision": average_precision_score(truth, probability),
        "accuracy": accuracy_score(truth, prediction),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
        "f1": f1_score(truth, prediction),
        "sensitivity": recall_score(truth, prediction),
        "specificity": recall_score(1 - truth, 1 - prediction),
        "precision": precision_score(truth, prediction),
        "mcc": matthews_corrcoef(truth, prediction),
        "brier": brier_score_loss(truth, probability),
    }
    fpr_threshold, tpr_threshold, candidate_thresholds = roc_curve(truth, probability)
    best_index = int(np.argmax(tpr_threshold - fpr_threshold))
    summary["youden_threshold"] = float(candidate_thresholds[best_index])
    summary["youden_sensitivity"] = float(tpr_threshold[best_index])
    summary["youden_specificity"] = float(1 - fpr_threshold[best_index])
    for metric_name, function, values in [
        ("auc_roc", roc_auc_score, probability),
        ("accuracy", accuracy_score, prediction),
        ("f1", f1_score, prediction),
    ]:
        low, high = bootstrap_ci(truth, values, function)
        summary[f"{metric_name}_ci95_low"] = low
        summary[f"{metric_name}_ci95_high"] = high
    (TABLES / f"{prefix}_metrics_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(classification_report(
        truth, prediction, target_names=["Normal", disease], output_dict=True, zero_division=0
    )).T.to_csv(TABLES / f"{prefix}_classification_report.csv")

    cm_binary = confusion_matrix(truth, prediction)
    pd.DataFrame(cm_binary, index=["Normal", disease], columns=["Normal", disease]).to_csv(
        TABLES / f"{prefix}_confusion_matrix.csv"
    )
    plt.figure(figsize=(5.5, 5))
    sns.heatmap(cm_binary, annot=True, fmt="d", cmap="Purples",
                xticklabels=["Normal", disease], yticklabels=["Normal", disease])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{disease} confusion matrix")
    save_figure(f"{prefix}_confusion_matrix.png")

    fpr, tpr, roc_thresholds = roc_curve(truth, probability)
    precision, recall, pr_thresholds = precision_recall_curve(truth, probability)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(
        TABLES / f"{prefix}_roc_points.csv", index=False
    )
    pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1],
                  "threshold": pr_thresholds}).to_csv(
        TABLES / f"{prefix}_pr_points.csv", index=False
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC={summary['auc_roc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"{disease} ROC")
    axes[0].legend()
    axes[1].plot(recall, precision, lw=2, label=f"AP={summary['average_precision']:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title=f"{disease} precision-recall")
    axes[1].legend()
    save_figure(f"{prefix}_roc_and_pr.png")

    observed, predicted = calibration_curve(truth, probability, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(predicted, observed, "o-", label=disease)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title(f"{disease} reliability diagram")
    plt.legend()
    save_figure(f"{prefix}_calibration.png")
    decision_curve(truth, probability, prefix)
    return summary


glaucoma_summary = evaluate_binary_disease("Glaucoma", GLAUCOMA_MODEL)
cataract_summary = evaluate_binary_disease("Cataract", CATARACT_MODEL)

# %% [markdown]
# ## Lesion-segmentation evaluation
#
# Manifests make image/mask pairing explicit and auditable. Never rely on
# unsorted directory order. The outputs include per-image metrics, pooled
# pixel ROC/PR curves, threshold sweeps, bootstrap intervals, and galleries
# showing image, ground truth, prediction, and overlay.

# %%
SEGMENTATION_SPECS = {
    "OD": ("UnetPlusPlus", "resnet34", "best_od_model_fp16.pth", 0.50),
    "EX": ("UnetPlusPlus", "efficientnet-b2", "best_ex_model_fp16.pth", 0.40),
    "SE": ("UnetPlusPlus", "efficientnet-b2", "best_se_model_fp16.pth", 0.40),
    "MA": ("UnetPlusPlus", "efficientnet-b4", "best_ma_model_fp16.pth", 0.30),
    "HE": ("UnetPlusPlus", "efficientnet-b4", "best_he_model_fp16.pth", 0.30),
}


def resolve_kaggle_path(value):
    path = Path(str(value))
    return path if path.is_absolute() else Path("/kaggle/input") / path


class SegmentationDataset(Dataset):
    def __init__(self, manifest):
        self.frame = pd.read_csv(manifest)
        assert {"image_path", "mask_path"}.issubset(self.frame.columns)
        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        image_path = resolve_kaggle_path(self.frame.iloc[index].image_path)
        mask_path = resolve_kaggle_path(self.frame.iloc[index].mask_path)
        bgr = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            raise FileNotFoundError(f"{image_path} | {mask_path}")
        rgb = cv2.cvtColor(cv2.resize(bgr, (512, 512)), cv2.COLOR_BGR2RGB)
        mask = (cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
        return self.transform(rgb), torch.from_numpy(mask[None]), str(image_path), str(mask_path)


def segmentation_metrics(truth, pred):
    truth, pred = truth.astype(bool), pred.astype(bool)
    tp = np.sum(truth & pred)
    tn = np.sum(~truth & ~pred)
    fp = np.sum(~truth & pred)
    fn = np.sum(truth & ~pred)
    eps = 1e-8
    return {
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
        "specificity": (tn + eps) / (tn + fp + eps),
        "pixel_accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }


def lesion_detection_metrics(truth, pred):
    """Connected-component lesion sensitivity and false positives per image."""
    truth = truth.astype(np.uint8)
    pred = pred.astype(np.uint8)
    n_truth, truth_labels = cv2.connectedComponents(truth)
    n_pred, pred_labels = cv2.connectedComponents(pred)
    detected = 0
    for component in range(1, n_truth):
        if np.any(pred[truth_labels == component] > 0):
            detected += 1
    false_positive_components = 0
    for component in range(1, n_pred):
        if not np.any(truth[pred_labels == component] > 0):
            false_positive_components += 1
    lesion_count = n_truth - 1
    return {
        "gt_lesions": lesion_count,
        "detected_lesions": detected,
        "lesion_sensitivity": detected / lesion_count if lesion_count else np.nan,
        "false_positive_lesions": false_positive_components,
    }


def evaluate_segmentation(biomarker):
    arch, encoder, weight_name, threshold = SEGMENTATION_SPECS[biomarker]
    dataset = SegmentationDataset(SEGMENTATION_MANIFESTS[biomarker])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    model = getattr(smp, arch)(
        encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1,
        decoder_attention_type="scse"
    )
    model = load_state(model, SEGMENTATION_MODEL_DIR / weight_name).to(DEVICE).eval()
    records, example_data = [], []
    sampled_truth, sampled_probability = [], []
    rng = np.random.default_rng(SEED)

    with torch.no_grad():
        for images, masks, image_paths, mask_paths in tqdm(loader, desc=biomarker):
            probability = torch.sigmoid(model(images.to(DEVICE))).cpu().numpy()[0, 0]
            truth = masks.numpy()[0, 0].astype(np.uint8)
            prediction = (probability >= threshold).astype(np.uint8)
            row = segmentation_metrics(truth, prediction)
            row.update(lesion_detection_metrics(truth, prediction))
            row.update({"image_path": image_paths[0], "mask_path": mask_paths[0],
                        "threshold": threshold, "biomarker": biomarker})
            records.append(row)
            flat_count = truth.size
            take = min(5000, flat_count)
            index = rng.choice(flat_count, size=take, replace=False)
            sampled_truth.append(truth.ravel()[index])
            sampled_probability.append(probability.ravel()[index])
            if len(example_data) < 8:
                example_data.append((image_paths[0], truth, probability, prediction))

    frame = pd.DataFrame(records)
    frame.to_csv(TABLES / f"seg_{biomarker.lower()}_per_image_metrics.csv", index=False)
    summary_rows = []
    for metric in ["dice", "iou", "precision", "recall", "specificity", "pixel_accuracy"]:
        values = frame[metric].to_numpy()
        rng_ci = np.random.default_rng(SEED)
        means = [np.mean(rng_ci.choice(values, len(values), replace=True)) for _ in range(N_BOOTSTRAP)]
        summary_rows.append({
            "biomarker": biomarker, "metric": metric, "mean": values.mean(),
            "sd": values.std(ddof=1), "median": np.median(values),
            "ci95_low": np.percentile(means, 2.5), "ci95_high": np.percentile(means, 97.5),
            "n_images": len(values),
        })
    summary = pd.DataFrame(summary_rows)
    lesion_totals = {
        "biomarker": biomarker,
        "metric": "lesion_detection",
        "mean": frame.detected_lesions.sum() / frame.gt_lesions.sum()
        if frame.gt_lesions.sum() else np.nan,
        "sd": frame.lesion_sensitivity.std(ddof=1),
        "median": frame.lesion_sensitivity.median(),
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "n_images": len(frame),
        "false_positives_per_image": frame.false_positive_lesions.mean(),
        "total_gt_lesions": int(frame.gt_lesions.sum()),
    }
    summary = pd.concat([summary, pd.DataFrame([lesion_totals])], ignore_index=True)
    summary.to_csv(TABLES / f"seg_{biomarker.lower()}_summary.csv", index=False)

    pixel_truth = np.concatenate(sampled_truth)
    pixel_probability = np.concatenate(sampled_probability)
    fpr, tpr, roc_thresholds = roc_curve(pixel_truth, pixel_probability)
    precision, recall, pr_thresholds = precision_recall_curve(pixel_truth, pixel_probability)
    pixel_auc = roc_auc_score(pixel_truth, pixel_probability)
    pixel_ap = average_precision_score(pixel_truth, pixel_probability)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(
        TABLES / f"seg_{biomarker.lower()}_pixel_roc_points.csv", index=False
    )
    pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1],
                  "threshold": pr_thresholds}).to_csv(
        TABLES / f"seg_{biomarker.lower()}_pixel_pr_points.csv", index=False
    )

    thresholds = np.arange(0.05, 0.96, 0.05)
    sweep = []
    for value in thresholds:
        metrics = segmentation_metrics(pixel_truth, pixel_probability >= value)
        metrics["threshold"] = value
        sweep.append(metrics)
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(TABLES / f"seg_{biomarker.lower()}_threshold_sweep.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].plot(fpr, tpr, label=f"AUC={pixel_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"{biomarker} pixel ROC")
    axes[0].legend()
    axes[1].plot(recall, precision, label=f"AP={pixel_ap:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title=f"{biomarker} pixel precision-recall")
    axes[1].legend()
    axes[2].plot(sweep.threshold, sweep.dice, label="Dice")
    axes[2].plot(sweep.threshold, sweep.iou, label="IoU")
    axes[2].plot(sweep.threshold, sweep.recall, label="Recall")
    axes[2].plot(sweep.threshold, sweep.precision, label="Precision")
    axes[2].axvline(threshold, color="black", ls="--", label=f"Used={threshold}")
    axes[2].set(xlabel="Threshold", ylabel="Metric", title=f"{biomarker} threshold sensitivity")
    axes[2].legend()
    save_figure(f"seg_{biomarker.lower()}_roc_pr_threshold.png")

    fig, axes = plt.subplots(len(example_data), 4, figsize=(14, 3.5 * len(example_data)))
    if len(example_data) == 1:
        axes = np.asarray([axes])
    for row, (image_path, truth, probability, prediction) in enumerate(example_data):
        rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (512, 512))
        overlay = rgb.copy()
        overlay[truth.astype(bool)] = 0.55 * overlay[truth.astype(bool)] + 0.45 * np.array([0, 255, 0])
        overlay[prediction.astype(bool)] = 0.55 * overlay[prediction.astype(bool)] + 0.45 * np.array([255, 0, 0])
        for axis, image, title in zip(
            axes[row], [rgb, truth, probability, overlay.astype(np.uint8)],
            ["Fundus", "Ground truth", "Probability map", "Overlay: GT green, prediction red"]
        ):
            axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
            axis.set_title(title)
            axis.axis("off")
    save_figure(f"seg_{biomarker.lower()}_qualitative_gallery.png")
    checkpoint_metadata(f"Segmentation {biomarker}", model, SEGMENTATION_MODEL_DIR / weight_name)
    return summary


segmentation_summaries = []
for biomarker, manifest in SEGMENTATION_MANIFESTS.items():
    if manifest.exists():
        segmentation_summaries.append(evaluate_segmentation(biomarker))
    else:
        print(f"SKIPPED {biomarker}: add manifest at {manifest}")

if segmentation_summaries:
    pd.concat(segmentation_summaries, ignore_index=True).to_csv(
        TABLES / "segmentation_all_biomarkers_summary.csv", index=False
    )

# %% [markdown]
# ## Optional genuine training-history, DME, and reproducibility evidence

# %%
if TRAINING_HISTORY_CSV is not None and Path(TRAINING_HISTORY_CSV).exists():
    history = pd.read_csv(TRAINING_HISTORY_CSV)
    required_history = {"model", "epoch"}
    assert required_history.issubset(history.columns), f"History requires {required_history}"
    history.to_csv(TABLES / "genuine_training_history.csv", index=False)
    metric_columns = [
        column for column in
        ["train_loss", "val_loss", "qwk", "auc", "accuracy", "macro_f1", "dice", "iou"]
        if column in history.columns
    ]
    for model_name, part in history.groupby("model"):
        if not metric_columns:
            continue
        columns = 2
        rows = math.ceil(len(metric_columns) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows))
        axes = np.asarray(axes).reshape(-1)
        for axis, metric in zip(axes, metric_columns):
            for phase, phase_part in part.groupby("phase") if "phase" in part.columns else [("all", part)]:
                axis.plot(phase_part.epoch, phase_part[metric], label=str(phase))
            axis.set(xlabel="Epoch", ylabel=metric, title=f"{model_name}: {metric}")
            axis.legend()
        for axis in axes[len(metric_columns):]:
            axis.axis("off")
        save_figure(f"history_{str(model_name).lower().replace(' ', '_')}.png")
else:
    print(
        "TRAINING HISTORY NOT AVAILABLE: genuine loss/metric curves require the "
        "epoch records saved during training or a complete retraining run."
    )


def evaluate_existing_binary_predictions(path, prefix):
    frame = pd.read_csv(path)
    assert {"y_true", "probability"}.issubset(frame.columns)
    truth = frame.y_true.to_numpy(int)
    probability = frame.probability.to_numpy(float)
    prediction = (probability >= 0.5).astype(int)
    summary = {
        "n": len(frame),
        "auc_roc": roc_auc_score(truth, probability),
        "average_precision": average_precision_score(truth, probability),
        "sensitivity": recall_score(truth, prediction),
        "specificity": recall_score(1 - truth, 1 - prediction),
        "precision": precision_score(truth, prediction, zero_division=0),
        "f1": f1_score(truth, prediction, zero_division=0),
    }
    (TABLES / f"{prefix}_metrics_summary.json").write_text(json.dumps(summary, indent=2))
    frame.assign(y_pred=prediction).to_csv(TABLES / f"{prefix}_predictions.csv", index=False)
    fpr, tpr, _ = roc_curve(truth, probability)
    precision, recall, _ = precision_recall_curve(truth, probability)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, label=f"AUC={summary['auc_roc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"{prefix.upper()} ROC")
    axes[0].legend()
    axes[1].plot(recall, precision, label=f"AP={summary['average_precision']:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title=f"{prefix.upper()} precision-recall")
    axes[1].legend()
    save_figure(f"{prefix}_roc_and_pr.png")
    decision_curve(truth, probability, prefix)


if DME_PREDICTIONS_CSV is not None and Path(DME_PREDICTIONS_CSV).exists():
    evaluate_existing_binary_predictions(DME_PREDICTIONS_CSV, "dme")
else:
    print("DME EVIDENCE NOT AVAILABLE: attach y_true and probability columns to evaluate it.")

pd.DataFrame(model_metadata_rows).to_csv(TABLES / "model_checkpoint_and_runtime_metadata.csv", index=False)

# %% [markdown]
# ## Export one Excel workbook and ZIP attachment package

# %%
with pd.ExcelWriter(OUTPUT / "OphthalmicAI_publication_results.xlsx", engine="openpyxl") as writer:
    for csv_path in sorted(TABLES.glob("*.csv")):
        sheet = csv_path.stem[:31]
        pd.read_csv(csv_path).to_excel(writer, sheet_name=sheet, index=False)

manifest_rows = []
for path in sorted(OUTPUT.rglob("*")):
    if path.is_file():
        manifest_rows.append({
            "relative_path": str(path.relative_to(OUTPUT)),
            "size_bytes": path.stat().st_size,
        })
pd.DataFrame(manifest_rows).to_csv(OUTPUT / "attachment_manifest.csv", index=False)

draft_lines = [
    "# Results section - generated evidence index",
    "",
    "This index is generated from the current evaluation run. Replace no values manually.",
    "",
    "## Primary DR severity experiment",
]
if (TABLES / "dr_metrics_summary.json").exists():
    values = json.loads((TABLES / "dr_metrics_summary.json").read_text())
    draft_lines += [
        f"- Evaluation images: {values['n']}",
        f"- Accuracy: {values['accuracy']:.4f} "
        f"(95% CI {values['accuracy_ci95_low']:.4f}-{values['accuracy_ci95_high']:.4f})",
        f"- QWK: {values['qwk']:.4f} "
        f"(95% CI {values['qwk_ci95_low']:.4f}-{values['qwk_ci95_high']:.4f})",
        f"- Macro F1: {values['macro_f1']:.4f} "
        f"(95% CI {values['macro_f1_ci95_low']:.4f}-{values['macro_f1_ci95_high']:.4f})",
        f"- MAE: {values['mae_grades']:.4f} grades",
        f"- Within-one-grade accuracy: {values['within_one_grade_accuracy']:.4f}",
    ]
draft_lines += [
    "",
    "## Required interpretation",
    "- Existing model-selection splits must be called internal validation.",
    "- Glaucoma and cataract are secondary DR-pretrained transfer-learning experiments.",
    "- ROC/PR for the scalar DR regressor is reported by clinical grade threshold.",
    "- See the tables and figures folders for all source evidence.",
]
(OUTPUT / "RESULTS_SECTION_GENERATED_INDEX.md").write_text("\n".join(draft_lines))

import shutil
archive = shutil.make_archive("/kaggle/working/OphthalmicAI_publication_results", "zip", OUTPUT)
print("DONE")
print("Download:", archive)
print("Also download:", OUTPUT / "OphthalmicAI_publication_results.xlsx")
