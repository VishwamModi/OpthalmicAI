# OphthalmicAI publication results attachment checklist

## Scope statement

- Deployed application: five-grade diabetic retinopathy severity prediction, five DR-related segmentation outputs, Grad-CAM, clinical measurements, and report generation.
- Primary journal experiment: DR severity grading.
- Secondary journal experiments: glaucoma and cataract classification using a DR-pretrained EfficientNet-B3 backbone.
- Glaucoma and cataract are research transfer-learning experiments and are not deployed application capabilities.

## Evidence already available

### DR severity

- APTOS internal-validation confusion matrix with 550 images.
- Per-grade precision, recall, F1-score, and support.
- Overall accuracy approximately 0.81 and QWK 0.8912 from the executed transfer-learning notebook.
- Raw epoch-by-epoch EyePACS pretraining and APTOS fine-tuning console outputs.
- Saved `OphthalmicAI_Final_EffNetB3.pt` checkpoint.

### Transfer-learning experiments

- Glaucoma internal-validation classification report, sample size 313, and reported ROC AUC 0.9565.
- Cataract internal-validation classification report, sample size 317, confusion matrix, and reported ROC AUC 0.9959.
- Saved glaucoma and cataract checkpoints.

### Segmentation

- Saved checkpoints for optic disc, hard exudates, soft exudates, microaneurysms, and hemorrhages.
- Training-curve screenshots and sample ground-truth/predicted-mask panels.
- Metric screenshots for all five segmenters.

## Required attachments to export from Kaggle

Run `OphthalmicAI_Kaggle_Publication_Evaluation.ipynb` and download the generated ZIP and Excel workbook.

### Mandatory source tables

- Image-level DR file containing image ID, true grade, raw ordinal score, predicted grade, and absolute grade error.
- DR 5x5 confusion matrix as counts and row-normalized percentages.
- DR per-grade sensitivity, specificity, precision, NPV, recall, F1-score, and support.
- DR overall accuracy, balanced accuracy, macro/weighted F1, QWK, MAE, within-one-grade accuracy, and MCC.
- Bootstrap 95% confidence intervals for accuracy, macro F1, and QWK.
- Image-level glaucoma and cataract files containing ground truth, probability, and predicted class.
- Glaucoma and cataract confusion matrices, classification reports, ROC AUC, average precision, calibration, and 95% confidence intervals.
- Image-level predictions and four-class probabilities from EfficientNet-B0,
  MobileNetV3-Large, and EfficientNet-B3 for the identical evaluation image IDs.
- Competing-model accuracy, balanced accuracy, macro/weighted F1, MCC,
  multiclass one-vs-rest AUC, log loss, parameter count, and inference time.
- Cochran's Q omnibus comparison, pairwise exact McNemar tests with Holm
  correction, and paired bootstrap 95% confidence intervals for differences
  in accuracy, macro F1, and macro AUC.
- Per-image segmentation metrics for every biomarker.
- Segmentation mean, SD, median, and bootstrap 95% confidence intervals.
- Exact threshold-sweep results for each segmentation model.

### Mandatory figures

- DR class distribution for every train/validation/test split.
- DR confusion matrix: raw counts.
- DR confusion matrix: normalized by true class.
- DR per-grade precision/recall/F1 bar chart.
- DR ordinal error-distribution chart.
- DR ROC curves for Any DR, Referable DR, and Vision-threatening DR.
- DR precision-recall curves for the same clinical endpoints.
- Referable-DR reliability/calibration diagram.
- Referable-DR decision-curve analysis generated from real image-level outputs.
- Glaucoma ROC and precision-recall curves.
- Cataract ROC and precision-recall curves.
- Glaucoma and cataract reliability diagrams.
- Competing-model performance chart, side-by-side confusion matrices, and
  per-class ROC and precision-recall comparisons.
- Segmentation pixel ROC and precision-recall curves for OD, EX, SE, MA, and HE.
- Segmentation threshold-versus-Dice/IoU/precision/recall curves.
- Segmentation training and validation loss/Dice/IoU curves exported directly from saved history, not manually redrawn.

### Mandatory qualitative panels

- At least two correctly classified and two misclassified DR examples per grade where available.
- Original fundus, actual preprocessing output, Grad-CAM heatmap, true grade, predicted grade, and raw score.
- For every segmentation biomarker: original fundus, ground-truth mask, probability map, thresholded prediction, and color overlay.
- At least five representative segmentation cases plus two difficult/failure cases per biomarker.
- Clear legends specifying overlay colors and thresholds.
- De-identified image identifiers only.

### Reproducibility attachment

- Dataset versions and download links.
- Patient/image split method and random seed.
- Train/validation/test counts by class.
- Model architecture, pretrained checkpoint, input size, optimizer, learning rates, batch size, epoch counts, augmentations, and stopping rule.
- Model-selection metric and selected epoch.
- Software versions, GPU type, and runtime.
- SHA-256 hashes of final checkpoints.
- Total and per-image inference time, peak RAM, peak GPU memory, parameter count, and checkpoint size.

## Numerical conflicts that must be resolved

- Thesis section 5.1 reports DR QWK 0.912; section 5.3 reports 0.89; the executed notebook reports best validation QWK 0.8912.
- Thesis accuracy is 81.4%; the notebook classification report rounds to 0.81. Recompute from saved image-level predictions for the exact value.
- Thesis narrative reports optic-disc Dice 0.964 and hard-exudate Dice 0.863, but thesis Table 5.1 reports 0.89 and 0.43.
- Saved segmentation screenshots contain a third set of results. Some displayed Dice, IoU, and F1 values are not algebraically consistent, implying different averaging rules or a calculation error.
- Thesis claims Grad-CAM agreement on 104 images and DME/DCA performance, but the supporting image-level annotation/prediction files are not present.
- The existing application can generate synthetic confidence values and synthetic DCA fallback data. These must never be used in a journal results section.

## Interpretation rules

- Use "internal validation" for the existing model-selection splits.
- Statistical comparisons are valid only when every competing model is
  evaluated on the exact same image IDs. State whether those images were
  excluded from training and checkpoint selection for all models.
- Do not call them independent test results.
- Do not report a five-class one-vs-rest ROC curve from the scalar DR regression checkpoint.
- For the scalar DR model, report ROC/PR for clinically meaningful ordinal thresholds: Any DR (grade >=1), Referable DR (grade >=2), and Vision-threatening DR (grade >=3).
- If a true independent test set is produced, freeze thresholds and checkpoints before evaluating it.
- Keep glaucoma and cataract in a separate transfer-learning subsection and explicitly state that they are not integrated into the deployed application.
