# OphthalmicAI research evaluation package

`kaggle_publication_evaluation.py` is a cell-delimited Kaggle workflow that exports:

- image-level DR predictions and ordinal error analysis;
- DR confusion matrices, per-grade metrics, QWK, MAE, MCC, bootstrap confidence intervals;
- clinically valid ROC and precision-recall curves for Any DR, Referable DR, and Vision-threatening DR;
- referable-DR calibration and decision-curve analysis;
- Grad-CAM galleries containing correct and error cases from each available DR grade;
- glaucoma and cataract ROC, precision-recall, calibration, confusion matrices, DCA, and image-level predictions;
- per-image lesion-segmentation Dice, IoU, precision, recall, specificity, pixel accuracy, bootstrap intervals;
- connected-component lesion sensitivity and false-positive lesions per image;
- segmentation pixel ROC/PR curves, threshold sweeps, and fundus/ground-truth/prediction/overlay galleries;
- checkpoint SHA-256 hashes, sizes, parameter counts, throughput, and per-image inference timing;
- optional subgroup analysis when sex, age group, camera, or site metadata is available;
- optional genuine training-history plots and DME evaluation when their source CSVs are supplied;
- a generated, metric-backed results-section evidence index;
- a consolidated Excel workbook and downloadable ZIP attachment package.

Before running:

1. Add the APTOS, Eye Diseases Classification, segmentation-test datasets, and model checkpoints as Kaggle inputs.
2. Edit the path block near the top.
3. Provide one segmentation manifest CSV per biomarker with `image_path` and `mask_path`.
4. Use an untouched test set where possible. The original 85/15 and 15% binary splits were used for model selection and should be described as internal validation.
5. Download both `OphthalmicAI_publication_results.zip` and `OphthalmicAI_publication_results.xlsx`.

Do not claim a five-class DR ROC from the scalar regression checkpoint. The valid ROC analyses for this model are threshold-based clinical endpoints.
