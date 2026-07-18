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

The notebook is configured for these Kaggle inputs:

- `Diabetic Retinopathy Lesion Segmentation/lesion_test/image` and `mask`;
- `eye_diseases_classification/dataset/{normal,cataract,glaucoma,diabetic_retinopathy}`;
- `models/OphthalmicAI_Final_EffNetB3.pt` and the five `best_*_model_fp16.pth` files;
- `APTOS-2019 dataset/{test_images,train_images,val_images}` with `test.csv`, `train_1.csv`, and `valid.csv`;
- the optional `Diabetic Retinopathy (resized)` dataset for dataset-description evidence.

Before running:

1. Add those datasets and checkpoints as Kaggle inputs.
2. Also add `Glaucoma_EffNetB3.pt` and `Cataract_EffNetB3.pt` to the `models` input if their paper results are required. The notebook skips them honestly when absent.
3. Select a T4 GPU and enable Internet for the initial package-install cell.
4. Run all cells. Paths, repeated Kaggle directories, APTOS CSV columns, and lesion image-mask manifests are discovered automatically.
5. Review `tables/segmentation_pairing_audit.csv`. Only unambiguous image-mask pairs are evaluated.
6. Download both `OphthalmicAI_publication_results.zip` and `OphthalmicAI_publication_results.xlsx`.

When `test.csv` contains labels, the notebook evaluates the full provided test set.
Otherwise it tries `valid.csv`, then reproduces the 15% stratified split from
`train_1.csv`. The original binary-disease 15% splits were used for model
selection and must be described as internal validation.

Do not claim a five-class DR ROC from the scalar regression checkpoint. The valid ROC analyses for this model are threshold-based clinical endpoints.
