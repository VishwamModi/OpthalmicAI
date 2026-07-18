# OpthalmicAI

OpthalmicAI is a full-stack retinal screening app with:

- a **Next.js frontend** for landing, login, uploads, dashboard, and reports
- a **FastAPI backend** for authentication, inference, Grad-CAM, and segmentation
- **PyTorch model weights** in [Models/](Models)
- a **YouTube-hosted walkthrough video** and UI images in [frontend/public/images/](frontend/public/images)

## What is included

### Frontend
- [frontend/](frontend)
- Public UI assets in [frontend/public/images/](frontend/public/images)
- Walkthrough video is embedded from YouTube (see `NEXT_PUBLIC_APP_OVERVIEW_YT` below)

### Backend
- [backend/](backend)
- FastAPI routes, database logic, model loading, Grad-CAM, and report generation

### Models
- [Models/](Models)
- DR classifier weights and segmentation weights used at inference time
- Git LFS tracking is enabled for large `.pt` and `.pth` files

## Important GitHub note

Model weights are large, so Git LFS is required for `.pt` and `.pth` files.

If Git LFS is not installed yet:

```bash
git lfs install
```

Tracked file types are configured in [`.gitattributes`](.gitattributes).

## Models included

This repo includes the required model files:

- `Models/OphthalmicAI_Final_EffNetB3.pt`
- `Models/Cataract_EffNetB3.pt`
- `Models/Glaucoma_EffNetB3.pt`
- `Models/Robust_Green_EffNetB3.pt`
- `Models/EfficientNetB3.pt`
- `Models/EfficientNetB0.pth`
- `Models/MobileNetV3_1.pth`
- `Models/segmentation/best_od_model_fp16.pth`
- `Models/segmentation/best_ex_model_fp16.pth`
- `Models/segmentation/best_se_model_fp16.pth`
- `Models/segmentation/best_ma_model_fp16.pth`
- `Models/segmentation/best_he_model_fp16.pth`

## Local setup

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

If the frontend needs a different backend URL or a YouTube walkthrough embed URL, set:

```bash
# frontend/.env.local
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_APP_OVERVIEW_YT=https://www.youtube.com/embed/opcj0hZPnxU
```

## Deployment and publication resources

- [Deployment from basics](DEPLOYMENT_FROM_BASICS.md)
- [Kaggle publication evaluation notebook](research/OphthalmicAI_Kaggle_Publication_Evaluation.ipynb)
- [Publication results attachment checklist](research/PUBLICATION_RESULTS_ATTACHMENT_CHECKLIST.md)
- [Comprehensive publication results package](research/OphthalmicAI_Comprehensive_Publication_Results_Package.docx)

The deployed application is intentionally limited to DR severity grading,
DR-related segmentation, and Grad-CAM. Glaucoma and cataract checkpoints are
reported only as DR-pretrained transfer-learning experiments for research.

## GitHub upload steps

If the remote is [https://github.com/VishwamModi/OpthalmicAI](https://github.com/VishwamModi/OpthalmicAI), push the workspace like this:

```bash
git lfs install
git add .
git commit -m "Update app assets, models, and docs"
git push origin main
```

## Free Hugging Face deployment

The free deployment is a DR-only Gradio Space and does not require Docker.

- Ready-to-upload Space package: [huggingface_space/](huggingface_space)
- Basic instructions: [DEPLOYMENT_FROM_BASICS.md](DEPLOYMENT_FROM_BASICS.md)
- Required checkpoint: `Models/OphthalmicAI_Final_EffNetB3.pt`

Static hosting cannot execute the PyTorch model. Choose the Gradio SDK and
free CPU Basic hardware for the interactive inference demonstration.

## Troubleshooting

- If the video does not appear, confirm `NEXT_PUBLIC_APP_OVERVIEW_YT` points to a valid YouTube embed URL.
- If the Space says the checkpoint is missing, upload
  `OphthalmicAI_Final_EffNetB3.pt` to the Space repository root.
- If the build fails, verify the Space SDK is Gradio rather than Static.
- If browser cache looks stale after deploy, do a hard refresh.

## Quick checklist before pushing

- [ ] `NEXT_PUBLIC_APP_OVERVIEW_YT` is set to your YouTube embed URL
- [ ] `Models/` is present in the repo
- [ ] `README.md` is updated
- [ ] the four files in `huggingface_space/` are ready for the Space
- [ ] the Git remote points to `VishwamModi/OpthalmicAI`
