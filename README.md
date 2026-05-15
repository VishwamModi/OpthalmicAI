# OpthalmicAI

OpthalmicAI is a full-stack retinal screening app with:

- a **Next.js frontend** for landing, login, uploads, dashboard, and reports
- a **FastAPI backend** for authentication, inference, Grad-CAM, and segmentation
- **PyTorch model weights** in [Models/](Models)
- a bundled **app overview video** and UI images in [frontend/public/images/](frontend/public/images)

## What is included

### Frontend
- [frontend/](frontend)
- Public UI assets in [frontend/public/images/](frontend/public/images)
- Overview demo video: [frontend/public/images/app_overview.mov](frontend/public/images/app_overview.mov)

### Backend
- [backend/](backend)
- FastAPI routes, database logic, model loading, Grad-CAM, and report generation

### Models
- [Models/](Models)
- DR classifier weights and segmentation weights used at inference time
- Git LFS tracking is enabled for large `.mov`, `.pt`, and `.pth` files

## Important GitHub note

The overview video is about 406 MB, so it cannot be pushed to GitHub as a normal file. This repository is prepared for **Git LFS**, which is required for the video and recommended for the model weights.

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

If the frontend needs a different backend URL, set:

```bash
# frontend/.env.local
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## GitHub upload steps

If the remote is [https://github.com/VishwamModi/OpthalmicAI](https://github.com/VishwamModi/OpthalmicAI), push the workspace like this:

```bash
git lfs install
git add .
git commit -m "Update app assets, models, and docs"
git push origin main
```

If your branch is not `main`, replace it with your branch name.

## Deploy on Render

Recommended as **two Render services**:

### 1) Backend web service
- **Root Directory:** `backend`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

### 2) Frontend web service
- **Root Directory:** `frontend`
- **Build Command:** `npm install && npm run build`
- **Start Command:** `npm run start -- -p $PORT`
- **Environment Variable:** `NEXT_PUBLIC_API_BASE_URL=https://<your-backend-service>.onrender.com`

The included [render.yaml](render.yaml) can be used as a starting point.

## Troubleshooting

- If the video does not appear on GitHub, confirm Git LFS is installed and the file was committed after LFS tracking was enabled.
- If the frontend cannot reach the backend, verify `NEXT_PUBLIC_API_BASE_URL` points to the deployed backend URL.
- If models fail to load on Render, make sure [Models/](Models) was pushed correctly through Git LFS and is present at runtime.
- If browser cache looks stale after deploy, do a hard refresh.

## Quick checklist before pushing

- [ ] `frontend/public/images/app_overview.mov` is tracked by Git LFS
- [ ] `Models/` is present in the repo
- [ ] `README.md` is updated
- [ ] `render.yaml` matches your deployment setup
- [ ] the Git remote points to `VishwamModi/OpthalmicAI`
