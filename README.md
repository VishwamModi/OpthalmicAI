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
NEXT_PUBLIC_APP_OVERVIEW_YT=https://www.youtube.com/embed/VIDEO_ID
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
- **Root Directory:** repo root (`.`)
- **Build Command:** `pip install -r backend/requirements.txt`
- **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

### 2) Frontend web service
- **Root Directory:** `frontend`
- **Build Command:** `npm install && npm run build`
- **Start Command:** `npm run start -- -p $PORT`
- **Environment Variables:**
	- `NEXT_PUBLIC_API_BASE_URL=https://<your-backend-service>.onrender.com`
	- `NEXT_PUBLIC_APP_OVERVIEW_YT=https://www.youtube.com/embed/VIDEO_ID`

The included [render.yaml](render.yaml) can be used as a starting point.

## Troubleshooting

- If the video does not appear, confirm `NEXT_PUBLIC_APP_OVERVIEW_YT` points to a valid YouTube embed URL.
- If the frontend cannot reach the backend, verify `NEXT_PUBLIC_API_BASE_URL` points to the deployed backend URL.
- If models fail to load on Render, make sure [Models/](Models) was pushed correctly through Git LFS and is present at runtime.
- If browser cache looks stale after deploy, do a hard refresh.

## Quick checklist before pushing

- [ ] `NEXT_PUBLIC_APP_OVERVIEW_YT` is set to your YouTube embed URL
- [ ] `Models/` is present in the repo
- [ ] `README.md` is updated
- [ ] `render.yaml` matches your deployment setup
- [ ] the Git remote points to `VishwamModi/OpthalmicAI`
