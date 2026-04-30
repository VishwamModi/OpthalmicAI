# OpthalmicAI (Eye-Disease-App)

End-to-end retinal screening web app with:
- FastAPI backend (`backend/`)
- Next.js frontend (`frontend/`)
- Integrated DR staging + lesion segmentation models (`Models/`)

## Repository Structure

- `backend/` API, model inference engine, DB logic
- `frontend/` web UI (landing, login, dashboard, notes, reports)
- `Models/` all required `.pt` / `.pth` model files, including `Models/segmentation`

## Models Included

This repo includes all required model files:

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

## Local Run

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

Set frontend API URL if needed:

```bash
# frontend/.env.local
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Render Deployment

Two services are recommended:

1. **Backend Web Service**
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

2. **Frontend Web Service**
   - Root Directory: `frontend`
   - Build Command: `npm install && npm run build`
   - Start Command: `npm run start -- -p $PORT`
   - Environment Variable:
     - `NEXT_PUBLIC_API_BASE_URL=https://<your-backend-service>.onrender.com`

## Notes

- Backend model loading resolves local repo models first (`Eye-Disease-App/Models`).
- `backend/api_keys.env` is intentionally git-ignored. Add your real keys on Render using environment variables.
- If browser favicon cache is stale, do a hard refresh after deploy.
