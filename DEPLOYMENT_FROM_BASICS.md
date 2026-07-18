# OphthalmicAI deployment from basics

## Intended public scope

The public application serves diabetic-retinopathy severity grading, five DR-related
segmentation outputs, and Grad-CAM. Glaucoma and cataract remain research-only
transfer-learning experiments and are not exposed by the API or interface.

## Recommended free demonstration architecture

- Backend: Hugging Face Docker Space, CPU Basic.
- Frontend: Vercel Hobby project connected to the `frontend` directory.
- Source control: GitHub.

The backend production image has already been built locally and passed
`/api/health` with the real DR and segmentation checkpoints loaded.

## 1. Publish the repository

1. Install GitHub CLI from <https://cli.github.com/>.
2. Open a new terminal and run `gh auth login`.
3. Select GitHub.com, HTTPS, and browser authentication.
4. Confirm with `gh auth status`.
5. Publish the prepared branch and merge its pull request after reviewing the files.

Large `.pt` and `.pth` checkpoints require Git LFS:

```text
git lfs install
git lfs pull
```

## 2. Create the Hugging Face backend

1. Create or sign in to a Hugging Face account.
2. Open Spaces and choose **Create new Space**.
3. Name it `ophthalmicai-backend`.
4. Choose **Docker** as the SDK and **CPU Basic** as hardware.
5. Choose Public visibility for a freely accessible demonstration.
6. Clone the new Space repository.
7. Copy the following project items into the Space repository:
   - `Dockerfile`
   - `.dockerignore`
   - `backend/`
   - `Models/OphthalmicAI_Final_EffNetB3.pt`
   - `Models/segmentation/`
8. Replace the Space repository's README with `HUGGINGFACE_SPACE_README.md`.
9. Commit and push. Hugging Face will build and start the container automatically.
10. Open:

```text
https://YOUR-HF-USERNAME-ophthalmicai-backend.hf.space/api/health
```

The response must contain:

```json
{
  "status": "ok",
  "gradcam_available": true
}
```

Optional Space secrets:

- `GEMINI_API_KEY`: enables the Gemini-backed assistant.
- `DATABASE_URL`: external PostgreSQL connection if persistent accounts are needed.

Without an external database, SQLite data can disappear when free infrastructure
restarts or sleeps. Do not treat the demo database as permanent storage.

## 3. Create the Vercel frontend

1. Create or sign in to Vercel using GitHub.
2. Select **Add New -> Project**.
3. Import `VishwamModi/OpthalmicAI`.
4. Set **Root Directory** to `frontend`.
5. Confirm framework preset **Next.js**.
6. Add these environment variables for Production, Preview, and Development:

```text
NEXT_PUBLIC_API_BASE_URL=https://YOUR-HF-USERNAME-ophthalmicai-backend.hf.space
NEXT_PUBLIC_APP_OVERVIEW_YT=https://www.youtube.com/embed/opcj0hZPnxU
```

7. Deploy.
8. Open the generated `vercel.app` URL.

## 4. End-to-end acceptance test

Complete every check before sharing the URL:

1. Landing page loads and the updated YouTube video plays.
2. Signup and login succeed.
3. A valid retinal fundus image uploads.
4. DR grade and raw analysis return without an API error.
5. Grad-CAM is visible.
6. OD, EX, SE, MA, and HE segmentation panels load.
7. Report export completes.
8. Invalid file types and oversized files show a useful error.
9. Browser developer tools show no mixed-content or CORS errors.
10. `/api/health` still returns `status: ok` after analysis.

## 5. Alternative one-provider Render deployment

The root `render.yaml` defines a Docker backend and Node frontend. In Render:

1. Choose **New -> Blueprint**.
2. Connect the GitHub repository.
3. Select the branch containing `render.yaml`.
4. Set `NEXT_PUBLIC_API_BASE_URL` to the backend `onrender.com` URL after the
   backend service is created.
5. Deploy and repeat the acceptance tests.

Render free web services sleep after inactivity and use an ephemeral filesystem.
The ML backend can also exceed small-instance memory limits. For this project,
Hugging Face CPU Basic is the preferred free demonstration backend.

## 6. Research-integrity checks before demonstration

- Do not describe synthetic application confidence values as calibrated probability.
- Do not use the synthetic DCA fallback in the paper.
- Keep the interface and README explicit that glaucoma and cataract are not deployed.
- Add a visible research-only / not-for-clinical-use statement.
- Do not store identifiable patient data on free public infrastructure.
