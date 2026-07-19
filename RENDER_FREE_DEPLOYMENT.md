# OphthalmicAI — Render Free Deployment

This deployment runs the original Next.js frontend as a free static site and
the DR-severity FastAPI service as one free Render web service.

## Free configuration

- Frontend: Next.js static export (`frontend/out`)
- Backend: EfficientNet-B3 DR severity only
- Backend instance: Render Free, 512 MB RAM, 0.1 CPU
- Persistent data: external Supabase Free PostgreSQL
- Segmentation: disabled on the hosted demo
- Grad-CAM: disabled on the hosted demo
- Worker count: one

Segmentation and Grad-CAM remain available for local/research runs by installing
the full requirements and setting `ENABLE_SEGMENTATION=true` and
`ENABLE_GRADCAM=true`.

## Deploy from GitHub

1. Sign in at https://dashboard.render.com using GitHub.
2. Choose **New > Blueprint**.
3. Select `VishwamModi/OpthalmicAI`.
4. Use the `main` branch and the repository-root `render.yaml`.
5. Enter `SUPABASE_DATABASE_URL` for the backend.
6. Initially enter a placeholder for the frontend
   `NEXT_PUBLIC_API_BASE_URL`, such as `https://example.onrender.com`.
7. Apply the Blueprint and wait for the backend to finish.
8. Copy the backend URL, normally
   `https://opthalmicai-backend.onrender.com`.
9. Open the frontend service settings and replace
   `NEXT_PUBLIC_API_BASE_URL` with that backend URL.
10. Trigger **Manual Deploy > Deploy latest commit** for the frontend.
11. Verify:
    - Backend: `/api/health`
    - Frontend: open the static-site URL
    - Create a test patient and submit one fundus image

## Expected free-tier behavior

The backend sleeps after 15 minutes without inbound traffic. Its first request
after sleeping can take about one minute, plus model-loading time. The frontend
does not sleep because it is a static site.

Render provides 750 free instance hours per workspace per month. If no payment
method is stored, services are suspended instead of charging when applicable
limits are exhausted. Render sends email notifications near and at limits.

## If the backend exceeds 512 MB

The next free option is browser-side ONNX inference on a static site. It removes
the Python backend for inference, but requires converting the checkpoint and
does not provide the current server-side Grad-CAM or database workflow.
