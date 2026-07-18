# OphthalmicAI free deployment

## Recommended provider: Streamlit Community Cloud

Streamlit Community Cloud connects directly to GitHub and hosts public Python
applications for free. The deployment is intentionally limited to:

- diabetic-retinopathy severity grades 0–4;
- the raw ordinal score;
- explicitly uncalibrated ordinal proximity;
- Grad-CAM;
- the updated project video.

The ready application is `streamlit_app/app.py`. It loads
`Models/OphthalmicAI_Final_EffNetB3.pt` directly from this repository.

## Deployment steps

1. Confirm the repository is public:
   `https://github.com/VishwamModi/OpthalmicAI`
2. Open `https://share.streamlit.io`.
3. Sign in using GitHub.
4. Select **Create app**.
5. Choose:
   - Repository: `VishwamModi/OpthalmicAI`
   - Branch: `main`
   - Main file path: `streamlit_app/app.py`
6. Open **Advanced settings** and select Python 3.11 if a version is requested.
7. Deploy.
8. Wait for PyTorch, timm and OpenCV to install. The first build and first CPU
   prediction may take several minutes.

The generated address will be similar to:

```text
https://ophthalmicai.streamlit.app
```

## Acceptance test

1. The updated video plays.
2. PNG, JPG and JPEG fundus images can be selected.
3. The app displays DR grade 0–4 and the raw ordinal score.
4. Ordinal proximity is explicitly marked uncalibrated.
5. Grad-CAM appears.
6. The research-only disclaimer remains visible.
7. No identifiable patient information is uploaded.

## Free alternatives

### Modal Starter

Modal currently includes monthly compute credits and can host web functions.
It is a good fallback if Streamlit cannot fit the PyTorch runtime, but requires
converting the application to Modal's deployment format.

### Google Cloud Run

Cloud Run has a monthly free usage allowance and can run the existing Docker
image with more memory. It normally requires a Google Cloud billing account,
and charges are possible if the free allowance is exceeded.

### Render

Render offers free web services, but the free instance has approximately
512 MB RAM. This is too small to recommend for the PyTorch application.

### Static hosts

GitHub Pages, Hugging Face Static Spaces, Netlify and Vercel static hosting can
publish the landing page and video, but cannot execute the local `.pt` model.

## Research-integrity rules

- The public application is DR-only.
- Ordinal proximity is not a calibrated probability.
- Do not expose glaucoma, cataract or competing-model experiments in the app.
- Do not use synthetic confidence or decision-curve values.
- The deployment is a research demonstration, not a medical device.
