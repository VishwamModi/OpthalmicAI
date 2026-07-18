# OphthalmicAI deployment from basics

## Public demonstration scope

The free public demonstration contains:

- diabetic-retinopathy severity grades 0–4;
- the raw ordinal model score;
- clearly labelled uncalibrated ordinal proximity;
- a Grad-CAM visualization;
- the updated project video.

Glaucoma, cataract, segmentation, comparative models, user accounts, and LLM
reports remain research or full local-application features. They are not part
of the free public Space.

## Recommended free deployment

Use a **public Hugging Face Gradio Space on CPU Basic**. It runs Python and
PyTorch without the Docker SDK. A Static Space can host only HTML/JavaScript
and cannot execute the local EfficientNet checkpoint.

The ready-to-upload package is in `huggingface_space/`.

## 1. Keep GitHub on main

Run Git commands inside the repository, not from `C:\Users\VISHWAM`:

```powershell
cd "C:\Users\VISHWAM\OneDrive\Desktop\Research\MAJOR_PROJECT\OpthalmicAI-publish"
git lfs install
git lfs pull
git status
```

The deployment and publication files are maintained directly on `main`.

## 2. Create the free Hugging Face Space

1. Sign in to Hugging Face.
2. Open **Spaces → Create new Space**.
3. Name it `ophthalmicai-dr-severity`.
4. Select **Gradio** as the SDK.
5. Select **Public** visibility.
6. Keep the free **CPU Basic** hardware.
7. Create the Space.

Do not select Static: it cannot run PyTorch inference. Docker is not required.

## 3. Upload the Space files

Upload these files to the root of the new Space:

- `huggingface_space/README.md`
- `huggingface_space/app.py`
- `huggingface_space/requirements.txt`
- `Models/OphthalmicAI_Final_EffNetB3.pt`

The final Space repository must look like:

```text
README.md
app.py
requirements.txt
OphthalmicAI_Final_EffNetB3.pt
```

Hugging Face will install the dependencies and start the app. The first build
and first CPU prediction may take several minutes.

## 4. Test before sharing

1. The Space status becomes **Running**.
2. Upload a valid retinal fundus image.
3. Confirm a DR grade from 0 to 4 is shown.
4. Confirm the raw ordinal score is shown.
5. Confirm the confidence-like value says **uncalibrated**.
6. Confirm the Grad-CAM image appears.
7. Confirm the research-only disclaimer is visible.
8. Test a non-retinal image and document that this is an unsupported input.
9. Do not upload identifiable patient information.

The public URL will be similar to:

```text
https://YOUR-USERNAME-ophthalmicai-dr-severity.hf.space
```

## 5. Static-only fallback

If the account interface genuinely offers only Static Spaces, deploy only a
landing page and the YouTube video there. Static hosting cannot perform DR
inference with the `.pt` model. Do not present a random or browser-mocked
prediction as AI output.

## Research-integrity rules

- The ordinal proximity is not a calibrated probability.
- The Space is a research demonstration, not a medical device.
- Do not include synthetic decision-curve or confidence values in the paper.
- Glaucoma, cataract, segmentation, and competing-model experiments belong in
  the journal evidence package, not the current public application.
