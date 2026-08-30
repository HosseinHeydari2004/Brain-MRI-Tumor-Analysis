# 🧠 Brain MRI Tumor Analysis

AI pipeline for **brain tumor classification and segmentation** on MRI scans,
built on the **BRISC 2025** dataset. A ResNet34 backbone classifies each scan
into `glioma`, `meningioma`, `pituitary`, or `no_tumor`, and — when a tumor is
detected — a **ResNet34 encoder + UNet3+ decoder** segments the exact tumor
region, localizes it with bounding boxes, and produces an overlay/report image.

The project ships with a **presentation-ready Streamlit demo**, a **FastAPI
backend**, and Docker support, so it can be run as a single local demo or
deployed as two independent services.

> ⚠️ **Disclaimer:** This project is a research / portfolio demonstration.
> It is **not** a certified medical device and must never be used for real
> clinical decision-making without review by a qualified professional.

---

## 1. Features

- **Two-stage pipeline:** classification first, segmentation only runs when a
  tumor is predicted (faster, and avoids drawing a mask on a healthy scan).
- **Segmentation model:** ResNet34 encoder + UNet3+-style decoder with
  full-scale skip connections between all encoder stages.
- **Classification model:** ResNet34 backbone adapted to single-channel
  (grayscale) MRI input.
- **Visual analysis:** tumor overlay, bounding boxes, convex hull, region
  centroids, and a combined side-by-side report figure — downloadable as PNG.
- **Streamlit demo app:** polished, client-facing UI with live status
  indicators, confidence bars, metric cards, and a downloadable report.
- **FastAPI backend:** REST API (`/predict/classification`,
  `/predict/segmentation`, `/predict/full-analysis`) with interactive docs at
  `/docs`, so the models can be consumed by any other client.
- **Graceful degradation:** both apps start up fine even without trained
  weights present — they clearly show a "model not loaded" state instead of
  crashing, so the UI can always be reviewed.
- **CPU/GPU agnostic:** auto-detects CUDA, falls back to CPU with a warning
  instead of failing (useful for demoing on a laptop without a GPU).

---

## 2. Project structure

```
.
├── app/
│   ├── api/
│   │   ├── main.py           # FastAPI backend (REST API)
│   │   └── schemas.py        # Pydantic request/response models
│   └── streamlit_app/
│       └── app.py            # Streamlit demo UI
├── src/
│   ├── model.py               # SegmentationModel (ResNet34+UNet3+), ClassificationModel (ResNet34)
│   ├── transforms.py          # Shared train/inference preprocessing
│   ├── utils.py                # load_model, predict_mask, predict_tumor_class, download_data
│   └── visualization.py        # Overlay / bbox / hull / centroid / report figure helpers
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/                    # Place trained .pth weights here (see models/README.md)
├── requirements.txt
└── README.md
```

---

## 3. Quickstart (fastest path to a running demo)

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd brain-tumor-segmentation_and_classification

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your trained weights (see models/README.md)
#    models/best_model_classif.pth
#    models/best_model_seg.pth

# 5. Run the demo
streamlit run app/streamlit_app/app.py
```

Open **http://localhost:8501** — that's the whole demo, in one command
(step 5) once the environment and weights are in place.

> If the weight files aren't available yet, the app still opens and clearly
> shows which model is missing — everything else in the UI can be reviewed.

---

## 4. Installation details

### 4.1 Python environment

Requires **Python 3.10+**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 4.2 PyTorch (CPU vs. GPU)

`requirements.txt` installs a working CPU build of `torch`/`torchvision` by
default. If you have an NVIDIA GPU and want CUDA acceleration, install the
matching build **before** the rest of the requirements — pick your CUDA
version at <https://pytorch.org/get-started/locally/>, for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The app auto-detects CUDA at runtime (`torch.cuda.is_available()`) — no
config needed either way.

### 4.3 Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4.4 Model weights

Place your two trained checkpoints in `models/` (see
[`models/README.md`](models/README.md) for exact filenames and format).
Without them, the apps still run — analysis is simply disabled until the
weights are added.

---

## 5. Running the Streamlit demo

```bash
streamlit run app/streamlit_app/app.py
```

- App: **http://localhost:8501**
- Upload an MRI slice (PNG/JPG), click **Run Analysis**.
- The sidebar shows live status of the device (CPU/GPU) and whether each
  model loaded successfully.
- Results include: predicted class + confidence, per-class probability bars,
  tumor coverage %, number of regions, and (for detected tumors) a full
  visual report with a downloadable PNG.

---

## 6. Running the FastAPI backend

```bash
uvicorn app.api.main:app --reload --port 8000
```

- API root: **http://localhost:8000**
- Interactive docs (Swagger UI): **http://localhost:8000/docs**

### Endpoints

| Method | Path                          | Description                                                                 |
|--------|-------------------------------|-------------------------------------------------------------------------------|
| GET    | `/health`                      | Service + model load status, current device.                                |
| POST   | `/predict/classification`      | Upload an image, get the predicted tumor class + probabilities.             |
| POST   | `/predict/segmentation`        | Upload an image, get the segmentation mask, overlay (base64 PNG), and stats.|
| POST   | `/predict/full-analysis`       | Runs classification, then segmentation only if a tumor is detected.         |

Example request:

```bash
curl -X POST http://localhost:8000/predict/full-analysis \
  -F "file=@sample_scan.png"
```

A missing model returns `503` with a clear `detail` message rather than a
generic server error.

---

## 7. Running with Docker

Two services (`api` on port 8000, `streamlit` on port 8501), sharing the
same image and mounting `models/` read-only:

```bash
cd docker
docker compose up --build
```

- Streamlit demo → **http://localhost:8501**
- API + docs → **http://localhost:8000/docs**

To run only the Streamlit demo as a single container:

```bash
docker build -f docker/Dockerfile -t brain-mri-app .
docker run -p 8501:8501 -v "$(pwd)/models:/workspace/models:ro" brain-mri-app
```

---

## 8. Architecture notes

**Segmentation — `SegmentationModel`** (`src/model.py`)
- Encoder: ResNet34 (ImageNet-pretrained), first conv adapted to 1-channel
  input by averaging the pretrained RGB filters.
- Decoder: UNet3+-style full-scale skip connections — every decoder stage
  fuses **all** encoder feature maps (resized to a common resolution) instead
  of only the matching-resolution one, then fuses them through a projection +
  convolutional fusion block.
- Output: 2-channel logits (background vs. tumor) → softmax → argmax mask.

**Classification — `ClassificationModel`** (`src/model.py`)
- ResNet34 backbone, first conv adapted to 1-channel input the same way, final
  FC layer replaced with a 4-way classification head.

**Shared preprocessing** (`src/transforms.py`) is the single source of truth
used at both training and inference time, so predictions always match the
exact pipeline the model was trained on.

---

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Sidebar shows "Not loaded" for a model | The corresponding `.pth` file isn't in `models/` yet, or the filename doesn't match exactly — see `models/README.md`. |
| `RuntimeError: Error loading model from ...` | The checkpoint doesn't match the model architecture/kwargs (e.g. wrong `num_classes` or `out_channels`). Re-check how the model was instantiated during training. |
| App is very slow on CPU | Expected — segmentation especially benefits from a GPU. Install a CUDA build of PyTorch (see §4.2) if you have an NVIDIA GPU. |
| `Address already in use` when starting Streamlit/uvicorn | Another process is using port 8501/8000. Add `--server.port` / `--port` with a free port. |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the **project root**, not from inside `app/`. Both `app.py` and `main.py` already add the project root to `sys.path`, but the working directory still needs to make `requirements.txt`/`models/` resolvable. |

---

## 10. Author

**Hossein Heydari** — Computer Engineering (Software), AI Builders Iran
GitHub: [@HosseinHeydari2004](https://github.com/HosseinHeydari2004)
