# 🧠 Brain MRI Tumor Analysis

An end-to-end **Deep Learning pipeline for brain tumor classification and segmentation** on MRI scans, built using the **BRISC 2025** dataset.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Inference%20API-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Classical%20CV-5C3EE8?logo=opencv&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)


The system first classifies an MRI scan into one of four categories:

- `glioma`
- `meningioma`
- `pituitary`
- `no_tumor`

When a tumor is detected, the pipeline performs **tumor segmentation** to identify the affected region, followed by visual localization using bounding boxes, convex hulls, centroids, and an overlay visualization.

The project includes a **Streamlit web application**, **FastAPI REST API**, and **Docker support**, making it suitable for local demonstrations and portfolio deployment.

> ⚠️ **Disclaimer:** This project is a research and portfolio demonstration. It is **not a certified medical device** and must not be used for real clinical diagnosis or treatment decisions. Any clinical use would require appropriate validation and review by qualified medical professionals.

---

# Demo

<p align="center">
  <img src="Demo/Demo_clip.gif"  alt="Project Demo">
</p>

---

## 1. Features

### 🔍 Two-Stage AI Pipeline

The system follows a two-stage workflow:

```text
MRI Image
    │
    ▼
┌─────────────────────┐
│ Tumor Classification│
└──────────┬──────────┘
           │
     Tumor detected?
       │         │
      No        Yes
       │         │
       ▼         ▼
   No Tumor   Segmentation
                 │
                 ▼
          Tumor Localization
                 │
                 ▼
          Visual Analysis
```

Segmentation is executed only when the classification model predicts a tumor.

### 🧠 Classification

- ResNet34 backbone
- Adapted for single-channel grayscale MRI images
- Four-class classification:
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor

### 🎯 Segmentation

- ResNet34 ImageNet-pretrained encoder
- UNet3+-style decoder
- Full-scale feature fusion between encoder and decoder stages
- Binary tumor segmentation
- Single-channel output
- Sigmoid activation for tumor probability
- Binary mask generation using a probability threshold

### 📊 Visual Analysis

The segmentation result is further processed to generate:

- Tumor mask
- Tumor probability map
- Tumor coverage percentage
- Tumor confidence
- Bounding boxes
- Convex hull
- Region centroids
- Tumor overlay
- Combined visual report
- Downloadable PNG report

### 🖥️ Streamlit Application

The project provides a presentation-ready Streamlit interface with:

- MRI image upload
- Classification results
- Confidence scores
- Per-class probabilities
- Segmentation results
- Tumor coverage
- Bounding boxes
- Visual overlays
- Model/device status
- Downloadable analysis report

### ⚡ FastAPI Backend

REST API endpoints are provided for integrating the models with other applications:

- `/health`
- `/predict/classification`
- `/predict/segmentation`
- `/predict/full-analysis`

Interactive API documentation is available through Swagger UI at `/docs`.

### 🐳 Docker Support

The project includes Docker configuration for running:

- Streamlit frontend
- FastAPI backend

Both services can share the same Docker image while using the trained model weights through a read-only volume.




### 💻 GPU Requirement

This project requires an **NVIDIA GPU with CUDA support** for model inference.

The application does **not** fall back to CPU. If CUDA is unavailable while the application requests GPU execution, a `RuntimeError` is raised indicating that an NVIDIA GPU with CUDA support is required.

The device selection logic is:

```text
NVIDIA GPU + CUDA available
            │
            ▼
          CUDA
            │
            │
     CUDA unavailable
            │
            ▼
       RuntimeError
       GPU required
```

For example:

```text
RuntimeError:
CUDA is not available.
This model is intended to run on an NVIDIA GPU.
Please run the code on a machine with CUDA support.
```

Therefore, make sure that:

- An NVIDIA GPU is available.
- NVIDIA drivers are correctly installed.
- A CUDA-compatible PyTorch installation is used.
- `torch.cuda.is_available()` returns `True`.

You can verify CUDA availability with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```text
True
```

> **Note:** CPU-only execution is intentionally not supported because the trained models and intended deployment environment are designed for NVIDIA GPU inference.

---

## 2. Dataset

This project uses the **BRISC 2025** dataset for brain MRI classification and segmentation.

The dataset contains brain MRI images associated with different tumor categories and segmentation annotations.

Dataset:

**BRISC 2025 — Brain Tumor MRI Dataset**

The models in this project were trained using preprocessing pipelines designed specifically for grayscale MRI images.

---

## 3. Project Structure

```text
brain-tumor-classification-segmentation/
│
├── app/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   └── streamlit_app/
│       └── app.py
│
├── src/
│   ├── model.py
│   ├── transforms.py
│   ├── utils.py
│   └── visualization.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── models/
│   ├── best_model_classif.pth
│   ├── best_model_seg.pth
│   └── README.md
│
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

### Main Components

| File / Directory | Description |
|---|---|
| `app/streamlit_app/app.py` | Streamlit web application |
| `app/api/main.py` | FastAPI REST API |
| `app/api/schemas.py` | Pydantic request/response schemas |
| `src/model.py` | Classification and segmentation architectures |
| `src/transforms.py` | Shared preprocessing and augmentation |
| `src/utils.py` | Model loading and inference utilities |
| `src/visualization.py` | Mask, bounding box, overlay and report utilities |
| `models/` | Trained model checkpoints |
| `docker/` | Docker configuration |

---

## 4. Quickstart

### 4.1 Clone the Repository

```bash
git clone <your-repo-url>
cd brain-tumor-classification-segmentation
```

### 4.2 Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4.3 Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4.4 Add Model Weights

Place the trained model checkpoints inside the `models/` directory:

```text
models/
├── best_model_classif.pth
└── best_model_seg.pth
```

See `models/README.md` for the expected model architecture and checkpoint format.

### 4.5 Run the Streamlit Application

```bash
streamlit run app/streamlit_app/app.py
```

Open:

```text
http://localhost:8501
```

Upload an MRI image and run the analysis.

---

## 5. Model Architecture

### 5.1 Classification Model

The classification model uses **ResNet34** as the backbone.

Because brain MRI images are grayscale, the first convolutional layer is adapted from three input channels to one input channel.

```text
MRI Image
  │
  ▼
Grayscale Input
  │
  ▼
ResNet34
  │
  ▼
Feature Extraction
  │
  ▼
Fully Connected Layer
  │
  ▼
4 Classes
```

Output classes:

```text
0 → glioma
1 → meningioma
2 → no_tumor
3 → pituitary
```

The exact class-index mapping should match the training configuration used by the classification model.

---

### 5.2 Segmentation Model

The segmentation model combines:

- ResNet34 encoder
- UNet3+-style decoder
- Full-scale feature fusion
- Single-channel binary segmentation output

Architecture overview:

```text
                    ResNet34 Encoder
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
      Stage 1          Stage 2          Stage 3
        │                │                │
        └────────┬───────┴────────┬───────┘
                 │                │
                 ▼                ▼
             UNet3+ Decoder
                 │
                 ▼
          1-Channel Output
                 │
                 ▼
              Sigmoid
                 │
                 ▼
        Tumor Probability Map
                 │
                 ▼
          Binary Tumor Mask
```

The segmentation model produces a **single-channel logit map**:

```text
(B, 1, H, W)
```

The output is converted to tumor probabilities using:

```python
torch.sigmoid(output)
```

A probability threshold is then used to generate the binary segmentation mask:

```text
0 → background
1 → tumor
```

---

## 6. Preprocessing

The project uses shared preprocessing utilities to keep training and inference consistent.

### Classification

```text
MRI Image
   ↓
Grayscale
   ↓
Resize
   ↓
Tensor Conversion
   ↓
Normalization
   ↓
Classification Model
```

### Segmentation

```text
MRI Image
   ↓
Grayscale
   ↓
Resize to 512 × 512
   ↓
Tensor Conversion
   ↓
Normalization
   ↓
Segmentation Model
```

Using the same preprocessing pipeline during training and inference helps prevent inconsistencies between model development and deployment.

---

## 7. Running the Streamlit Demo

```bash
streamlit run app/streamlit_app/app.py
```

The application runs at:

```text
http://localhost:8501
```

### Workflow

1. Upload an MRI scan.
2. Run the analysis.
3. The classification model predicts the tumor class.
4. If `no_tumor` is predicted, segmentation is skipped.
5. If a tumor is detected, the segmentation model generates a tumor mask.
6. The application calculates tumor coverage and confidence.
7. Tumor regions are localized using bounding boxes and additional geometric analysis.
8. A visual report is generated.

---

## 8. FastAPI Backend

Start the API with:

```bash
uvicorn app.api.main:app --reload --port 8000
```

API:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service and model status |
| `POST` | `/predict/classification` | Classify an MRI image |
| `POST` | `/predict/segmentation` | Generate a tumor segmentation |
| `POST` | `/predict/full-analysis` | Run classification followed by conditional segmentation |

### Example

```bash
curl -X POST http://localhost:8000/predict/full-analysis \
  -F "file=@sample_scan.png"
```

If a required model is unavailable, the API returns an appropriate error response instead of failing with an unhandled exception.

---

## 9. Docker

The project supports running both services with Docker Compose.

### Start Both Services

From the project root:

```bash
docker compose -f docker/docker-compose.yml up --build
```

Services:

```text
Streamlit → http://localhost:8501
FastAPI   → http://localhost:8000
Swagger   → http://localhost:8000/docs
```

### Run Only Streamlit

```bash
docker build -f docker/Dockerfile -t brain-mri-app .
```

Then:

```bash
docker run \
  -p 8501:8501 \
  -v "$(pwd)/models:/workspace/models:ro" \
  brain-mri-app
```

On Windows PowerShell, the volume syntax may need to be adjusted according to the local Docker setup.

---

## 10. Inference Pipeline

The complete inference pipeline is:

```text
                    ┌──────────────┐
                    │  MRI Image   │
                    └──────┬───────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │  Classification  │
                 │     ResNet34     │
                 └────────┬─────────┘
                          │
                ┌─────────┴─────────┐
                │                   │
             No Tumor             Tumor
                │                   │
                ▼                   ▼
          Final Result       ┌──────────────┐
                             │ Segmentation │
                             │ ResNet34 +   │
                             │   UNet3+     │
                             └──────┬───────┘
                                    │
                                    ▼
                              Tumor Mask
                                    │
                         ┌──────────┼──────────┐
                         │          │          │
                         ▼          ▼          ▼
                       BBox      Overlay    Geometry
                         │          │          │
                         └──────────┼──────────┘
                                    │
                                    ▼
                              Final Report
```

---

## 11. Output

For a tumor-positive MRI, the system can provide:

### Classification

```text
Predicted Class: Glioma
Confidence: 94.2%
```

### Segmentation

```text
Tumor Coverage: 8.37%
Segmentation Confidence: 91.6%
```

### Visual Results

```text
Original MRI
      +
Tumor Mask
      +
Bounding Box
      +
Tumor Overlay
      +
Geometric Localization
```

The final visualization can be exported as a PNG report.

---

## 12. Graceful Model Loading

The applications are designed to remain usable even when model weights are unavailable.

If a checkpoint is missing:

```text
Model not loaded
```

is displayed instead of causing the entire application to crash.

This makes it possible to review the UI and application structure independently from the trained checkpoints.

---

## 13. CPU / GPU Configuration

The application automatically detects CUDA:

```python
torch.cuda.is_available()
```

If CUDA is available:

```text
Device → CUDA
```

Otherwise:

```text
Device → CPU
```

This allows the project to run across different environments without requiring manual device configuration.

---

## 14. Technologies

### Machine Learning

- Python
- PyTorch
- Torchvision
- ResNet34
- UNet3+-style architecture
- OpenCV
- NumPy
- Pillow

### Backend

- FastAPI
- Pydantic
- Uvicorn

### Frontend

- Streamlit

### Deployment

- Docker
- Docker Compose

### Dataset

- BRISC 2025

---

## 15. Future Improvements

Potential future improvements include:

- [ ] Improve segmentation performance with additional augmentation
- [ ] Experiment with Dice + BCE / Focal-based losses
- [ ] Add Dice, IoU and HD95 evaluation dashboards
- [ ] Add model explainability methods such as Grad-CAM
- [ ] Add DICOM support
- [ ] Add batch inference
- [ ] Add experiment tracking
- [ ] Add automated model versioning
- [ ] Add CI/CD pipeline
- [ ] Add production-grade authentication and API rate limiting
- [ ] Deploy the application to a cloud GPU environment

---

## 16. Authors & Contributors

### Author

**Hossein Heydari**  
Computer Engineering (Software) · AI Builders Iran

GitHub: [@HosseinHeydari2004](https://github.com/HosseinHeydari2004)

### Contributor

**Seyede Reyhane Khorashadizade**

GitHub: [@Seyede-Reyhane-Khorashadizade](https://github.com/Seyede-Reyhane-Khorashadizade)

---

## 17. Disclaimer

This project is intended exclusively for **research, education, and portfolio demonstration**.

The predictions, segmentation masks, confidence values, and visualizations generated by this system should **not** be considered medical advice or a clinical diagnosis.

The system has not been certified as a medical device and should not be used to make decisions regarding patient diagnosis, treatment, or medical care.