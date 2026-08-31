# NHA-4-48: Student Attention Monitoring via Full-Face Gaze Estimation

A research-oriented computer-vision project for estimating where a student is looking and using that gaze direction as a proxy for attention and focus during learning activities. The repository centers on a PyTorch-based ResNet-50 gaze regression model, image normalization for head pose compensation, and live webcam-based inference to approximate a student's attention on a screen.

This project addresses a practical classroom and e-learning problem: detecting whether a learner is looking at the lesson content, drifting away, or displaying reduced attention, using non-invasive camera-based analysis.

## ✨ Key Features

- Full-face gaze estimation using a ResNet-50 architecture
- Head pose normalization and gaze-angle regression in yaw and pitch
- MPIIFaceGaze-compatible preprocessing pipeline for face images
- Real-time webcam inference for gaze direction estimation
- Screen calibration logic to convert gaze angles into on-screen coordinates
- Attention monitoring and keyboard interaction demo built on gaze predictions
- Training and evaluation workflow embedded in a Jupyter notebook
- Model history and test visualizations included in the repository assets

## 🧰 Tech Stack & Architecture

The codebase is primarily a Python research notebook and local CV demo, with the following technologies clearly used in the implementation:

- Python 3.x
- PyTorch for deep learning model training and inference
- TorchVision for ResNet-50 backbone and image transforms
- OpenCV (`cv2`) for face detection, image preprocessing, normalization, and webcam processing
- NumPy for numerical computations and matrix operations
- SciPy (`scipy.io`) for MATLAB file loading used in gaze dataset calibration data
- Matplotlib for visualization and metrics plots
- `tqdm` for progress indicators during training/evaluation
- Jupyter Notebook for the main development workflow
- `screeninfo` and `ffpyplayer` for screen-size-aware and video/audio monitoring support in later demo cells

Architecture-wise, the project follows a standard gaze-estimation pipeline:

1. Face detection and crop extraction from the webcam image
2. Full-face preprocessing and normalization using camera/head geometry
3. ResNet-based regression from image to gaze angles (`yaw`, `pitch`)
4. Calibration from gaze angles to screen coordinates
5. Real-time visual feedback and attention-related interaction logic

## 📁 Project Structure

```text
.
├── README.md
├── commands.txt
├── full_face_resnet.ipynb             # Main training, preprocessing, and inference notebook
├── gaze_project.zip                   # Archived project bundle / related materials
├── gaze_resnet_v3_history.png         # Training history visualization
├── gaze_resnet_v3_test_p00.png        # Test visualization / output chart
├── s2.mp4                             # Video asset used in demo workflows
├── 000_up_n1_gt(758,257)_pred(786,422)_err167px.jpg
├── 001_left_n2_gt(676,469)_pred(881,536)_err215px.jpg
├── 003_up_left_n1_gt(627,246)_pred(975,292)_err351px.jpg
├── Requirements_Gaze_Estimation .pdf  # Project reference / requirements document
├── Student Monitoring System.pdf      # Related design or project reference
├── System_Analysis_&_Design_Students_Focus_Monitoring_System_using_Gaze_Estimation.pdf
├── plan.pdf                           # Project plan / documentation
└── .git/                              # Repository metadata (not source code)
```

### Key entry points

- `full_face_resnet.ipynb`: the central implementation for model definition, training pipeline, data normalization, calibration, and webcam demos.
- `commands.txt`: contains local launch commands for a backend/front-end flow and development server usage.
- `gaze_resnet_v3_history.png` and related outputs: visual references for model behavior and evaluation.

> Note: This repository does not currently include a packaged web backend or application source tree. The model code and live demo logic are embedded in the notebook rather than being split into separate Python modules.

## ⚙️ Prerequisites & Getting Started

### System requirements

- Python 3.10+ recommended
- 8 GB RAM minimum; 16 GB preferred for training workflows
- Webcam for live gaze detection
- Optional: NVIDIA GPU with CUDA support for faster training/inference
- Linux, macOS, or Windows environment with OpenCV and camera support

### Install dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install torch torchvision opencv-python numpy scipy matplotlib tqdm jupyter screeninfo ffpyplayer
```

If you want a CUDA-enabled installation, install the appropriate PyTorch build for your machine from the official PyTorch website before installing the other packages.

### Dataset and model paths

The notebook contains hardcoded local paths such as:

```python
root = r'd:\DEPI\CV_projects\Gaze_full_face\MPIIFaceGaze'
SAVE_DIR = r'D:\DEPI\CV_projects\Gaze_full_face\v2_camera_check\checkpoints'
```

These are local machine paths and must be updated for your environment before running training or inference. The repository currently does not include a bundled dataset or checkpoint directory.

### Environment variables

No formal `.env` file or environment-variable-based configuration is present in the repository. Configuration is mostly defined directly inside the notebook through Python variables and local filesystem paths.

## 🚀 Usage

### 1. Launch the notebook

Open the project in Jupyter and run the cells in `full_face_resnet.ipynb` in order:

```bash
jupyter notebook
```

Then open `full_face_resnet.ipynb` and execute the cells sequentially.

### 2. Model training workflow

The notebook includes:

- Model definition (`GazeResNet`)
- Dataset creation for MPIIFaceGaze samples
- Normalization logic for face images and gaze vectors
- Training loop with optimizer and scheduler setup
- Validation and evaluation reporting
- Model checkpoint generation

### 3. Live webcam demo

The notebook contains later cells that use `cv2.VideoCapture(0)` for real-time gaze inference and calibration. These cells:

- detect the face in the webcam feed
- crop the face region
- preprocess it for the model
- estimate gaze angles in yaw/pitch
- map the gaze vector to screen coordinates
- visualize attention and screen interaction behavior

### 4. Local development commands

The repository includes `commands.txt`, which references a backend and frontend server flow:

```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
python -m http.server 5000
```

These commands appear to be a reference for a local web demo, but the actual backend server code is not present in the checked-in repository. The current source of truth remains the notebook-based CV pipeline.

## 🧠 API / Core Modules Overview

Because this repository is notebook-centric rather than a formal service API, the main "modules" are functions and classes defined inside `full_face_resnet.ipynb`.

### Core pieces

- `GazeResNet`: custom PyTorch model based on ResNet-50 for gaze regression
- `GazeDataset`: dataset wrapper for training on MPIIFaceGaze-style samples
- `get_normalization_matrix(...)`: builds the geometric normalization matrix for face alignment
- `normalize_image(...)`: warps the input image into a normalized frontal-facing view
- `normalize_gaze(...)`: converts gaze vectors into yaw/pitch values after geometry normalization
- `process_sample(...)`: encapsulates the full sample preprocessing flow for a single image
- `GazeCalibrator`: maps gaze predictions into screen-space coordinates for interaction demos
- `preprocess_face(...)`: standardizes webcam image input for the model during inference

### Inference behavior

The system regresses two values for each sample:

- `yaw`: horizontal gaze angle
- `pitch`: vertical gaze angle

These values are then used to estimate where the user is looking on a screen or whether the gaze is focused on expected target regions.

## 🗺️ Future Roadmap

This project is a strong prototype and research foundation. Suggested next steps include:

- Consolidate notebook logic into a clean Python package structure (`src/`, `data/`, `models/`)
- Add a proper dataset loader and reproducible training configuration
- Package the model into a CLI or inference API
- Improve calibration robustness for real-world classroom environments
- Add attention score aggregation over time (e.g., per-minute focus metrics)
- Create a web dashboard for classroom monitoring and analytics
- Support multi-student tracking and per-student gaze reporting
- Add tests and CI for preprocessing and inference pipeline validation

## 🤝 Contributing

Contributions are welcome. A typical workflow would be:

1. Fork the repository
2. Create a feature branch
3. Implement your model, calibration, or data improvements
4. Validate the notebook logic and update relevant documentation
5. Submit a pull request with a clear summary of the change

For this project, especially, please preserve the notebook’s reproducibility and clearly document any changes to paths, preprocessing assumptions, or training configuration.

## 📄 License

No explicit license file or licensing statement is present in the repository at this time. Please add a project license before public distribution or production use.

A common choice for this kind of project would be:

- MIT License for simplicity and broad adoption
- Apache 2.0 for enterprise or research-oriented distribution

---

This project represents a practical AI prototype for monitoring student attention using gaze estimation. It is best viewed as a research and demonstration pipeline rather than a fully packaged production application.
