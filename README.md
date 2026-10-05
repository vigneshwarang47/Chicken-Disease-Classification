# 🐔 Chicken Disease Classification — End-to-End Deep Learning & MLOps

An end-to-end Deep Learning project that classifies chicken fecal images into **Healthy** and **Coccidiosis** categories using a Convolutional Neural Network (CNN) with transfer learning.

The project follows a modular MLOps-style architecture with data ingestion, base-model preparation, callbacks, model training, evaluation, DVC pipeline management, and a Flask-based prediction API.

---

## 📌 Project Overview

Chicken diseases can significantly affect poultry health and farm productivity. This project uses image classification to automatically identify whether a chicken is:

- **Healthy**
- **Coccidiosis**

The model is trained using chicken fecal images and uses a transfer-learning approach based on **VGG16 with ImageNet weights**.

The complete workflow is organized into reusable pipeline stages instead of keeping the implementation inside a single notebook.

---

## 🎯 Objective

Build an image classification system that can:

1. Ingest the chicken fecal image dataset.
2. Prepare a transfer-learning base model.
3. Train the CNN model.
4. Evaluate the trained model.
5. Serve predictions through a Flask application.

---

## 🏗️ Project Architecture

```text
                    Chicken Fecal Images
                            │
                            ▼
                  ┌──────────────────┐
                  │  Data Ingestion  │
                  └─────────┬────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ Prepare Base     │
                  │ Model / VGG16    │
                  └─────────┬────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ Data Generators  │
                  │ + Augmentation   │
                  └─────────┬────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ Model Training   │
                  │   CNN / VGG16    │
                  └─────────┬────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ Model Evaluation │
                  │ Loss / Accuracy  │
                  └─────────┬────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ Flask Prediction │
                  │       API        │
                  └──────────────────┘
```

---

## 📂 Project Structure

```text
Chicken-disease-Classification/
│
├── .dvc/
├── .github/
│   └── workflows/
│
├── artifacts/
│   └── data_ingestion/
│
├── config/
│   └── config.yaml
│
├── research/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_prepare_callbacks.ipynb
│   ├── 04_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── trails.ipynb
│
├── src/
│   └── cnnClassifier/
│       ├── components/
│       │   ├── data_ingestion.py
│       │   ├── prepare_base_model.py
│       │   ├── prepare_callbacks.py
│       │   ├── training.py
│       │   └── evaluation.py
│       │
│       ├── config/
│       ├── constants/
│       ├── entity/
│       ├── pipeline/
│       │   ├── predict.py
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_prepare_base_model.py
│       │   ├── stage_03_training.py
│       │   └── stage_04_evaluation.py
│       │
│       └── utils/
│
├── templates/
│   └── index.html
│
├── app.py
├── main.py
├── params.yaml
├── config/config.yaml
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── scores.json
├── setup.py
├── Dockerfile
└── LICENSE
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | Deep Learning |
| CNN | Image classification |
| VGG16 | Transfer learning |
| NumPy | Numerical operations |
| Pandas | Data processing |
| Matplotlib | Visualization |
| Seaborn | Visualization |
| Flask | Prediction API / web application |
| DVC | Data and pipeline versioning |
| Jupyter Notebook | Experimentation |
| PyYAML | Configuration management |
| Python-Box | Configuration handling |
| Git & GitHub | Version control |
| Docker | Containerization |

---

## 🧠 Model

The project uses **VGG16 transfer learning** with weights pretrained on **ImageNet**.

### Model configuration

```yaml
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 16
INCLUDE_TOP: False
EPOCHS: 8
CLASSES: 2
WEIGHTS: imagenet
LEARNING_RATE: 0.01
AUGMENTATION: True
```

The VGG16 classification head is adapted for the two target classes.

### Classes

```text
0 → Coccidiosis
1 → Healthy
```

---

## 🔄 Data Augmentation

During training, image augmentation is used to improve model generalization.

The training pipeline includes:

- Rotation
- Horizontal flipping
- Width shifting
- Height shifting
- Shearing
- Zooming
- Pixel rescaling

Images are resized to:

```text
224 × 224
```

and normalized using:

```python
1./255
```

---

## 📊 Model Evaluation

The model is evaluated using:

- Loss
- Accuracy

The current evaluation result stored in the project is approximately:

```text
Loss:     1.5181
Accuracy: 82.76%
```

The metrics are stored in:

```text
scores.json
```

> Evaluation results can change when the model is retrained with different data, parameters, or TensorFlow versions.

---

## 🔬 Research & Development

The `research` directory contains notebooks used during development:

```text
01_data_ingestion.ipynb
02_prepare_base_model.ipynb
03_prepare_callbacks.ipynb
04_training.ipynb
05_model_evaluation.ipynb
trails.ipynb
```

The experimental workflow is then converted into reusable Python components under:

```text
src/cnnClassifier/
```

---

## 🔁 DVC Pipeline

DVC is used to organize the machine learning workflow into reproducible stages.

The pipeline contains:

```text
Data Ingestion
      ↓
Prepare Base Model
      ↓
Training
      ↓
Evaluation
```

The pipeline definition is available in:

```text
dvc.yaml
```

To reproduce the pipeline after installing DVC:

```bash
dvc repro
```

---

## 🌐 Flask Prediction Application

The project includes a Flask application that allows an image to be submitted for prediction.

### Start the application

```bash
python app.py
```

The Flask application is configured to run on:

```text
http://localhost
```

The application provides:

```text
GET  /
POST /predict
GET/POST /train
```

### Prediction Flow

```text
Input Image
     ↓
Flask API
     ↓
Image Decoding
     ↓
Resize to 224 × 224
     ↓
Trained CNN Model
     ↓
Class Prediction
     ↓
Healthy / Coccidiosis
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/vigneshwarang47/Chicken-disease-Classification.git
```

```bash
cd Chicken-disease-Classification
```

### 2. Create a virtual environment

For Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

For Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Complete Pipeline

To execute the complete training workflow:

```bash
python main.py
```

The following stages are executed:

```text
Stage 1 → Data Ingestion
Stage 2 → Prepare Base Model
Stage 3 → Model Training
Stage 4 → Model Evaluation
```

---

## 🔮 Make a Prediction

After training the model, start the Flask application:

```bash
python app.py
```

Upload a chicken fecal image through the application.

The model returns one of:

```text
Healthy
```

or

```text
Coccidiosis
```

---

## 🔧 Configuration

### Pipeline configuration

```text
config/config.yaml
```

This file contains paths for:

- Data ingestion
- Base model
- Callbacks
- Training model

### Model parameters

```text
params.yaml
```

This file controls:

- Image size
- Batch size
- Number of classes
- Epochs
- ImageNet weights
- Learning rate
- Data augmentation

---

## 📦 DVC

DVC is used for reproducible data and pipeline management.

Useful commands:

```bash
dvc status
```

```bash
dvc repro
```

```bash
dvc dag
```

---

## 🐳 Docker

The repository also contains a `Dockerfile` for containerizing the application.

Build the image:

```bash
docker build -t chicken-disease-classifier .
```

Run the container:

```bash
docker run -p 80:80 chicken-disease-classifier
```

Then access:

```text
http://localhost
```

---

## ✨ Key Features

- End-to-end image classification pipeline
- Transfer learning using VGG16
- Image augmentation
- Modular Deep Learning architecture
- Configuration-driven development
- DVC pipeline management
- Model evaluation
- Flask prediction API
- Reusable prediction pipeline
- Jupyter research notebooks
- Docker support
- Git/GitHub version control

---

## 🚀 Future Improvements

- [ ] Add precision, recall and F1-score
- [ ] Add confusion matrix
- [ ] Add model training visualizations
- [ ] Add automated testing
- [ ] Add CI/CD using GitHub Actions
- [ ] Improve model accuracy through hyperparameter tuning
- [ ] Experiment with EfficientNet / ResNet / MobileNet
- [ ] Add model monitoring
- [ ] Deploy the application to a cloud platform
- [ ] Add a more user-friendly prediction interface

---

## 👨‍💻 Author

### Vigneshwaran G

**Aspiring Data Scientist | Machine Learning | Deep Learning | MLOps**

GitHub:

https://github.com/vigneshwarang47

---

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational and portfolio purposes.
