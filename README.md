# Face Mask Detection using PyTorch

A real-time **Face Mask Detection** system built with **PyTorch**, **Torchvision**, and **OpenCV**. The project uses **Transfer Learning (MobileNetV2)** to classify detected faces as **with mask** or **without mask**.

---

## Features

* Real-time webcam detection
* Face detection using OpenCV Haar Cascade
* Transfer Learning with MobileNetV2
* Binary image classification
* Model evaluation with multiple metrics
* Single image prediction
* Modular and production-style project structure

---

## Demo Pipeline

```mermaid
flowchart TD

A[Webcam / Image]
--> B[Face Detection]

B --> C[Crop Face]

C --> D[Resize 224×224]

D --> E[Normalize]

E --> F[MobileNetV2]

F --> G[Softmax]

G --> H{Prediction}

H -->|Class 0| I[With Mask]

H -->|Class 1| J[Without Mask]
```

---

# Project Structure

```text
Face-Mask-Detection/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── models/
|   ├── haarcascade_frontalface_default.xml
│   └── best_model.pth
│
├── outputs/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── transforms.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── realtime.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── main.py
```

---

# Dataset

Download the **Face Mask Dataset** from Kaggle.

```
dataset/
│
├── with_mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── without_mask/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

---

# Model Architecture

```mermaid
graph LR

A[224×224 RGB Image]
--> B[MobileNetV2 Feature Extractor]

B --> C[Dropout 0.2]

C --> D[Linear Layer]

D --> E[2 Output Classes]

E --> F[With Mask]

E --> G[Without Mask]
```

---

# Training Workflow

```mermaid
flowchart LR

A[Load Dataset]
--> B[Apply Transforms]

B --> C[Create DataLoader]

C --> D[Forward Pass]

D --> E[CrossEntropy Loss]

E --> F[Backpropagation]

F --> G[Adam Optimizer]

G --> H[Validation]

H --> I{Best Accuracy?}

I -->|Yes| J[Save Model]

I -->|No| D
```

---

# Real-Time Inference

```mermaid
sequenceDiagram

participant Camera
participant OpenCV
participant Model
participant Screen

Camera->>OpenCV: Capture Frame

OpenCV->>OpenCV: Detect Face

OpenCV->>Model: Crop + Transform

Model-->>OpenCV: Prediction

OpenCV-->>Screen: Draw Bounding Box
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/maroofiums/Face-Mask-Detection.git

cd Face-Mask-Detection
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate it.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Train the Model

```bash
python src/train.py
```

The best model will be saved in

```text
models/best_model.pth
```

---

# Evaluate the Model

```bash
python src/evaluate.py
```

Metrics include:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

# Predict a Single Image

```bash
python src/predict.py
```

Example output

```text
Prediction : with_mask
Confidence : 99.43%
```

---

# Run Real-Time Detection

```bash
python src/realtime.py
```

Press **Q** to exit.

---

# Technologies

* Python
* PyTorch
* Torchvision
* OpenCV
* Pillow
* NumPy
* Scikit-learn
* Matplotlib
* tqdm

---

# Future Improvements

* MediaPipe Face Detection
* YOLOv8 Face Detection
* EfficientNet
* TensorBoard Logging
* Early Stopping
* Learning Rate Scheduler
* Mixed Precision Training (AMP)
* ONNX Export
* FastAPI Deployment
* Docker Support

---

# Learning Outcomes

This project demonstrates:

* Binary Image Classification
* Transfer Learning
* Computer Vision
* OpenCV
* PyTorch
* Model Evaluation
* Real-Time Inference
* Deep Learning Project Structure

---

# License

This project is licensed under the MIT License.
