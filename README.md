# 🌿 Plant Disease Detector — TensorFlow

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-EfficientNetB0-red?logo=keras)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-brightgreen)

A plant disease classifier based on leaf photographs, built on **Transfer
Learning** and **Fine Tuning** on the **EfficientNetB0** architecture
(TensorFlow / Keras). A project presenting a full, production-style
pipeline: from data, through training, to evaluation and a ready-to-use
prediction function for new images.

---

## 📖 Project Description

The model recognizes **38 classes** — combinations of plant species and a
specific disease (or a healthy state) — from a single leaf photograph, using
the **[PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)**
dataset.

The goal of this project is to demonstrate how TensorFlow/Keras can be used for image classification, covering the entire workflow—from data exploration and augmentation, through two-stage training (transfer learning and fine-tuning), to comprehensive model evaluation using multiple performance metrics and a production-ready prediction pipeline for new data.

## 🧠 Model Architecture

```
Input (224, 224, 3)
        ↓
Data Augmentation (RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomTranslation)
        ↓
EfficientNetB0 (base, ImageNet weights)
   Phase 1: fully frozen (transfer learning)
   Phase 2: last 30 layers unfrozen, low learning rate (fine-tuning)
        ↓
GlobalAveragePooling2D
        ↓
Dropout (0.3)
        ↓
Dense (38, softmax)
```

**Training strategy — two phases:**

| Phase | Base model | Learning rate | Goal |
|---|---|---|---|
| 1. Transfer Learning | frozen | 1e-3 | training the classification head |
| 2. Fine Tuning | last 30 layers unfrozen | 1e-5 | precise adjustment of high-level features |

## 🛠️ Technologies Used

- **TensorFlow 2.x / Keras** — model building and training
- **EfficientNetB0** (`tf.keras.applications`) — transfer learning base
- **Kaggle API** — loading the PlantVillage dataset
- **Data Augmentation** — Keras layers (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`, `RandomTranslation`)
- **Callbacks**: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, `TensorBoard`
- **scikit-learn** — precision, recall, F1-score, confusion matrix, classification report
- **Pandas / Matplotlib / Seaborn** — data analysis and visualization

## 📊 Results

> Metric values are generated automatically in the notebook (Sections 15–22)
> after training and depend on the specific training run and hardware.

| Metric |
|---|
| Test Accuracy |
| Precision (macro) |
| Recall (macro) |
| F1-score (macro) |

Full results (confusion matrix, per-class classification report) are
automatically saved to the `results/` folder:
- `results/metrics_summary.csv`
- `results/classification_report.csv`
- `results/confusion_matrix.png`

Saved predictions on new images (Section 28) go to the `predictions/` folder.

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/AIResearchForge/Plant-Disease-Detector.git
cd Plant-Disease-Detector
```

### 2. Virtual environment and dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Kaggle API key (to download the dataset)

The notebook downloads the PlantVillage dataset directly from Kaggle.

1. Create a free account at [kaggle.com](https://www.kaggle.com).
2. **Account → API → Create New API Token** — this downloads a `kaggle.json`
   file.
3. The notebook will interactively prompt for `KAGGLE_USERNAME` and
   `KAGGLE_KEY` (`getpass`) — you'll find the values in the downloaded
   `kaggle.json` file. They are never hard-coded anywhere.

### 4. Run the notebook

```bash
jupyter notebook notebooks/Plant_Disease_Detector_TensorFlow.ipynb
```

The notebook can also be run directly in **Google Colab** (recommended —
free GPU access): just upload the `.ipynb` file and set
`Runtime → Change runtime type → GPU`.

> ℹ️ On first run, the dataset (a few GB) will be downloaded from Kaggle and
> cached locally — subsequent runs will be instant.

### 5. Predict on your own image(Section 28)

A full "file-to-result" inference pipeline — with automatic saving of the prediction result (image + label + confidence) to the predictions/ folder, simulating how results might be archived in a real application.

How to actually use this function on your own image — three ways:

1. Google Colab (easiest): run the cell below — a "Choose Files" button will appear, letting you pick an image from your computer. It will be automatically uploaded to the notebook and classified.

2. Local Jupyter / any environment: upload your image to the images/ folder in the repository, then in a new cell call:
predict_new_image("images/my_photo.jpg", loaded_model, loaded_class_names)
(replacing the filename with your own).

3. Image already present on the server/Colab disk — simply provide the full path to it as the first argument of the function.

## 📁 Repository Structure

```
Plant-Disease-Detector/
├── notebooks/
│   └── Plant_Disease_Detector_TensorFlow.ipynb
│   └── Plant_Disease_Detector_TensorFlow_example.ipynb
├── models/
├── images/           
├── predictions/       
├── results/          
├── docs/             
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## 🔭 Development Roadmap

- [ ] Fine-tuning on field-condition data (not just laboratory images)
- [ ] Larger EfficientNet variants (B3–B7) or newer architectures (ConvNeXt, ViT)
- [ ] Model explainability (Grad-CAM) — verifying what the model actually
      focuses on
- [ ] Export to TensorFlow Lite / ONNX for mobile devices (an offline app
      for farmers)
- [ ] Deployment as a REST API (FastAPI + Docker)
- [ ] Class weighting / oversampling for less populous classes
- [ ] Multi-label classification (co-occurring diseases, severity assessment)
- [ ] Production model monitoring (e.g. MLflow) — data drift detection

A full description of the model's limitations and possible development
directions can be found in Sections 30–31 of the notebook.

## 📄 License

This project is released under the [MIT](LICENSE) license. The PlantVillage
dataset is subject to the license set by its authors — see
[Kaggle — PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).

## 👤 Author

AIResearchForge — a portfolio project prepared to demonstrate practical proficiency
with TensorFlow/Keras (Transfer Learning, Fine Tuning) in image
classification tasks.
