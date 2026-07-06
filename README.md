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

The goal of this project is to demonstrate practical, thorough proficiency
with TensorFlow/Keras in an image classification task: from data
exploration, through augmentation, two-phase training (transfer learning +
fine-tuning), all the way to full evaluation with multiple metrics and a
ready-to-use prediction pipeline for new data.

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
> after training and depend on the specific training run and hardware. Once
> you've run the project, it's worth pasting the final numbers here, e.g.:

| Metric | Result |
|---|---|
| Test Accuracy | _fill in after training_ |
| Precision (macro) | _fill in after training_ |
| Recall (macro) | _fill in after training_ |
| F1-score (macro) | _fill in after training_ |

Full results (confusion matrix, per-class classification report) are
automatically saved to the `results/` folder:
- `results/metrics_summary.csv`
- `results/classification_report.csv`
- `results/confusion_matrix.png`

Saved predictions on new images (Section 28) go to the `predictions/` folder.

## 🖼️ Screenshots

> After running the notebook, it's worth saving key visualizations here (as
> files in `images/`), e.g.:
> - example images from the dataset (Section 8),
> - the data augmentation effect (Section 9),
> - accuracy/loss curves (Section 14),
> - the confusion matrix (Section 21),
> - examples of mispredictions (Section 25).
>
> ```markdown
> ![Training curves](images/training_curves.png)
> ![Confusion matrix](images/confusion_matrix.png)
> ```

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/plant-disease-detector.git
cd plant-disease-detector
```

### 2. Virtual environment and dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Kaggle API key (to download the dataset)

The notebook downloads the PlantVillage dataset directly from Kaggle (a much
more reliable source than the version available via `tensorflow_datasets`,
which relies on a Mendeley server that can block cloud IP addresses, e.g.
Google Colab — resulting in a `403 Forbidden` error).

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

### 5. Predict on your own image

After training (or loading a saved model):

```python
from tensorflow import keras
import json

model = keras.models.load_model("models/plant_disease_efficientnetb0_final.keras")
with open("models/class_names.json") as f:
    class_names = json.load(f)

# predict_new_image() function defined in the notebook (Section 28)
predict_new_image("images/my_leaf.jpg", model, class_names)
```

## 📁 Repository Structure

```
plant-disease-detector/
├── notebooks/
│   └── Plant_Disease_Detector_TensorFlow.ipynb   # main notebook (31 sections)
├── models/
│   ├── phase1_feature_extraction.keras           # checkpoint after phase 1 (generated)
│   ├── phase2_fine_tuning.keras                  # checkpoint after phase 2 (generated)
│   ├── plant_disease_efficientnetb0_final.keras  # final model (generated)
│   └── class_names.json                          # list of 38 class names (generated)
├── images/            # example images / README screenshots
├── predictions/        # saved prediction results on new images (generated)
├── results/           # metrics, confusion matrix, classification report (generated)
├── docs/              # additional documentation
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
