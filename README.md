# Generalization-oriented Ultrasound Biometry (GU_Biometry) - Baseline Code

Welcome to the Generalization-oriented Ultrasound Biometry Challenge (GU_Biometry) at MICCAI 2026! This repository provides baseline code to help you quickly get started with training, inference, and submission.

## 📋 Table of Contents

- [Competition Overview](#-competition-overview)
- [Competition Tasks](#-competition-tasks)
- [Quick Start](#-quick-start)
- [Code Structure](#-code-structure)
- [Training Model](#-training-model)
- [Local Inference](#-local-inference)
- [Docker Build and Test](#-docker-build-and-test)
- [Important Notes](#-important-notes)
- [FAQ](#-faq)

---

## 🎯 Competition Overview

Accurate and reliable quantification of ultrasound parameters is fundamental to disease diagnosis, longitudinal monitoring, and evidence-based clinical decision-making across prenatal, pediatric, and adult populations. However, current clinical practice still relies heavily on manual or semi-automated measurements, which remain time-consuming, operator-dependent, and prone to substantial inter- and intra-observer variability.

GU_Biometry aims to establish the first large-scale, unified ultrasound biometry benchmark capable of automatically measuring more than 20 key ultrasound parameters spanning prenatal, intrapartum, and routine adult ultrasound.

### Conference & Workshop

- **Conference**: MICCAI 2026
- **Workshops**: 
  - International Workshop on Advances in Simplifying Medical Ultrasound (ASMUS)
  - The 1st MICCAI Workshop on Medical World Models (MWM)

---

## 🏆 Competition Tasks

This challenge addresses **Landmark Detection for Ultrasound Parameter Measurement** across multiple clinical domains:

| Task Domain | Image Count | Description | Key Parameters |
|-------------|-------------|-------------|----------------|
| **Intrapartum Ultrasound** | 36,000 | Labor progression assessment | Angle of Progression (AoP), Head-Symphysis Distance |
| **Cardiac Ultrasound** | 16,000 | Pediatric & adult cardiac function | LV dimensions, Aortic diameter, etc. |
| **Prenatal Ultrasound** | 14,000 | Fetal biometry & growth monitoring | BPD, HC, AC, FL, HL, Cervical Length |

### Target Measurements

**Maternal-Fetal Ultrasound Parameters:**
- Biparietal Diameter (BPD)
- Head Circumference (HC)
- Abdominal Circumference (AC)
- Femur Length (FL)
- Humerus Length (HL)
- Cervical Length

**Labor Progression Ultrasound Parameters:**
- Angle of Progression (AoP)
- Head-Symphysis Distance

**Pediatric and Adult Cardiac Ultrasound Parameters:**
- Parasternal Long-Axis View: Aortic annulus, Ascending aorta, Left atrial diameter, RV wall thickness, Mitral annulus diameter, LV dimensions
- Parasternal Short-Axis View: Main pulmonary artery, RV outflow tract, Pulmonary arteries, Coronary artery
- Apical Four-Chamber View: Atrial/ventricular dimensions, Mitral/tricuspid annulus diameters

---

## 🚀 Quick Start

### 1. Clone the Code

```
https://github.com/lijiake2408/Foundation-Model-for-Ultrasound-Biometry

```

### 2. Prepare Data

Data directory structure:
```
data/
├── train/
│   ├── images/                 # Training images
│   ├── landmarks/              # Landmark annotations (1,000 annotated)
│   └── csv_files/              # CSV index files
│
├── val/                        # Validation set (1,000 images, fully annotated)
│   ├── images/
│   ├── landmarks/
│   └── csv_files/
```

### 3. Train Model

```bash
python train.py
```

Training will automatically:
- Load data (60,000 images with 1,000 annotated)
- Train multi-task landmark detection model
- Save best model as `best_model.pth`

### 4. Local Inference

```bash
python model.py
```

---

## 📁 Code Structure

```
baseline/
├── train.py                    # Training script
├── model.py                    # Core: Inference model
├── model_factory.py            # Multi-task model factory
├── dataset.py                  # Dataset loader
├── utils.py                    # Utility functions
├── best_model.pth             # Trained model weights
└── README.md                  # This file
```

---

## 🎓 Training Model

### Training Script

```bash
python train.py
```

### Main Parameters (can be modified in train.py)

```python
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 50
DATA_ROOT_PATH = '/path/to/train'
MODEL_SAVE_PATH = 'best_model.pth'
```

### Dataset Statistics

| Split | Total Images | Annotated Images | Annotation Rate |
|-------|--------------|------------------|-----------------|
| Training | 60,000 | 1,000 | ~1.7% |
| Validation | 1,000 | 1,000 | 100% |
| Test | 5,000 | 5,000 | 100% |

### Training Output

- **Model weights**: `best_model.pth`
- **Training logs**: Terminal output

---

## 🔮 Local Inference

### Using model.py

Modify the paths in `model.py`:

```python
if __name__ == '__main__':
    data_root = '/path/to/your/data'  # Change to your data path
    output_dir = 'predictions/'
    batch_size = 4
    
    model = Model()
    model.predict(data_root, output_dir, batch_size=batch_size)
```

Then run:

```bash
python model.py
```

### Expected Output Structure

```
predictions/
├── regression_predictions.json     # Regression results
```

---

## 📊 Evaluation Metrics

### Primary Metrics

| Metric | Weight | Description |
|--------|--------|-------------|
| **Mean Radial Error (MRE)** | 50% | Average distance between predicted and ground truth landmarks |
| **Parameter Error** | 50% | Absolute difference between predicted and manually measured parameters |

### MRE Calculation

For each case with K required landmarks:
```
MRE = (1/K) * Σ ||p_k - g_k||_2
```
where `p_k` is the predicted landmark and `g_k` is the ground truth landmark.

### Ranking Method

- Final rankings use a ChallengeR-style rank-then-aggregate framework
- Domain-balanced weighting across four data domains
- Both overall and domain-stratified leaderboards are reported
- Tie-breaking: priority given to method with lower normalized parameter error

---


---

## ⚠️ Important Notes

### 1. About model.py

**`model.py` is the core file!** You can freely modify its internal implementation, but must follow these specifications:

#### Required Interface

```python
class Model:
    def __init__(self):
        """Initialize model"""
        pass
    
    def predict(self, data_root: str, output_dir: str, batch_size: int = 8):
        """
        Perform inference
        
        Args:
            data_root: Input data root directory (/input/ in Docker)
            output_dir: Output directory (/output/ in Docker)
            batch_size: Batch size
        """
        pass
```

#### Output Format Requirements

##### 1) Landmark Predictions Output

**Format**: JSON file

**Path**: `{output_dir}/landmark_predictions.json`

**Content**:
```json
[
  {
    "image_path": "relative/path/image_001.jpg",
    "task_id": "id",
    "predicted_points_normalized": [0.3, 0.4, 0.6, 0.7],
    "predicted_points_pixels": [150, 200, 300, 350]
  },
  ...
]
```

**Description**:
- `predicted_points_normalized`: Normalized landmark coordinates (x1, y1, x2, y2, ...) in range [0, 1]
- `predicted_points_pixels`: Pixel coordinates (x1, y1, x2, y2, ...) in original image space

### 2. Multi-task Model Requirement

**IMPORTANT**: Participants are required to develop a **unified model** capable of handling all subgroups and imaging domains within the challenge. 

- ❌ Subgroup-specific models are NOT permitted
- ❌ Separate submissions for individual tasks are NOT permitted
- ✅ Each team must submit a single integrated solution

### 3. Baseline Comparison Requirement

An official baseline model will be provided by the organizers. Submitted methods must demonstrate **performance improvements over this baseline** to be eligible for ranking and awards.

### 4. Docker Environment Requirements

- **Input mount point**: `/input/` (read-only)
- **Output mount point**: `/output/` (writable)
- **Memory limit**: Recommended not to exceed 16GB

---

## 📊 Data Sources

### Imaging Devices

- Philip-cx50
- Toshiba Aplio300
- Voluson P8
- Esaote Mylab
- Lian-med ObEye
- Youkey Q7

### Data Collection Centers

- Cho Ray Hospital, Vietnam
- Aga Khan University Hospital, Kenya
- University of Barcelona, Spain
- Zhenjiang Maternal and Child Health Hospital, China
- Zhujiang Hospital, Southern Medical University, China
- And 20+ additional centers worldwide

---

## ❓ FAQ

### Q1: How to modify model architecture?

**A**: You can freely modify the model structure in `model_factory.py`, but make sure:
- Input: Ultrasound images (various sizes)
- Output: Landmark coordinates and derived parameters
- The model must handle all clinical domains

### Q2: What if I run out of GPU memory during training?

**A**: Reduce the batch size:
```python
# train.py
BATCH_SIZE = 4  # Change from 8 to 4
```

### Q3: Can I use semi-supervised learning?

**A**: Yes! The training set contains 55,000 unlabeled images specifically designed to support semi-supervised learning strategies. This reflects real-world clinical conditions where expert annotation is limited.

### Q4: How to handle different imaging views?

**A**: Each image is associated with a predefined landmark template specifying which landmarks are applicable. Not all clinical parameters apply to every image.

### Q5: Can I use my own pre-trained model?

**A**: Yes! Model weight files are included in the Docker image. You can use foundation models or pre-trained weights.

### Q6: How to verify if output format is correct?

**A**: 
1. Run Docker on validation set
2. Upload output to CodaLab platform
3. Platform will automatically validate format and return evaluation results

### Q7: What is the clinical significance of the metrics?

**A**: 
- MRE < 5mm is generally considered clinically acceptable for most ultrasound measurements
- Parameter errors within 5-10% of expert measurements are considered good
- These thresholds are based on inter-observer variability in clinical practice

---

## 📄 License

This baseline code is for competition use only.

---

## 🎉 Good Luck with the Competition!

Remember the key steps:
1. ✅ Download data (60,000 training images)
2. ✅ Train unified multi-task model
3. ✅ Test inference locally
4. ✅ Build Docker
5. ✅ **Test Docker on validation set**
6. ✅ **Upload predictions to CodaLab for validation**
7. ✅ Submit Docker image

**Passing the CodaLab evaluation on the validation set ensures your final submission is correct!**

Good luck! 🚀

---

## 📚 References

1. Bai J, et al. Beyond Benchmarks of IUGC: Rethinking Requirements of Deep Learning Method for Intrapartum Ultrasound Biometry from Fetal Ultrasound Videos. Medical Image Analysis, 2026.
2. Bai J, et al. IUGC: A Benchmark of Landmark Detection in End-to-End Intrapartum Ultrasound Biometry. Medical Image Analysis, 2026.
3. Bai J, Zhou Z, Ou Z, et al. PSFHS challenge report: Pubic symphysis and fetal head segmentation from intrapartum ultrasound images. Medical Image Analysis, 2025.
4. Deng, Bo, et al. "Baseline Method of the Foundation Model Challenge for Ultrasound Image Analysis." arXiv preprint arXiv:2602.01055 (2026).

For more information, visit: [BIAS Initiative](https://www.dkfz.de/en/cami/research/topics/biasInitiative.html)
