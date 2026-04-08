# Foundation Model for Ultrasound Biometry - Baseline Code
Welcome to the Foundation Model Challenge for Ultrasound Image Analysis (FMC_UIA)! This repository provides baseline code to help you quickly get started with training, inference, and submission.

Get the train dataset through the following channels:

 Baiduyunpan: https://pan.baidu.com/s/12Ra7s-3R4zC5CTVV9N7Hzg?pwd=cjwu 
 
 OneDrive: https://1drv.ms/u/c/cbab6c8ddcdb5d82/IQDtKN9FZ-98RrjXpkLmEioPAXHEvL3htBFyrIA9K1TVQRM?e=zAeVhL
 
 GoogleDrive: https://drive.google.com/file/d/1SivTzkK6IVLH44S_2B5xSR00j96jkJ4-/view?usp=sharing

New Dataset Task Released – Update Your Training Pipeline
  Download: Please download the new dataset files from the following link: https://pan.baidu.com/s/1NNCZLkr8V0Txso669evXnw?pwd=1eqp Or https://drive.google.com/file/d/18oU6FraMa3ybs_XmQhDNrZESRorbGqaG/view?usp=sharing



## 📋 Table of Contents

- [Competition Tasks](#-competition-tasks)
- [Quick Start](#-quick-start)
- [Code Structure](#-code-structure)
- [Training Model](#-training-model)
- [Local Inference](#-local-inference)
- [Docker Build and Test](#-docker-build-and-test)
- [Important Notes](#-important-notes)
- [FAQ](#-faq)

---

## 🎯 Competition Tasks

This challenge includes **4 types of medical image analysis tasks** with a total of **27 subtasks**:

| **Regression** | 27 | Keypoint localization | JSON file |

---


## 🚀 Quick Start

### 1. Clone the Code

### 2. Prepare Data

Data directory structure:
```
data/
├── train/
│   ├── csv_files/              # CSV index files
│   │   ├── task1.csv
│   │   ├── task2.csv
│   │   └── ...
│   └── Regression/            # Regression task images
│
└── val/                       # Validation set (same structure as above)
    ├── csv_files/
    └── Regression/
```

### 3. Train Model

```bash
python train.py
```

Training will automatically:
- Load data
- Train multi-task model
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
├── model.py                    # ⭐ Core: Inference model
├── model_factory.py            # Multi-task model factory
├── dataset.py                  # Dataset loader
├── utils.py                    # Utility functions
├── best_model.pth             # Trained model weights
├── docker/                    # ⭐ Docker submission related
│   ├── Dockerfile
│   ├── model.py              # Docker version of inference code
│   ├── model_factory.py
│   ├── requirements.txt
│   ├── build.sh              # Build script
│   ├── run_test.sh           # Test script
│   ├── README.md             # Docker detailed documentation
│   └── QUICKSTART.md         # Quick guide
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
BATCH_SIZE = 8
NUM_EPOCHS = 50
DATA_ROOT_PATH = '/path/to/train'
MODEL_SAVE_PATH = 'best_model.pth'
```

### Training Output

- **Model weights**: `best_model.pth`
- **Training logs**: Terminal output

---

## 🔮 Local Inference

###  Using model.py

Modify the paths in `model.py`:

```python
if __name__ == '__main__':
    data_root = '/path/to/your/data'  # Change to your data path
    output_dir = 'predictions/'
    batch_size = 8
    
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
── regression_predictions.json         # Regression results
   
```
**Segmentation result path**: Must maintain the same relative path as the `mask_path` field in CSV

Example:
```
mask_path in CSV: ../Segmentation/Two/dataset_name/MASKS/mask_001.png

Output path: {output_dir}/Segmentation/Two/dataset_name/MASKS/mask_001.png
```

## 🐳 Docker Build and Test

### ⚠️ Important Reminder

**Docker is the final submission method!** Please make sure to complete the following steps before submission:

### Step 1: Prepare Docker Files

Copy the following files to the `docker/` directory:

```bash
cd docker/
# Required files:
# - model.py (modified inference code)
# - model_factory.py
# - best_model.pth (trained model)
# - requirements.txt
# - Dockerfile
```

### Step 2: Build Docker Image

```bash
cd docker/
./build.sh
```

Or build manually:

```bash
docker build -f Dockerfile -t my-submission:latest .
```

### Step 3: Test on Validation Set ⭐

**This step is very important!** Before submission, you must test the Docker on the validation set:

```bash
# Method 1: Use test script
./run_test.sh /path/to/validation /path/to/output

# Method 2: Manual run
docker run --gpus all --rm \
  -v /path/to/validation:/input/:ro \
  -v /path/to/output:/output \
  my-submission:latest
```

### Step 4: Validate Output Format

Check the output directory:

```bash
ls -lh /path/to/output/

# Should contain:
# - classification_predictions.json
# - detection_predictions.json
# - regression_predictions.json
# - Segmentation/ (directory)
```

### Step 5: Upload to Codabench for Evaluation

1. Package output files:
   ```bash
   cd /path/to/output
   zip -r predictions.zip .
   ```

2. Log in to Codabench platform
3. Upload `predictions.zip` for validation set evaluation
4. View evaluation results

**If evaluation passes, it means Docker is built correctly!** You can proceed to the next step.

### Step 6: Submit Docker Image

#### Method A: Docker Hub (Recommended)

```bash
docker login
docker tag my-submission:latest YOUR_USERNAME/my-submission:latest
docker push YOUR_USERNAME/my-submission:latest

# Send the image address to the organizing committee:
# YOUR_USERNAME/my-submission:latest
```

#### Method B: Save as File

```bash
docker save -o my-submission.tar my-submission:latest

# Upload my-submission.tar to cloud storage
# Send the link to the organizing committee
```

---

## ⚠️ Important Notes

### 1. About model.py ⭐

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

#### Output Format Requirements 🔴 Important!

##### 1) Regression Task Output

**Format**: JSON file

**Path**: `{output_dir}/regression_predictions.json`

**Content**:
```json
[
  {
    "image_path": "relative/path/image_001.jpg",
    "task_id": "FUGC",
    "predicted_points_normalized": [0.3, 0.4, 0.6, 0.7],
    "predicted_points_pixels": [150, 200, 300, 350]
  },
  ...
]
```

**Description**:
- `predicted_points_normalized`: [x1, y1, x2, y2, ...] (normalized coordinates)
- `predicted_points_pixels`: [x1, y1, x2, y2, ...] (pixel coordinates)

### 2. Docker Environment Requirements

- **Input mount point**: `/input/` (read-only)
- **Output mount point**: `/output/` (writable)
- **Memory limit**: Recommended not to exceed 16GB



## ❓ FAQ

### Q1: How to modify model architecture?

**A**: You can freely modify the model structure in `model_factory.py`, but make sure:
- Input: RGB images
- Output: Format that meets each task's requirements

### Q2: What if I run out of GPU memory during training?

**A**: Reduce the batch size:
```python
# train.py
BATCH_SIZE = 4  # Change from 8 to 4
```

### Q3: Docker build is very slow?

**A**: 
- First build needs to download base image (~6GB), which takes time
- Subsequent builds will use cache and be faster
- Ensure stable network connection

### Q4: How to verify if output format is correct?

**A**: 
1. Run Docker on validation set
2. Upload output to Codabench platform
3. Platform will automatically validate format and return evaluation results
4. If format is incorrect, there will be clear error messages

### Q5: Can I use my own pre-trained model?

**A**: Yes!
- Model weight files are included in the Docker image

### Q6: Must the inference output path be strictly followed?

**A**: **Yes!** Especially for segmentation task mask paths, they must be completely consistent with the `mask_path` field in CSV (removing the leading `../`). Otherwise, the evaluation platform cannot find the files.

### Q7: How to debug Docker internal issues?

**A**: Enter container for debugging:
```bash
docker run --gpus all --rm \
  -v /path/to/data:/input/:ro \
  -v /path/to/output:/output \
  -it my-submission:latest /bin/bash

# Run manually inside container
python model.py
```


## 📄 License

This baseline code is for competition use only.

---

## 🎉 Good Luck with the Competition!

Remember the key steps:
1. ✅ Train model
2. ✅ Test inference locally
3. ✅ Build Docker
4. ✅ **Test Docker on validation set**
5. ✅ **Upload predictions to Codabench for validation**
6. ✅ Submit Docker image

**Passing the Codabench evaluation on the validation set ensures your final submission is correct!**

Good luck! 🚀

