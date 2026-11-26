# Adaptive Static-Dynamic Layer Fusion for Lightweight Zero-Shot Industrial Anomaly Detection

## Abstract
Zero-shot industrial anomaly detection aims to identify defective patterns in unseen target domains without requiring target-specific anomaly training samples. However, effectively aggregating multi-layer visual representations remains a challenge, as different feature layers may contribute differently to anomaly localization. This paper investigates Adaptive Static-Dynamic Layer Fusion (ALFv2), a layer weighting strategy integrated into the lightweight LiZAD framework. ALFv2 combines learnable static layer weights with input-dependent dynamic weights through a controllable fusion coefficient, while retaining the original visual backbone and vision-language components. The method is evaluated using four cross-dataset settings involving VisA, BTAD, MPDD, and MVTec-AD, with pixel-level Area Under the ROC Curve (AUC-Pixel) as the evaluation metric. Experimental results show that ALFv2 improves the average AUC-Pixel from 90.574\% to 92.382\%, achieving an absolute gain of 1.808 percentage points over the original LiZAD baseline. These results indicate that static-dynamic layer fusion can improve cross-dataset anomaly localization while preserving the original lightweight framework architecture.

## Installation
Create a new Conda environment and install the required dependencies:

```bash
conda create -n LiZAD python=3.12
conda activate LiZAD
pip install -r requirements.txt
```

## Install MobileCLIP2
To install MobileCLIP2, follow the steps below (see the official [MobileCLIP repository](https://github.com/apple/ml-mobileclip) for more details).

### 1. Install MobileCLIP
Clone the MobileCLIP repository and install it:
```bash
git clone https://github.com/apple/ml-mobileclip.git
cd ml-mobileclip
pip install .
```

### 2. Download the MobileCLIP2-S0 Checkpoint
Download the pretrained MobileCLIP2-S0 checkpoint from Hugging Face:
```bash
hf download apple/MobileCLIP2-S0 --local-dir ./pretrained
```

### 3. Prepare the Dataset
We follow the standard dataset preprocessing pipeline. The framework requires a `meta.json` file in the root directory of each dataset.
To generate the required `meta.json` files, please follow the instructions provided in the [AdaCLIP repository](https://github.com/caoyunkang/AdaCLIP).

### 4. Configure Dataset Paths
Update the `base_dir` variable in `datasets/__init__.py` to point to your dataset root directory.

### 5. Training
Use the following command to train the model:

```bash
bash train.sh
```

### 6. Evaluation
Use the following command to evaluate the trained model:

```bash
bash test.sh
```
### 7. Acknowledgement
We gratefully acknowledge the open-source contributions of the [AdaCLIP](https://github.com/caoyunkang/AdaCLIP), [AA-CLIP](https://github.com/Mwxinnn/AA-CLIP), [MobileCLIP](https://github.com/apple/ml-mobileclip), [DINOv3](https://github.com/facebookresearch/dinov3) authors, whose work made this project possible.

