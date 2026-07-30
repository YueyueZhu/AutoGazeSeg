# AutoGazeSeg: Advancing Gaze-Supervised Medical Image Segmentation with Automatic Pseudo-Mask Generation and Calibration

AutoGazeSeg is a gaze-supervised medical image segmentation framework that automatically constructs stable dual pseudo masks and calibrates their ambiguous regions with prototype-guided soft supervision. This anonymous release contains the main-method training and evaluation code for all four datasets used in the paper.

![Motivation and automatic pseudo-mask generation](assets/motivation.png)

## Method

AutoGazeSeg contains three main components.

1. **Automatic Pseudo-Mask Generation (APG).** CRF-refined gaze maps are sampled at multiple response levels and converted into box and point prompts for SAM. A level-stability graph combines candidate quality and inter-mask consistency. Greedy search selects a stable candidate family whose intersection forms \(M_{\mathrm{high}}\), while family voting forms \(M_{\mathrm{low}}\).
2. **Text-guided dual-branch U-Net.** Target-aware text embeddings are injected into two U-Net branches through cross-attention. The high-confidence branch is supervised by \(M_{\mathrm{high}}\), and the low-confidence branch is supervised by \(M_{\mathrm{low}}\).
3. **Prototype-Guided Pseudo-Mask Calibration (PPC).** The dual masks define reliable foreground \(R^+=M_{\mathrm{high}}\), reliable background \(R^-=1-M_{\mathrm{low}}\), and uncertain region \(U=M_{\mathrm{low}}-M_{\mathrm{high}}\). Feature and image-appearance prototypes jointly estimate the foreground probability \(a_m\) for branch \(m\), producing a calibrated soft pseudo mask:

   \[
   S_m = R^+ + U a_m.
   \]

   Cross-branch calibration uses the calibrated mask from one branch to supervise the other:

   \[
   L_{\mathrm{PPC}}
   = L_{\mathrm{wbd}}(P_1,S_2)
   + L_{\mathrm{wbd}}(P_2,S_1),
   \]

   where \(L_{\mathrm{wbd}}\) is weighted BCE-Dice loss. The complete objective is:

   \[
   L_{\mathrm{total}}
   = L_{\mathrm{SUP}}
   + \lambda_p L_{\mathrm{PPC}}
   + \lambda_c L_{\mathrm{COS}}.
   \]

At inference, the foreground predictions from the two branches are averaged.

![AutoGazeSeg framework](assets/framework.png)

![Prototype-Guided Pseudo-Mask Calibration](assets/ppc_calibration.png)

## Repository scope

The repository exposes only the proposed method. Each entry script trains one dataset and then automatically evaluates the best checkpoint:

| Dataset | Entry script |
|---|---|
| Kvasir-SEG | `train_test_kvasir.sh` |
| NCI-ISBI | `train_test_nci.sh` |
| ISIC | `train_test_isic.sh` |
| BraTS 2019 | `train_test_brats2019.sh` |

APG pseudo-mask generation and text-description encoding are separate data-preparation stages. This compact release expects the resulting dual pseudo masks and text embeddings to be prepared before training.

## Installation

Python 3.10 or newer and a CUDA-capable GPU are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepared data

### Paper splits

| Dataset | Training split | Test split | Image type |
|---|---:|---:|---|
| Kvasir-SEG | 900 images | 100 images | Endoscopic polyp images |
| NCI-ISBI | 60 volumes / 789 retained slices | 10 volumes / 117 retained slices | T2-weighted prostate MR images |
| ISIC | 800 images | 200 images | Dermoscopic images |
| BraTS 2019 | 4,495 slices | 878 slices | FLAIR brain MR images |

### Directory layout

The default paths are repository-relative. `train.txt` and `test.txt` contain one sample identifier per line. Only the first whitespace-separated field is read.

```text
data/
└── <dataset>/
    ├── train.txt
    ├── test.txt
    ├── images/
    ├── masks/
    ├── embeddings/
    └── pseudo_masks/
        ├── labelshigh/
        └── labelslow/
```

Training samples require an image, text embedding, high-confidence pseudo mask, and low-confidence pseudo mask. Test samples require an image, text embedding, and ground-truth mask.

| Dataset | Image | Ground truth | \(M_{\mathrm{high}}\) | \(M_{\mathrm{low}}\) | Text embedding |
|---|---|---|---|---|---|
| Kvasir-SEG | `images/<id>.jpg` | `masks/<id>.jpg` | `labelshigh/<id>.jpg` | `labelslow/<id>.jpg` | `embeddings/<id>.npy` |
| NCI-ISBI | `images/<id>.dcm` | `masks/<id>.png` | `labelshigh/<id>.png` | `labelslow/<id>.png` | `embeddings/<id>.npy` |
| ISIC | `images/<id>.jpg` | `masks/<id>.png` | `labelshigh/<id>.jpg` | `labelslow/<id>.jpg` | `embeddings/<id>.npy` |
| BraTS 2019 | `images/<id>.png` | `masks/<id>.png` | `labelshigh/<id>.png` | `labelslow/<id>.png` | `embeddings/<id>.npy` |

The embedding arrays must contain the 2,560-dimensional token representations expected by the model. \(M_{\mathrm{high}}\) must be contained in \(M_{\mathrm{low}}\).

The train/test scripts do not load foundation-model checkpoints. Recreating the prepared inputs requires the SAM ViT-H weights used by APG and the Qwen3-Embedding-4B weights used for text encoding. These weights are not distributed in this repository.

## Training and evaluation

Each invocation processes one random seed. The default command trains AutoGazeSeg and immediately evaluates the saved best checkpoint.

```bash
SEED=0 DEVICE=0 bash train_test_kvasir.sh
SEED=0 DEVICE=0 bash train_test_nci.sh
SEED=0 DEVICE=0 bash train_test_isic.sh
SEED=0 DEVICE=0 bash train_test_brats2019.sh
```

The paper reports three independent runs with seeds `0`, `100`, and `10000`. Launch each seed separately:

```bash
SEED=0 DEVICE=0 bash train_test_kvasir.sh
SEED=100 DEVICE=0 bash train_test_kvasir.sh
SEED=10000 DEVICE=0 bash train_test_kvasir.sh
```

Prepared data may be stored outside the repository:

```bash
DATA_ROOT=data/Kvasir-SEG \
EMBEDDING_ROOT=data/Kvasir-SEG/embeddings \
PSEUDO_HIGH_ROOT=data/Kvasir-SEG/pseudo_masks/labelshigh \
PSEUDO_LOW_ROOT=data/Kvasir-SEG/pseudo_masks/labelslow \
OUTPUT_ROOT=outputs \
SEED=100 \
DEVICE=0 \
bash train_test_kvasir.sh
```

### Evaluation only

Use `EVAL_ONLY=1` and provide a checkpoint:

```bash
EVAL_ONLY=1 \
CKPT_PATH=outputs/checkpoints/AutoGazeSeg_kvasir_seed100.pth \
SEED=100 \
DEVICE=0 \
bash train_test_kvasir.sh
```

Checkpoints produced by the earlier implementation remain directly loadable through `CKPT_PATH`; no state-dictionary conversion is required.

### Environment overrides

| Variable | Purpose | Default |
|---|---|---|
| `PYTHON` | Python executable | `python` |
| `DATA_ROOT` | Dataset directory | `data/<dataset>` |
| `EMBEDDING_ROOT` | Precomputed text embeddings | `<DATA_ROOT>/embeddings` |
| `PSEUDO_HIGH_ROOT` | High-confidence pseudo masks | `<DATA_ROOT>/pseudo_masks/labelshigh` |
| `PSEUDO_LOW_ROOT` | Low-confidence pseudo masks | `<DATA_ROOT>/pseudo_masks/labelslow` |
| `OUTPUT_ROOT` | Common output directory | `outputs` |
| `RESULTS_DIR` | Predictions and metric tables | `<OUTPUT_ROOT>/results` |
| `LOG_DIR` | Training logs | `<OUTPUT_ROOT>/logs` |
| `CHECKPOINT_DIR` | Model checkpoints | `<OUTPUT_ROOT>/checkpoints` |
| `SEED` | Random seed | `0` |
| `DEVICE` | Visible CUDA device index | `0` |
| `BATCH_SIZE` | Training batch size | `8` |
| `SPATIAL_SIZE` | Training and inference size | `224` |
| `DATA_SIZE_RATE` | Fraction of each split used | `1` |
| `MAX_ITE` | Training iterations | `15000` |
| `NUM_WORKERS` | Data-loader workers | `4` |
| `FP16` | Mixed precision (`0` or `1`) | `1` |
| `VAL_STEP` | Validation interval | `100` |
| `LOG_STEP` | Logging interval | `100` |
| `PPC_WEIGHT` | PPC weight \(\lambda_p\) | `1.0` |
| `FEATURE_DISTILLATION_WEIGHT` | Feature-distillation weight \(\lambda_c\) | `0.5` |
| `IMAGE_PROTO_BLEND` | Image-appearance blending weight \(\alpha\) | `0.25` |
| `EVAL_ONLY` | Skip training and evaluate (`0` or `1`) | `0` |
| `CKPT_PATH` | Evaluation checkpoint | Dataset/seed checkpoint under `CHECKPOINT_DIR` |
| `SAVE_NAME` | Result subdirectory | `AutoGazeSeg_<dataset>_seed<SEED>` |

## Reproduction settings

The paper uses a 2D U-Net backbone trained from scratch for 15,000 iterations with batch size 8. The learning rate follows cosine decay from \(1\times10^{-2}\) to \(1\times10^{-4}\). Every result is the mean and standard deviation of three independent runs.

The paper settings are:

- candidate-level interval \(\delta=0.01\), with levels from 0.01 to 0.99;
- graph-density weights \(\lambda_n=0.4\) and \(\lambda_e=0.6\);
- low-confidence family-voting ratio \(\rho=0.50\);
- image-appearance blending weight \(\alpha=0.25\);
- PPC weight \(\lambda_p=1.0\);
- feature-distillation weight \(\lambda_c=0.5\).

## Outputs

```text
outputs/
├── checkpoints/
│   └── AutoGazeSeg_<dataset>_seed<seed>.pth
├── logs/
└── results/
    └── <dataset-and-seed>/
        ├── pred_png/
        ├── label_png/
        ├── per_case_metrics.csv
        └── summary_metrics.csv
```

## Results

Values are mean ± standard deviation over the three independent runs.

### Kvasir-SEG and NCI-ISBI

| Method | Supervision | Kvasir-SEG Dice (%) | NCI-ISBI Dice (%) | AT (hrs) |
|---|---|---:|---:|---:|
| UNet | Full | 82.12 ± 1.11 | 80.58 ± 0.48 | 18.7 |
| nnUNet | Full | 85.37 ± 0.48 | 81.54 ± 0.45 | 18.7 |
| BoxInst | Box | 65.72 ± 2.97 | 73.78 ± 1.15 | 3.1 |
| BoxTeacher | Box | 73.33 ± 1.30 | 75.60 ± 1.15 | 3.1 |
| PointSup | Point | 73.05 ± 1.64 | 73.46 ± 4.71 | 4.8 |
| AGMM | Point | 75.57 ± 0.84 | 73.86 ± 1.26 | 4.8 |
| AGMM | Scribble | 67.23 ± 1.02 | 72.70 ± 1.03 | 2.6 |
| CycleMix | Scribble | 76.43 ± 0.65 | 73.41 ± 1.09 | 2.6 |
| ShapePU | Scribble | 77.26 ± 0.73 | 73.06 ± 1.18 | 2.6 |
| ScribFormer | Scribble | 75.69 ± 0.48 | 74.31 ± 1.29 | 2.6 |
| TransUNet | Gaze | 70.38 ± 0.86 | 75.46 ± 1.20 | 2.2 |
| UNet | Gaze | 73.74 ± 0.94 | 74.75 ± 1.58 | 2.2 |
| nnUNet | Gaze | 74.42 ± 0.92 | 77.20 ± 1.03 | 2.2 |
| GazeMedSeg | Gaze | 77.80 ± 1.02 | 77.64 ± 0.57 | 2.2 |
| GiTNet | Gaze | 76.86 ± 0.23 | 78.98 ± 0.04 | 2.2 |
| FGI | Gaze | 80.78 ± 0.11 | 80.53 ± 0.49 | 2.2 |
| **AutoGazeSeg** | **Gaze** | **86.94 ± 0.09** | **82.31 ± 0.10** | **2.2** |

### ISIC and BraTS 2019

| Method | Supervision | ISIC Dice (%) | ISIC IoU (%) | ISIC HD95 | BraTS 2019 Dice (%) | BraTS 2019 IoU (%) | BraTS 2019 HD95 |
|---|---|---:|---:|---:|---:|---:|---:|
| UNet | Full | 88.17 ± 0.10 | 80.97 ± 0.05 | 7.96 ± 0.34 | 91.52 ± 0.28 | 84.89 ± 0.43 | 8.48 ± 0.54 |
| nnUNet | Full | 88.96 ± 0.18 | 81.98 ± 0.29 | 7.70 ± 0.20 | 91.85 ± 0.06 | 85.39 ± 0.10 | 8.99 ± 0.14 |
| UNet | Gaze | 82.95 ± 0.45 | 73.28 ± 0.61 | 11.65 ± 0.37 | 68.42 ± 0.40 | 52.86 ± 0.70 | 21.87 ± 0.14 |
| TransUNet | Gaze | 80.05 ± 0.67 | 69.61 ± 0.78 | 13.90 ± 0.27 | 67.56 ± 0.10 | 51.48 ± 0.18 | 21.50 ± 0.37 |
| nnUNet | Gaze | 82.27 ± 0.50 | 72.38 ± 0.70 | 11.87 ± 0.14 | 67.24 ± 0.37 | 51.19 ± 0.40 | 22.23 ± 0.51 |
| GiTNet | Gaze | 77.86 ± 0.29 | 66.49 ± 0.42 | 12.03 ± 0.13 | 77.84 ± 0.16 | 65.06 ± 0.25 | 17.61 ± 0.48 |
| GazeMedSeg | Gaze | 82.91 ± 0.02 | 73.38 ± 0.07 | 10.35 ± 0.47 | 79.96 ± 0.68 | 67.82 ± 0.86 | 14.49 ± 0.56 |
| FGI | Gaze | 84.22 ± 0.17 | 75.10 ± 0.21 | 10.76 ± 0.44 | 77.98 ± 2.26 | 64.95 ± 3.08 | 20.02 ± 0.79 |
| **AutoGazeSeg** | **Gaze** | **87.59 ± 0.51** | **79.72 ± 0.76** | **8.14 ± 0.28** | **86.17 ± 0.26** | **76.83 ± 0.44** | **12.10 ± 0.54** |


### Qualitative comparison

![Qualitative segmentation comparison](assets/qualitative.png)
