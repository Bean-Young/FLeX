# FLeX

FLeX is a clean implementation of lesion-first, frequency-aware test-time adaptation for breast ultrasound segmentation and classification.

The released code keeps only the core method:

- Lesion-first hierarchical prediction.
- Low/mid/high frequency prompt adaptation.
- Source-preserving residual fusion.
- Lesion-preserving unsupervised objective with `L_sub`, `L_src`, `L_pres`, and `L_neg`.

Historical exploratory branches such as PF-MoE, BDM routing, DAS-lite, topology-safe sweeps, domain-prior calibration, threshold tuning, and morphology-only post-processing are intentionally not included in this clean release.

## Method

Given a source model outputting segmentation logits `s` and classification logits `a`, FLeX first estimates lesion existence from the segmentation mask:

```text
r = mean_topk(sigmoid(s))
```

The three-class probability is then factorized as:

```text
P(Normal | x)    = 1 - r
P(Benign | x)    = r * q_benign
P(Malignant | x) = r * q_malignant
```

where `[q_benign, q_malignant] = softmax([a_benign, a_malignant])`.

For test-time adaptation, FLeX applies learnable amplitude prompts in low, mid, and high Fourier bands:

```text
x' = IFFT(A(x) * exp(M_low P_low + M_mid P_mid + M_high P_high), phase(x))
```

The prompt is optimized with:

```text
L = lambda_sub  L_sub
  + lambda_src  L_src
  + lambda_pres L_pres
  + lambda_neg  L_neg
  + lambda_p    ||P||^2
```

where:

- `L_sub` minimizes conditional benign/malignant subtype entropy.
- `L_src` keeps the adapted segmentation close to the source prediction.
- `L_pres` preserves source-supported lesion foreground.
- `L_neg` suppresses foreground expansion in source-supported background.

The final prediction is a conservative residual correction around the source model:

```text
p_seg = p_source + rho_seg * (p_adapt - p_source)
p_cls = p_source_cls + rho_cls * (p_adapt_cls - p_source_cls)
```

## Installation

```bash
git clone https://github.com/Bean-Young/FLeX.git
cd FLeX
pip install -r requirements.txt
```

## Data format

Use a CSV manifest with three columns:

```csv
image,mask,label
images/case001.png,masks/case001.png,0
images/case002.png,masks/case002.png,1
```

Class labels follow:

- `0`: Benign
- `1`: Malignant
- `2`: Normal

## Running inference

FLeX is model-agnostic. Your model factory must return a PyTorch module whose forward method returns:

```python
seg_logits, cls_logits = model(images)
```

Example:

```bash
python tools/run_flex.py \
  --manifest data/target.csv \
  --root data \
  --model-factory flex.models:build_unet \
  --checkpoint checkpoints/source_unet.pth \
  --output results/flex_target.json \
  --image-size 224 \
  --batch-size 8
```

For source-model evaluation without test-time adaptation:

```bash
python tools/run_flex.py \
  --manifest data/target.csv \
  --root data \
  --model-factory flex.models:build_unet \
  --checkpoint checkpoints/source_unet.pth \
  --output results/source_noadapt.json \
  --no-adapt
```

## Repository layout

```text
flex/
  data.py        CSV ultrasound dataset
  frequency.py   Low/mid/high Fourier prompt
  method.py      FLeX adapter and objective
  metrics.py     Segmentation and classification metrics
  training.py    Minimal segmentation/classification wrapper utilities
tools/
  run_flex.py    Model-agnostic evaluation entry point
configs/
  flex.yaml      Default hyperparameters
```

## Citation

The paper citation will be added after release.
