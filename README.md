# FLeX

Official clean implementation of:

**Frequency-aware Lesion Experts for Test-Time Adaptation in Breast Ultrasound Analysis**

FLeX is a source-free test-time adaptation (TTA) and continual test-time adaptation (CTTA) framework for joint breast ultrasound lesion segmentation and image-level diagnosis. The method treats lesion evidence as the unit of adaptation: it estimates lesion presence from the segmentation map, performs benign/malignant discrimination only within the lesion-present subspace, adapts target appearance with low/mid/high frequency lesion prompts, and forms the final prediction through source-preserving fusion.

## Method overview

FLeX follows the four-stage pipeline described in the paper.

### Stage 1: Source model

A source-trained network `f_theta` outputs segmentation logits and diagnostic logits:

```text
(z(x), a(x)) = f_theta(x)
p(x) = sigmoid(z(x))
q(x) = softmax(a(x))
```

During target deployment, source images are unavailable and the source network is frozen.

### Stage 2: Lesion-guided diagnostic hierarchy

FLeX separates lesion existence from lesion type. For a foreground probability map `p`, an adaptive high-confidence support is built as:

```text
eta(p) = clip(mu(p) + kappa * sigma(p), eta_min, eta_max)
S(p)   = {u | p_u >= eta(p)}
```

The lesion evidence score is:

```text
s(p) = mean_{u in S(p)} p_u
```

If `S(p)` is empty, `s(p)` is set to `max_u p_u`. The paper uses:

```text
kappa = 1.0, eta_min = 0.50, eta_max = 0.95
```

Diagnostic probabilities are then factorized as:

```text
[q_B, q_M] = softmax([a_B, a_M])
P_N = 1 - s(p)
P_B = s(p) * q_B
P_M = s(p) * q_M
```

Thus, Normal means absence of reliable lesion evidence, while Benign and Malignant are compared only inside the lesion-present probability mass.

### Stage 3: Frequency-aware lesion prompts

FLeX decomposes the target image into low, mid, and high Fourier bands:

```text
M_low  = 1[nu < alpha_1]
M_mid  = 1[alpha_1 <= nu < alpha_2]
M_high = 1[nu >= alpha_2]
```

Each band is transformed back to image space and receives a bounded additive prompt:

```text
x_b = Re(IFFT(M_b * FFT(x)))
Prompt(x_b, P_b) = x_b + xi * tanh(P_b)
```

The lesion-frequency representation is:

```text
phi_lesion(x) =
  omega_low  * Prompt(x_low,  P_low)
+ omega_mid  * Prompt(x_mid,  P_mid)
+ omega_high * Prompt(x_high, P_high)
```

The main configuration uses:

```text
omega_low = 0.25, omega_mid = 0.50, omega_high = 0.25
```

The mid-frequency component is emphasized because it captures lesion-background transition and boundary morphology, while low and high frequencies preserve coarse tissue context and fine texture.

### Stage 4: Constrained prompt adaptation and source-preserving fusion

Only the frequency prompts are updated on unlabeled target images. The source network remains frozen.

The source and adapted predictions are:

```text
(z_0, a_0) = f_theta(x)
(z_t, a_t) = f_theta(phi_lesion(x))
p_0 = sigmoid(z_0)
p_t = sigmoid(z_t)
```

FLeX uses the source output to define a detached lesion support weight:

```text
alpha_0 = stopgrad(s(p_0) * max(1 - q_{0,N}, delta))
```

The prompt objective is:

```text
L_tta =
  L_sub(a_t)
+ lambda_s L_src(p_t, p_0)
+ lambda_p L_pres(p_t, p_0)
+ lambda_n L_neg(p_t, p_0)
```

where:

```text
L_sub  = - alpha_0 * sum_{c in {B,M}} q_{t,c} log(q_{t,c})
L_src  = mean(|p_t - p_0|) + beta_A * |area(p_t) - area(p_0)|
L_pres = alpha_0 / (|S_0| + eps) * sum_{u in S_0} [p_{0,u} - p_{t,u}]_+
L_neg  = 1 / (|B_0| + eps) * sum_{u in B_0} [p_{t,u} - eta(p_0)]_+
```

`S_0 = S(p_0)` is the source-supported lesion region, and `B_0 = {u | p_{0,u} < eta(p_0)}` is the source-supported background region. The support sets are detached during optimization.

After prompt adaptation, FLeX uses source-preserving prediction fusion:

```text
p_final = (1 - rho_seg) * p_0 + rho_seg * p_t
P_final = (1 - rho_cls) * P_0 + rho_cls * P_t
```

The binary lesion mask and image-level label are:

```text
y_hat = 1[p_final > tau]
r_hat = argmax(P_final)
```

The paper uses `tau = 0.40`, selected on source validation data and fixed for all target domains.

### Output-level safeguards

The paper also uses two output-level safeguards:

- **Category prior calibration**, which adjusts diagnostic probabilities using source-validation-selected priors without changing the mask.
- **Morphology-constrained mask refinement**, which removes isolated foreground components that are unlikely to represent coherent lesions.

Both safeguards are implemented and configurable in this repository.

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

Class labels follow the released code convention:

- `0`: Benign
- `1`: Malignant
- `2`: Normal

## Running inference

FLeX is model-agnostic. The model factory must return a PyTorch module whose forward method returns:

```python
seg_logits, cls_logits = model(images)
```

Example with the included small UNet factory:

```bash
python tools/run_flex.py \
  --manifest data/target.csv \
  --root data \
  --model-factory flex.models:build_unet \
  --checkpoint checkpoints/source_unet.pth \
  --output results/flex_target.json \
  --image-size 224 \
  --batch-size 8 \
  --threshold 0.40
```

Optional category prior calibration:

```bash
python tools/run_flex.py \
  --manifest data/target.csv \
  --root data \
  --model-factory flex.models:build_unet \
  --checkpoint checkpoints/source_unet.pth \
  --output results/flex_target.json \
  --class-prior 0.40,0.30,0.30
```

For frozen source-model evaluation without prompt adaptation:

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
  frequency.py   Frequency-band decomposition and lesion prompts
  method.py      FLeX hierarchy, objective, fusion, and safeguards
  metrics.py     Segmentation and classification metrics
  models.py      Minimal UNet factory
  training.py    Minimal segmentation-classification wrapper utilities
tools/
  run_flex.py    Model-agnostic evaluation entry point
configs/
  flex.yaml      Default paper-aligned hyperparameters
```

## Citation

The citation will be added after publication.
