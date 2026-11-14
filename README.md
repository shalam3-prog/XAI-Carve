# SIFT-XAI (XAI-Carve)

This repository contains the source code and benchmark dataset for **SIFT-XAI**, an explainable file-fragment classification framework. SIFT-XAI processes raw 512-byte file fragments, trains a Deep Neural Funnel classifier alongside baseline models, and generates byte-level post-hoc explanations using Kernel SHAP.

More details on the methodology and experimental evaluation are available in the accompanying paper, **XAI-Carve**.

---

## Repository Contents

* **`SIFT-XAI.py`** — The main executable script that orchestrates corpus serialization, TF-IDF feature engineering, model training, evaluation, and SHAP explanation generation.
* **`Classify.py`** — Implementation of the Deep Neural Funnel Architecture (`DNN`), the ablation model without feature engineering (`DNN-W`), classical machine learning baselines, and the Kernel SHAP explainability module.
* **`sift_512.csv`** — The extracted benchmark dataset (derived from the Govdocs1 corpus) containing 47,482 raw 512-byte file fragments across 20 file types.

---

## Usage

Run the main pipeline using the following command:

```bash
python SIFT-XAI.py -f sift_512.csv
```

### Generated Outputs

Executing the script produces the following result files:

1. **SHAP Explanation Plots (PDF):** Class-specific waterfall explanations for True Positive (`TP`), False Positive (`FP`), and False Negative (`FN`) predictions across all 20 file types (e.g., for `csv`):
   * `XAI_TP_Sample_XX_csv.pdf`
   * `XAI_FP_Sample_XX_csv.pdf`
   * `XAI_FN_Sample_XX_csv.pdf`
   *(where `XX` denotes the selected sample index)*
2. **Baseline Evaluation Reports:** Classification reports and performance metrics for each conventional machine learning baseline.
3. **Neural Network Reports:** Evaluation metrics and confusion matrices for the proposed `DNN` and the `DNN-W` ablation variant.

---

## Running Component-Wise Ablation Studies

To reproduce the four internal architectural ablation experiments reported in the paper, toggle the following boolean flags one at a time in the configuration section of the code by setting them to `False`:

* `__DO__ = False` — Sets the Dropout rate to `0.0` (removes Dropout regularization).
* `__SWISH__ = False` — Replaces the Swish activation function with a linear activation.
* `__BN__ = False` — Removes the Batch Normalization layers.
* `__ADAMW__ = False` — Replaces the AdamW optimizer with standard SGD.
