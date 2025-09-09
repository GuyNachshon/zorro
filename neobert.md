# 📄 PRD: NeoBERT Classifier for Malicious Package Detection

---

## 1. Problem Statement

Malicious software packages often hide in ecosystems like **npm** and **PyPI**. They can be:

* **Trojanized benign libraries** (a hidden payload in an otherwise safe package).
* **Malicious-by-design libraries** (everything in the package is malicious).

Our dataset contains **package-level labels** (benign vs malicious), but no per-function labels.
We want a **scalable, modern classifier** using **NeoBERT** as backbone.

---

## 2. Goals

* Build a classifier using NeoBERT that detects malicious packages with high accuracy.
* Package-level labels only → system must work without per-function annotations.
* Support **large-scale training** (tens of thousands of packages).
* Support **fast inference** (< 2–5s/package).
* Provide optional **unit-level explanations** if attention pooling is used.

---

## 3. Non-Goals

* Not intended to replace ICN (no explicit divergence/plausibility modeling).
* Not designed for ultra-small footprint (mobile clients).
* Not a dynamic sandbox or runtime analysis tool.

---

## 4. Inputs

### Unit definition

* Package is split into **units**: files (preferred) or functions (if feasible).

### Input features per unit

* **NeoBERT embeddings:**

  * Tokenize source (truncate to 512 tokens).
  * Feed into NeoBERT → take \[CLS] embedding.

* **Augmented features (optional):**

  * Risky API counts: net, fs, crypto, subprocess, eval, env.
  * Entropy of string literals.
  * Phase tags (install/runtime/test).
  * File size / number of imports.

---

## 5. Architecture

**Stage 1: Local encoding**

* Each unit → NeoBERT encoder → embedding (768–1024d).
* Optionally concatenate engineered features → project to 512d.

**Stage 2: Package-level pooling**

* Aggregate embeddings across units:

  * **Option A (baseline):** mean pooling.
  * **Option B (better):** attention pooling (learned weights per unit).
  * **Option C:** MIL pooling for instance-level suspicion scores.

**Stage 3: Classifier**

* Package embedding → 2-layer MLP (hidden 512 → 128 → 1).
* Output: malicious probability.

---

## 6. Training

### Dataset strategy

* Start: 5:1 benign\:malicious ratio.
* Scale: 10:1 ratio for production.
* Mix: 50% popular packages, 50% long tail.

### Loss functions

* **Binary cross-entropy:** package-level malicious/benign.
* **Auxiliary (optional):**

  * Predict which risky APIs exist in package.
  * Predict phase distribution (install/runtime balance).

### Curriculum

1. **Stage A:** Small balanced dataset (5k benign, 1k malicious).
2. **Stage B:** Expand to 20–30k benign, 3–5k malicious.
3. **Stage C:** Robustness stage with augmented samples (minified, obfuscated, base64).

### Optimization

* Use AdamW with learning rate warmup.
* Train for 3–5 epochs per stage.
* Batch size = 32 packages (sample max 100 units each).

---

## 7. Evaluation

### Metrics

* **Detection:** ROC-AUC ≥ 0.95, PR-AUC, FPR < 2% at 95% TPR.
* **Generalization:** family hold-out (zero-day).
* **Robustness:** ≤5% performance drop on obfuscated samples.
* **Efficiency:** inference latency ≤ 2s small packages, ≤ 5s large packages.

### Optional (if MIL used)

* **Localization:** IoU ≥ 0.7 between high-attention units and known injected malicious files (synthetic trojans).

---

## 8. Risks & Mitigations

* **Token limit (512):** large files truncated → may miss payload.

  * Mitigation: split files into multiple chunks.

* **Overfitting to API usage:** model might misclassify benign utilities (e.g., `subprocess` in dev tools).

  * Mitigation: include hard negatives (benign packages with risky APIs).

* **Pooling too simplistic (mean pooling).**

  * Mitigation: switch to attention pooling or MIL for better sensitivity.

* **Data imbalance (real-world \~1000:1 benign\:malicious).**

  * Mitigation: cost-sensitive loss or threshold calibration after training.

---

## 9. Success Criteria

* **Primary:**

  * ROC-AUC ≥ 0.95 on validation.
  * FPR < 2% at 95% TPR.

* **Secondary:**

  * Robustness to obfuscation.
  * Zero-day generalization on unseen malware families.
  * Inference fast enough for CI/CD or registry scanning.

---

## 10. Roadmap

**Week 1–2:**

* Data pipeline (unitization, tokenization, feature extraction).
* NeoBERT embeddings baseline.

**Week 3–4:**

* Mean pooling classifier baseline.
* Train on small dataset (5k/1k).

**Week 5–6:**

* Add attention pooling / MIL pooling.
* Expand dataset to 20–30k benign, 3–5k malicious.

**Week 7–8:**

* Robustness training with obfuscation.
* Evaluate zero-day generalization.

**Week 9:**

* Threshold calibration for deployment.
* Analyst report generator (malicious probability + suspicious files if MIL).

---

## 11. Why NeoBERT

* **Stronger encoder:** better pretraining and optimization than classic BERT/CodeBERT.
* **Lighter than LLMs:** faster inference than Gemma or larger models.
* **Flexible:** can work with simple pooling or MIL.
* **Ideal baseline:** clean supervised classifier to benchmark against ICN.