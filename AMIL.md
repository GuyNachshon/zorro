
# Attention-MIL Classifier for Malicious Packages

---

## 1. Problem Statement

We need to detect malicious software packages (npm, PyPI, Rust crates) using only **package-level labels** (benign or malicious).
Challenge:

* For malicious packages, we **don’t know which file/function is malicious**.
* Trojanized packages may contain 99% benign code and 1% malicious payload.

Goal: Build a **lightweight classifier** that:

* Operates at package level.
* Surfaces suspicious units (files/functions) automatically.
* Runs fast enough for **CI/CD and registry scanning**.

---

## 2. Goals

* High recall of malicious packages (≥95%).
* Low false positive rate (<2%).
* Instance-level explanations: highlight the top 1–3 files most likely malicious.
* Training scalable to tens of thousands of packages.
* Inference latency ≤ 2s/package (average).

---

## 3. Non-Goals

* Dynamic sandboxing of packages.
* Full semantic verification of code correctness.
* Detecting non-code-based threats (e.g., typosquatting by package name).

---

## 4. Approach

We treat each package as a **bag of instances** (files/functions).

* Only the bag (package) has a label.
* Attention-MIL allows us to learn **instance-level importance** from bag-level supervision.
* Final output:

  * Malicious probability (package).
  * Attention weights (instances).

---

## 5. Inputs & Features

**Unit = file or function. Each unit gets:**

* **Code embedding**:

  * Encode source tokens with **GraphCodeBERT** (preferred) or CodeBERT.
  * Extract \[CLS] vector (hidden dim 768).

* **API/event features:**

  * Counts of `net`, `fs`, `crypto`, `subprocess`, `eval`, `env` calls.
  * Normalize counts (log scaling).

* **Entropy features:**

  * Shannon entropy of string literals.
  * Flag high-entropy blobs (base64, obfuscation).

* **Phase tags:**

  * Install-time (`setup.py`, `postinstall`).
  * Runtime.
  * Test/dev.

* **Metadata:**

  * File size.
  * Number of imports.
  * Number of functions.

---

## 6. Architecture

**Stage 1: Local encoding**

* Each unit → vector of size 768 + appended feature vector (API counts, entropy, phase).
* Concatenate and project to 512-dim embedding.

**Stage 2: Attention-MIL pooling**

* Package = bag of embeddings.
* Attention mechanism computes weight αᵢ per unit.
* Weighted sum → package embedding.

**Stage 3: Classifier**

* Package embedding → 2-layer MLP (hidden 512 → 128 → 1).
* Output = malicious probability.

**Outputs for analysts:**

* Malicious probability.
* Top-3 files/functions (by αᵢ).
* Reasons: risky API features of top units.

---

## 7. Training

* **Loss function:**

  * Binary cross-entropy (package label).
  * * Attention sparsity regularizer (encourages few strong units).
  * * Counterfactual loss: mask top-attention unit, require score to drop.

* **Curriculum:**

  * Stage A: balanced benign/malicious batches (5:1).
  * Stage B: add obfuscated variants (minified, base64, string-split).
  * Stage C: scale ratio to 10:1 for real-world calibration.

* **Batching:**

  * Max 32 packages/GPU.
  * Cap 100 units per package (sample top risky units if >100).

---

## 8. Evaluation

* **Classification metrics:**

  * ROC-AUC ≥ 0.95.
  * PR-AUC.
  * FPR at 95% TPR < 2%.

* **Localization metrics (synthetic trojans):**

  * IoU ≥ 0.7 between attended units and injected malicious units.
  * Counterfactual consistency: removing top attended file drops score ≥ 0.5.

* **Generalization tests:**

  * Hold out malware families for zero-day testing.
  * Test obfuscation robustness.

---

## 9. Risks & Mitigations

* **Attention instability:** model may spread weights over benign files.

  * Mitigation: add sparsity + counterfactual loss.

* **Overfitting to API keywords:** may misclassify benign tools using `subprocess` or `eval`.

  * Mitigation: combine embeddings + API context, not raw counts.

* **Large packages:** hundreds of files slow down inference.

  * Mitigation: file-level pooling → sample risky subsets.

---

## 10. Success Criteria

* High detection (ROC-AUC ≥ 0.95).
* Robust to obfuscation.
* Provides actionable explanations (top-3 files correct ≥70% of time on synthetic trojans).
* Fast inference (≤2s/package).

---

## 11. Roadmap

* **Week 1–2:** Parser pipeline (tokenization + feature extraction).
* **Week 3–4:** Local encoder + MIL pooling baseline.
* **Week 5–6:** Add attention sparsity + counterfactual loss.
* **Week 7–8:** Augmentation (obfuscation), eval on hold-out families.