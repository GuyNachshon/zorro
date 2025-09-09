# 📄 Product Requirements & Architecture Document (PRD)

**Project:** Intent Convergence Networks (ICN)
**Objective:** Detect malicious open-source packages (npm, PyPI, Rust, etc.) by modeling convergence/divergence of local and global code intents.
**Owner:** Security Research & ML Engineering Team
**Version:** Draft v1.0

---

## 1. Problem Statement

Open-source ecosystems are flooded with malicious packages that:

* Hide payloads inside install scripts (`setup.py`, `postinstall` in npm).
* Trojanize benign packages by adding a few malicious functions.
* Publish fully malicious packages masquerading as utilities.

Traditional approaches fail:

* **Flat classifiers (BERT/CodeBERT):** good at local anomalies, blind to package-level intent.
* **Static heuristics:** brittle, easy to bypass.
* **Dynamic sandboxes:** expensive, slow, evadable.

**Core Problem:** Maliciousness is an **emergent property** of how code units (functions/files) interact and align with package-level purpose.

---

## 2. Solution Overview

**Intent Convergence Networks (ICN):**

* Model each unit’s **local intent** (e.g., network, file IO, crypto).
* Derive a **global package intent** via iterative consensus.
* Detect malice as either:

    * **Divergence:** hidden malicious unit contradicts global intent.
    * **Plausibility failure:** global intent itself is abnormal (malicious-by-design).

ICN = HRM-inspired, but for security: **malice = failure of intent convergence or plausibility.**

---

## 3. Goals & Non-Goals

### Goals

* High recall on malicious packages (trojanized + malicious-by-design).
* Low false positives on benign networking/crypto libraries.
* Explainable outputs: highlight files/functions responsible.
* Scalable inference (seconds per package).
* Support multiple ecosystems (npm, PyPI, Rust).

### Non-Goals

* Dynamic runtime sandboxing.
* Detecting typosquatting by name (separate system).
* Full program verification (too costly).

---

## 4. Users & Use Cases

* **Security engineers / CISOs:** triage alerts from dependency scanners.
* **Developers:** pre-install package checks in CI/CD.
* **Registries (npm, PyPI, crates.io):** automated pre-publication screening.

---

## 5. Requirements

### Functional

* Parse packages into structured units (ASTs, tokens, API calls).
* Estimate local intents (fixed + latent).
* Run convergence loop to compute global intent.
* Output:

    * Binary score (malicious/benign).
    * Divergence map (files/functions).
    * Global plausibility score.
    * Phase heatmap.

### Non-Functional

* Throughput: median package ≤ 2s inference on A100.
* Accuracy: ROC-AUC ≥ 0.95 on test corpora.
* Robustness: must detect obfuscation + novel malware patterns.
* Explainability: forensic evidence must be human-readable.

---

## 6. Architecture

### 6.1 Data Flow

1. **Ingestion** → download & unpack package.
2. **Parsing** → tokenize, AST extraction, API event tagging, graph building.
3. **Unitization** → define local units (functions/files).
4. **Local Estimator** → per-unit intent distributions + embeddings.
5. **Global Integrator** → initialize global state, run convergence loop.
6. **Dual Detection Channels:**

    * Divergence metrics.
    * Global plausibility anomaly detection.
7. **Decision Head** → final verdict + evidence.
8. **Analyst Interface** → outputs JSON + UI with highlighted files/lines.

---

### 6.2 Core Components

#### Local Intent Estimator

* Shared transformer encoder across units.
* Inputs: tokens + AST features + API vector + phase.
* Outputs:

    * Fixed intent distribution.
    * Latent intent distribution.
    * Embedding vector $h_i$.

#### Global Intent Integrator

* Maintains global embedding $g^t$.
* Iteratively refines via attention over locals.
* Projects into global distribution $q^t$.

#### Convergence Loop

* 3–6 iterations.
* Local–global feedback updates both states.
* Stops when drift < ε or iteration cap reached.

#### Detection Channels

* **Divergence-driven:** mean divergence, max divergence, iteration count.
* **Plausibility-driven:** anomaly score on global embedding, phase alignment check, latent slot activation.

#### Decision Head

* Classifier over divergence + plausibility features.
* Outputs malicious probability, divergent units, abnormal global profile.

---

## 7. Training Strategy

### 7.1 Objectives

1. **Fixed intent supervision** (weak API labels).
2. **Latent intent discovery** (contrastive clustering).
3. **Benign convergence loss** (locals align quickly).
4. **Malicious margin loss** (trojans resist convergence).
5. **Global plausibility calibration** (one-class classifier on benign embeddings).
6. **Final classification** (binary).

### 7.2 Curriculum

* Stage A: pretrain locals on benign → intent prediction.
* Stage B: train global integrator on benign → convergence stability.
* Stage C: add malicious + synthetic → enable divergence & plausibility losses.
* Stage D: obfuscation-hardened training (minification, base64).

---

## 8. Evaluation

### Metrics

* **Detection:** ROC-AUC, PR-AUC, FPR\@TPR=95%.
* **Localization:** IoU\@k (flagged vs. true malicious units).
* **Convergence behavior:** benign = fast, malicious = divergent or implausible.
* **Zero-day generalization:** hold out attack families for test.
* **Explainability:** % of flagged units overlapping with ground truth.

### Benchmarks

* Datasets: curated benign corpora + malicious samples (PyPI typosquat malware, npm malicious disclosures, Rust malware repos).

---

## 9. Deployment

* **Inference:**

    * GPU batch local encodings.
    * Early-stop convergence when stable.
    * Caching of unit embeddings across versions.
* **Integration:**

    * CLI tool for CI/CD.
    * Registry scanner integration (pre-publication).
* **Output format:** JSON with score, divergent units, intents, phases.

---

## 10. Analyst Loop

* Review flagged units.
* Validate or label latent slots.
* Promote recurring latent slots into fixed taxonomy.
* Feedback loop improves interpretability over time.

---

## 11. Risks & Mitigations

* **Risk:** Global intent drift.

    * *Mitigation:* strong benign-only pretraining.
* **Risk:** Latent intents opaque.

    * *Mitigation:* analyst feedback loop.
* **Risk:** High package size.

    * *Mitigation:* hierarchical encoding (functions → file → package).
* **Risk:** Adversarial padding (inject benign noise).

    * *Mitigation:* focus on divergence + plausibility metrics, not raw size.

---

## 12. Roadmap (MVP → Production)

**Q1:**

* Data pipeline & AST extraction.
* Local Estimator pretraining.

**Q2:**

* Global Integrator & convergence loop.
* Train on benign corpora for stability.

**Q3:**

* Malicious + synthetic samples.
* Dual detection channel losses.
* Evaluation on trojanized & malicious-only test sets.

**Q4:**

* Obfuscation robustness.
* Analyst feedback loop.
* Integration into CI/CD + registry scanner PoC.

---

# ✅ Summary

ICN is a novel approach where:

* **Locals propose intents.**
* **Global integrates consensus.**
* **Malice emerges** when consensus fails (divergence) OR when the consensus itself is implausible.

It gives:

* Robustness to trojans & malicious-only packages.
* Interpretability via divergent units.
* Adaptability via hybrid intent space.