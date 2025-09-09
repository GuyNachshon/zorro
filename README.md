# 🧩 Zorro: Malicious Package Detection Framework

This repository contains two complementary approaches to malicious package detection:

## 🔍 ICN (Intent Convergence Networks)
Advanced iterative model for intent misalignment detection through local-global convergence.

## 🎯 AMIL (Attention-based Multiple Instance Learning)  
Lightweight attention-based classifier optimized for CI/CD and registry scanning.

---

# 🧩 ICN Development Plan & Guidelines

---

## 1. Conceptual Foundation

**Problem framing:**

* Malice is not a property of a single line.
* Malice is **emergent**: it appears when local code behavior (functions, scripts) and global package behavior (declared/majority purpose) misalign.
* ICN detects malice in two ways:

  * **Divergence:** locals disagree with global → hidden payloads (trojans).
  * **Plausibility failure:** locals + global agree, but intent profile is abnormal → malicious-by-design packages.

---

## 2. Data Preparation

### 2.1 Sources

* **Benign:** large crawls of npm, PyPI, crates.io (both top projects + long tail).
* **Malicious:**

  * Public disclosures (e.g., PyPI malware dataset).
  * Security advisories (npm malicious packages).
  * GitHub repos with documented supply-chain attacks.
* **Synthetic:** inject realistic payloads (e.g., env exfil in postinstall, base64-encoded eval).

### 2.2 Parsing

* For each file:

  * **Tokens:** subtokenize identifiers, literals.
  * **AST:** node type sequence, control flow.
  * **API events:** net, fs, crypto, subprocess, eval, env.
  * **Phase:** install / postinstall / runtime.
  * **Metadata:** file path, size, entropy features.

### 2.3 Unitization

* Define **units** = functions (if small) or whole files (if large).
* Add manifest (`setup.py`, `package.json`, `Cargo.toml`) as special units.

---

## 3. Intent Vocabulary

### 3.1 Fixed Intents

* Seed list of \~15 categories (net.outbound, net.inbound, fs.read, fs.write, proc.spawn, eval/exec, crypto, sys.env, installer/hooks, encoding, config).
* Supervise weakly via API→intent mappings.

### 3.2 Latent Intents

* Add 8–12 unlabeled slots.
* Learned unsupervised (contrastive clustering, InfoNCE).
* Analysts can promote stable latent slots into fixed taxonomy later.

---

## 4. Model Design

### 4.1 Local Intent Estimator

* Backbone: **CodeBERT-like encoder** (6–12 layers, hidden dim 256–512).
* Inputs: tokens + AST embedding + API event embedding + phase token.
* Outputs:

  * Fixed intent distribution (`p_fixed`).
  * Latent intent distribution (`p_latent`).
  * Embedding vector $h_i$.

### 4.2 Global Intent Integrator

* Maintains package-level state $g^t$.
* Initialization: mean pooling of locals + manifest embedding.
* Iterative refinement:

  * Attend over local embeddings.
  * Update $g^{t+1}$ via GRU/transformer.
  * Project into global intent distribution $q^{t+1}$.

### 4.3 Convergence Loop

* Repeat 3–6 iterations:

  1. Locals update → global state.
  2. Global sends feedback → locals adjust.
* Stop when drift < ε or max iterations.

### 4.4 Dual Detection Channels

* **Divergence channel:**

  * Mean divergence, max divergence, convergence speed.
* **Plausibility channel:**

  * Global embedding anomaly score (trained one-class on benign).
  * Phase misalignment rules (e.g., net.outbound in install phase).
  * Latent slot overactivation.

### 4.5 Decision Head

* Input: $[g^T, \bar{D}, D_{\max}, T, \text{phase features}]$.
* Output:

  * Malicious / benign score.
  * Divergent units (forensics).
  * Plausibility explanation.

---

## 5. Training Guidelines

### 5.1 Objectives

* **Intent supervision:** BCE on fixed slots with weak labels.
* **Latent discovery:** contrastive clustering on benign units.
* **Convergence loss:** enforce locals align with global on benign.
* **Margin loss:** enforce divergence persists on malicious.
* **Plausibility calibration:** train density model on global embeddings from benign.
* **Final classifier:** package-level malicious/benign.

### 5.2 Curriculum

1. Pretrain locals on benign → intent prediction.
2. Train global integrator on benign → convergence stability.
3. Introduce malicious + synthetic → divergence margin + classification.
4. Obfuscation robustness (augment with minified/encoded samples).

---

## 6. Inference Flow

1. Ingest package, parse into units.
2. Encode units → local intents + embeddings.
3. Run convergence loop (3–6 iterations).
4. Compute divergence metrics.
5. Check plausibility of global intent.
6. Decision head produces final verdict.
7. Output JSON report with:

   * Verdict (malicious/benign + probability).
   * Divergent units.
   * Intent mismatches.
   * Phase heatmap.

---

## 7. Analyst Loop

* Analysts review flagged units.
* Label recurring latent slots → promote to fixed taxonomy.
* Validate outputs in CI/CD / registry scanning.
* Feedback loop → retraining.

---

## 8. Deployment Guidelines

* **Batch encoding:** parallelize unit embeddings.
* **Early stopping:** skip extra iterations if convergence stable.
* **Caching:** reuse unit embeddings across package versions.
* **Latency target:** ≤2s for small/medium packages, <10s for large packages.
* **Integration:**

  * CLI tool for CI/CD.
  * Registry scanning API.
  * JSON + UI outputs for analysts.

---

## 9. Risks & Mitigations

* **Risk:** Global intent drift (all packages look the same).

  * *Mitigation:* benign-only pretraining with convergence loss.
* **Risk:** Latent slots too opaque.

  * *Mitigation:* analyst feedback + taxonomy promotion.
* **Risk:** Obfuscation hides signals.

  * *Mitigation:* train with obfuscated variants.
* **Risk:** Fully malicious packages converge.

  * *Mitigation:* plausibility channel.

---

## 10. Success Criteria

* Detection ROC-AUC ≥ 0.95.
* False positive rate < 2% on benign networking/crypto libs.
* Localization IoU ≥ 0.7 (flagged units overlap with malicious code).
* Zero-day generalization (holdout malware families detected with >70% recall).
* Analyst trust: divergent units + intent explanations are actionable.

---

## 11. Roadmap

**MVP (Q1–Q2):**

* Build parser pipeline.
* Pretrain local estimator.
* Package-level pooling baseline (no loop).
* Test divergence metric on synthetic trojans.

**Alpha (Q3):**

* Add convergence loop.
* Add plausibility channel.
* Evaluate on real malware datasets.

**Beta (Q4):**

* Robustness training (obfuscation, adversarial padding).
* Analyst feedback loop.
* CI/CD + registry PoC.

---

# 🎯 AMIL (Attention-based Multiple Instance Learning)

## Overview

AMIL is a lightweight, interpretable classifier designed for fast malicious package detection in CI/CD pipelines and registry scanning. Unlike ICN's iterative approach, AMIL uses attention-based multiple instance learning to classify packages in a single forward pass.

## Key Features

- **Speed**: ≤2s inference per package
- **Interpretability**: Attention-based unit localization
- **Lightweight**: Minimal compute requirements
- **Production-ready**: Optimized for CI/CD integration

## Architecture

1. **Feature Extraction**: Code embeddings + handcrafted features (APIs, entropy, metadata)
2. **Attention Pooling**: Multi-head attention aggregates unit representations
3. **Classification**: Binary malicious/benign prediction with confidence scores
4. **Localization**: Attention weights identify suspicious code units

## Usage

```bash
# Run AMIL demo
python amil_demo.py

# Train AMIL model
python -m amil.trainer --config amil_config.json

# Evaluate AMIL
python -m amil.evaluator --model-path trained_amil.pth --test-data test_samples/
```

## Success Criteria

- Detection ROC-AUC ≥ 0.95
- False positive rate < 2%  
- Localization IoU ≥ 0.7
- Inference speed ≤2s per package
- Integration with benchmark framework

---

✅ This plan ties together:

* **Data pipeline** → how to extract local code intents.
* **Model design** → local & global components, convergence loop.
* **Dual detection** → divergence vs. plausibility.
* **Analyst workflow** → interpretability & feedback.
* **Deployment** → integration into dev/security ecosystems.