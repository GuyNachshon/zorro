# CPG-GNN for Malicious Package Detection

---

## 1. Problem Statement

Malware often hides in **flows**: e.g.,
`read env vars → encode → send to network`.
Token-only classifiers may miss this.
We need a graph-based detector that captures **structural and semantic flows** inside code.

---

## 2. Goals

* Detect malicious packages by modeling data/control flow.
* Highlight suspicious subgraphs (motifs).
* More resilient to obfuscation than token models.

---

## 3. Non-Goals

* Sub-line explanations (down to specific AST node).
* Real-time CI/CD latency (GNNs are heavier).

---

## 4. Approach

Use **Code Property Graphs (CPGs)**:

* Nodes = AST elements, API calls, identifiers.
* Edges = AST, control flow, data flow.
* Train a **Graph Neural Network (GNN)** to classify package-level maliciousness.

---

## 5. Inputs & Features

* **Graph construction:**

  * Parse code into AST.
  * Add control flow edges.
  * Add data flow edges (var assignments → uses).
* **Node features:**

  * Token embeddings (CodeBERT pretrained vectors).
  * Node type (AST type, API type).
  * Risky API flags (net, fs, env, eval).
* **Graph-level metadata:**

  * Number of files.
  * File entropy scores.

---

## 6. Architecture

**Graph builder:**

* Combine all file-graphs into package-level graph.
* Add metadata node connected to all subgraphs.

**Encoder:**

* Graph Transformer or GIN (Graph Isomorphism Network).
* 3–5 message-passing layers.

**Pooling:**

* Attention pooling over nodes.
* Global embedding (size 512).

**Classifier:**

* MLP → malicious probability.

**Outputs:**

* Malicious probability.
* Top-k suspicious subgraphs (via attention scores).

---

## 7. Training

* **Loss:** binary cross-entropy at package level.
* **Auxiliary losses:**

  * Predict which risky APIs exist in package (multi-label).
  * Predict entropy level of strings (to train robustness).
* **Curriculum:**

  * Stage A: benign + malicious-intent packages.
  * Stage B: add compromised\_lib trojans.
  * Stage C: obfuscation augmentation.

---

## 8. Evaluation

* **Classification:** ROC-AUC ≥ 0.95, FPR < 2%.
* **Graph localization:** overlap of flagged subgraphs with injected malicious flows.
* **Obfuscation robustness:** measure performance drop ≤5% on obfuscated packages.
* **Scalability:** average inference time ≤ 10s/package.

---

## 9. Risks & Mitigations

* **Graph building cost:** AST/CFG parsing may fail on malformed code.

  * Mitigation: fallback to token embeddings.
* **GNN training instability:** large graphs → GPU OOM.

  * Mitigation: hierarchical pooling (file → package).
* **Analyst interpretability:** flagged subgraphs may be too abstract.

  * Mitigation: highlight API call paths (e.g., `env → encode → net`).

---

## 10. Success Criteria

* High detection (ROC-AUC ≥ 0.95).
* Localizes malicious flows with ≥70% accuracy on synthetic injections.
* Robust to obfuscation (≤5% drop).
* Reasonable latency (<10s/package).

---

## 11. Roadmap

* **Week 1–2:** CPG builder (AST + CFG + DFG).
* **Week 3–4:** Prototype GNN baseline (GIN).
* **Week 5–6:** Add attention pooling, auxiliary losses.
* **Week 7–8:** Augmentation (obfuscation), family hold-out eval.
