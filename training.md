# 🔑 Training Loss Mapping for ICN

Your dataset already has a **semantic split**:

* `compromised_lib` → benign library with an injected malicious unit.
* `malicious_intent` → package built from the ground up to be malicious.

That aligns 1:1 with ICN’s **dual detection channels**:

* Divergence channel → `compromised_lib`.
* Plausibility channel → `malicious_intent`.

---

## 1. Losses for `compromised_lib` Samples (Trojanized)

Goal: teach the model that **malice = locals vs. global failing to converge.**

* **Divergence Margin Loss**

  * At least one local intent distribution $p_i$ must remain *far* from the global intent $q$.
  * Encourage persistent mismatch:

    $$
    \mathcal{L}_{\text{margin}} = \max\!\left(0,\, \tau - \max_i \text{KL}(p_i \,\|\, q)\right)
    $$

* **Localization Signal**

  * Identify which unit(s) caused divergence.
  * Auxiliary loss: binary classify units as malicious vs. benign.

* **Convergence Regularization (for benign parts)**

  * Non-malicious units must align tightly with global intent.
  * Keeps the system from just labeling “everything malicious.”

---

## 2. Losses for `malicious_intent` Samples (Malicious-by-Design)

Goal: teach the model that **malice = abnormal global profile, even if locals agree.**

* **Global Plausibility Loss**

  * Global embedding $g$ should lie *outside* the benign intent manifold.
  * Use one-class or contrastive loss:

    $$
    \mathcal{L}_{\text{plaus}} = \max\!\left(0,\, \tau - d(g, \mathcal{M}_{\text{benign}})\right)
    $$

    where $d$ = distance from benign cluster center(s).

* **Phase Constraint Loss**

  * Penalize if risky intents (net, exec, env) occur in `install/postinstall`.
  * Encourages the model to weight context in plausibility checks.

* **Latent Activation Regularizer**

  * If latent slots dominate the global distribution, mark as suspicious.
  * Keeps latent slots from becoming “benign catch-alls.”

---

## 3. Losses for Pure Benign Samples

Goal: teach the model what *normal convergence* looks like.

* **Convergence Loss**

  * Locals should align rapidly with global.
  * Encourage minimal drift across iterations.
  * $$
    \mathcal{L}_{\text{conv}} = \frac{1}{N} \sum_i \text{KL}(p_i \,\|\, q)
    $$

* **Stability Loss**

  * Benign packages should converge in ≤3 iterations.
  * Penalize models that require more steps.

---

## 4. Combined Objective

For a minibatch with mixed categories:

$$
\mathcal{L} =
\lambda_1 \mathcal{L}_{\text{intent}} +
\lambda_2 \mathcal{L}_{\text{conv}} +
\lambda_3 \mathcal{L}_{\text{margin}} +
\lambda_4 \mathcal{L}_{\text{plaus}} +
\lambda_5 \mathcal{L}_{\text{cls}}
$$

* **$\mathcal{L}_{\text{intent}}$:** weak supervision on fixed intents (all samples).
* **$\mathcal{L}_{\text{conv}}$:** benign + benign parts of compromised\_lib.
* **$\mathcal{L}_{\text{margin}}$:** compromised\_lib only.
* **$\mathcal{L}_{\text{plaus}}$:** malicious\_intent only.
* **$\mathcal{L}_{\text{cls}}$:** final package-level malicious/benign classification.

---

## 5. Ratio & Sampling

* **Mix per batch:**

  * 5–10 benign : 1 malicious (malicious evenly split between compromised\_lib & malicious\_intent).
* **Sampling benigns:** 50% from top downloads, 50% long tail.
* **Batch diversity:** avoid batches of all one type (so the model always sees contrastive signals).

---

✅ This way:

* `compromised_lib` trains the **divergence channel**.
* `malicious_intent` trains the **plausibility channel**.
* Benign reinforces **stable convergence**.