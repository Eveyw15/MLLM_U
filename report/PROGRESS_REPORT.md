# Honours Thesis — Supervisor Progress Report

**Topic:** Cross-Modal Residual Leakage in Multimodal Unlearning

---

## Brief Overview

This thesis investigates a privacy vulnerability in multimodal AI systems: even after a model has been made to "forget" certain private information on the text side, that information may still be recoverable if the user supplies a relevant image at query time.

The project currently uses LLaVA-1.5-7B. The core idea is straightforward:

- Ask the model a factual question with a **blank white dummy image** -> model cannot answer correctly
- Ask the **same question** with the **real relevant image** -> model answers correctly

The gap between these two conditions is the **leakage signal**: the image is somehow helping the model recover an answer it could not produce without it.

The **proposed mitigation** is called inference-time channel suppression. Inside the model's transformer layers, there are thousands of internal "channels" (individual dimensions of hidden state vectors). Some of these channels respond much more strongly when a real image is present than when a blank image is used. The idea is to identify those channels and decrease them during inference, potentially cutting off the visual pathway that enables leakage without modifying any model weights and without retraining.

---

## 1. Completed Work

### 1.1 Experimental Pipeline

The full five-stage pipeline has been implemented as a basic framework, reproducible scripts that run on Google Colab with the quantised model:


| Stage | Script                        | What It Does                                                                                             |
| ----- | ----------------------------- | -------------------------------------------------------------------------------------------------------- |
| 1     | `download_coco_for_subset.py` | Downloads the relevant COCO images for the evaluation subset                                             |
| 2     | `minimal_eval_unlok.py`       | Runs baseline evaluation: dummy image vs. true image for each sample                                     |
| 3     | `analyze_channels.py`         | Uses forward hooks to identify which internal channels respond most differently to real vs. dummy images |
| 4     | `suppression.py`              | Inference-time module: registers a hook that zeros out selected channels during generation               |
| 5     | `eval_with_suppression.py`    | Three-way evaluation (dummy / true / suppressed) with a full parameter sweep                             |

No model weights are modified at any stage. All suppression is applied at inference time via PyTorch forward hooks.

---

### 1.2 Stage 2: Baseline Evaluation Results

Evaluated on 50 samples from the UnLOK-VQA benchmark. All 50 images located successfully.


| Condition                    | Soft Hit Rate | Notes                                        |
| ---------------------------- | ------------- | -------------------------------------------- |
| Dummy image (blank white)    | 0.260         | Model cannot see relevant visual information |
| True image (real COCO image) | 0.540         | Model has access to the relevant image       |
| **Leakage gap**              | **+0.280**    | Image enables recovery in 28% of cases       |

**Leak-like cases:** Of the 50 samples, **19** are "leak-like" that the true image enables a correct answer but the dummy image does not. These 19 cases are the primary analytical and evaluation target, because they directly demonstrate image-enabled knowledge recovery.

---

### 1.3 Stage 3: Channel Sensitivity Analysis

Run over the 19 leak-like cases. For each case, the model is run twice (dummy image and true image) while recording the hidden states at layers 28 and 31. The sensitivity score for each channel is the mean absolute difference in its activation between the two conditions, averaged over all leak-like cases.


| Layer  | Top-5 Sensitive Channels         | Top Channel Score |
| ------ | -------------------------------- | ----------------- |
| 28     | 1512, 2533, 1076, 1415, 2158     | 14.94             |
| **31** | **1512, 3241, 1360, 3556, 2660** | **30.09**         |

Layer 31 shows markedly stronger sensitivity than layer 28 (30.09 vs. 14.94). Channel 1512 appears in the top-5 for both layers. **Layer 31 was selected as the suppression target.**

---

### 1.4 Stage 4/5: Suppression Sweep Results

A sweep was run over top-k ∈ {1, 3, 5, 10} and α ∈ {0.0, 0.2, 0.5, 1.0} on layer 31. α = 0.0 zeros the selected channels out completely; α = 1.0 is a no-op (sanity check).

**Sanity check passed:** α = 1.0 reproduces the unsuppressed baseline exactly for all top-k values, confirming the hook is implemented correctly.

#### Full Sweep — All 50 Samples


| top-k | α      | Suppressed Rate | Delta (supp − true) | Good Flips (T→F) | Bad Flips (F→T) |
| ----- | ------- | --------------- | -------------------- | ----------------- | ---------------- |
| 1     | 0.0     | 0.540           | 0.000                | 0                 | 0                |
| 1     | 0.2     | 0.540           | 0.000                | 0                 | 0                |
| 1     | 0.5     | 0.540           | 0.000                | 0                 | 0                |
| 3     | 0.0     | 0.580           | +0.040               | 5                 | 7                |
| 3     | 0.2     | 0.560           | +0.020               | 4                 | 5                |
| 3     | 0.5     | 0.580           | +0.040               | 2                 | 4                |
| **5** | **0.0** | **0.560**       | **+0.020**           | **7**             | **8**            |
| 5     | 0.2     | 0.560           | +0.020               | 7                 | 8                |
| 5     | 0.5     | 0.580           | +0.040               | 3                 | 5                |
| 10    | 0.0     | 0.540           | 0.000                | 8                 | 8                |
| 10    | 0.2     | 0.560           | +0.020               | 4                 | 5                |
| 10    | 0.5     | 0.600           | +0.060               | 3                 | 6                |
| any   | 1.0     | 0.540           | 0.000                | 0                 | 0                |

**Full-set finding:** No setting reduces the aggregate hit rate. Good flips and bad flips approximately cancel out across the full set. The method is not a full-set improvement.

#### Full Sweep: Leak-Like Subset Only (n = 19)

Note: by definition, dummy rate = 0.000 and true rate = 1.000 for this subset (these are the cases where only the true image enables a correct answer).


| top-k | α       | Suppressed Rate | Delta (supp − true) | Good Flips (T→F) | Bad Flips (F→T) |
| ----- | -------- | --------------- | -------------------- | ----------------- | ---------------- |
| 1     | 0.0–0.5 | 1.000           | 0.000                | 0                 | 0                |
| 3     | 0.0      | 0.737           | −0.263              | 5                 | 0                |
| 3     | 0.2      | 0.789           | −0.211              | 4                 | 0                |
| 3     | 0.5      | 0.895           | −0.105              | 2                 | 0                |
| **5** | **0.0**  | **0.632**       | **−0.368**          | **7**             | **0**            |
| 5     | 0.2      | 0.632           | −0.368              | 7                 | 0                |
| 5     | 0.5      | 0.842           | −0.158              | 3                 | 0                |
| 10    | 0.0      | 0.632           | −0.368              | 7                 | 0                |
| 10    | 0.2      | 0.789           | −0.211              | 4                 | 0                |
| 10    | 0.5      | 0.842           | −0.158              | 3                 | 0                |
| any   | 1.0      | 1.000           | 0.000                | 0                 | 0                |

**Leak-like subset finding:** Suppression produces a clean targeted effect within this subset. **Zero bad flips occur at any setting** , that is, suppression never causes the model to produce a new correct answer on a case it previously got wrong. The best result (top-k = 5 or 10, α = 0.0) reduces the recovery rate from **1.000 -> 0.632**, corresponding to 7 out of 19 leak-like cases where image-enabled recovery is successfully disrupted.

---

## 2. Interpretation

### What the results show

The central finding is a **dissociation between the full set and the leak-like subset**:

- On the **full 50-sample set**, suppression does not help: good flips (suppression correctly blocks recovery) are approximately cancelled out by bad flips (suppression incorrectly disrupts a previously correct answer). The method is not globally improving.
- On the **19 leak-like cases, which represent direct evidence of image-enabled knowledge recovery suppression produces only good flips and zero bad flips. The method is surgically affecting the right cases.

This pattern makes conceptual sense: the sensitive channels carry visual information that is relevant both to the leak-like cases and to other image-conditioned responses in the full set. Zeroing them disrupts image-based reasoning more broadly, which causes bad flips outside the leak-like set. The selectivity of the effect on the leak-like set without bad flips within that set is genuine, but the collateral damage elsewhere is real.

### What remains unclear

- Whether any setting provides a net benefit after accounting for the utility cost from bad flips on the full set.
- Whether the sensitive channels identified are genuinely encoding private factual knowledge or are carrying general visual-semantic information.
- Whether the effect would hold on an explicitly unlearned model (current experiments use the pre-unlearning model).

### What is not being claimed

- That the method works overall — the full-set results do not support this.
- That 50 samples is sufficient to draw strong conclusions.
- That utility has been preserved — a proper retain-set evaluation has not yet been done.

---

## 3. Outstanding Work


| Item                                                   | Priority | Status               |
| ------------------------------------------------------ | -------- | -------------------- |
| Retain-set utility check (~50 unrelated samples)       | High     | Not yet run          |
| Layer 28 suppression comparison (matched sweep)        | Medium   | Not yet run          |
| Thesis draft — Discussion and Conclusion sections     | High     | Not yet written      |
| Thesis draft — Results tables updated with full sweep | Done     | All cells confirmed  |
| Two-layer suppression (layers 28 + 31 jointly)         | Low      | Exploratory, pending |
| Scale evaluation to N ≥ 200 samples                   | Medium   | Pending              |

---

## 4. Questions

1. **Utility evaluation:** Is the dummy-image hit rate a reasonable proxy for utility at this stage, or should I source a separate retain set? The concern is that channels carrying general visual reasoning ability are also being suppressed.
2. **Contribution framing:** Given that the full-set results are neutral-to-negative, is it defensible to present the primary contribution as (a) operationalising and measuring cross-modal leakage, and (b) showing a targeted suppression effect on the leak-like subset — rather than claiming a working suppression solution?
3. **Scope check:** Does the current experimental scope — one model, one dataset, 50 samples, single-layer suppression — feel appropriate for an honours thesis, or should I prioritise scaling before finalising results?

---
