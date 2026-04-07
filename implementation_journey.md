# Implementation Journey: From Fixed Suppression to Input-Conditioned Gating

## Project

**Multimodal Privacy Unlearning in MLLMs: Cross-Modal Residual Leakage / Backpath Leakage**

## Purpose of this document

This document records the implementation path of the project, including:

- what was tried,
- why each design choice was made,
- what evidence motivated the next step,
- what worked,
- what did not work,
- and what remains uncertain.

The goal is to preserve the reasoning behind the implementation, so that later thesis writing can describe the method development process with concrete evidence rather than only reporting the final result.

---

# 1. Initial research focus

The core research question is:

> Even if a multimodal model appears to have “forgotten” private information under text-only evaluation, the information may still be recoverable through image-conditioned inputs, prompt variants, or adjacent multimodal pathways.

This project is **not** intended to build a heavy retraining-based unlearning framework.
The fixed scope is:

- baseline evaluation
- leak-like case filtering
- channel sensitivity analysis
- lightweight selective suppression / gating
- suppression or gating evaluation

The project explicitly avoids:

- LoRA-heavy or full-model retraining
- large-scale benchmark expansion
- large engineering refactors
- complex new training pipelines unless necessary for a lightweight method

---

# 2. Stage I: Baseline evaluation

## Goal

Establish whether there is a clear cross-modal leakage signal in the current model setting.

## Setup

- Benchmark context: UnLOK-VQA style data and format
- Evaluation size: 50 samples
- Model family: LLaVA-1.5-7B
- Comparison:
  - dummy image condition
  - true image condition

## Result

- Total samples: 50
- Images found: 50/50
- Errors: 0
- Dummy soft hit rate: 0.260
- True soft hit rate: 0.540
- Leakage gap: **+0.280**

## Interpretation

This result provided the first concrete evidence that:

- the image modality materially increases target-answer recovery,
- and therefore there is a real cross-modal leakage signal worth studying.

## Why this mattered

This baseline justified moving beyond pure speculation.
Without this result, there would be no reason to pursue internal channel analysis or suppression.

---

# 3. Stage II: Leak-like case filtering

## Goal

Narrow the analysis to the subset of cases where leakage is most visible.

## Motivation

The full set is heterogeneous.
Not every sample exhibits the same degree of image-conditioned recovery.
A focused subset is more appropriate for:

- mechanism analysis,
- early intervention testing,
- and lightweight suppression experiments.

## Working definition

Leak-like cases are those where:

- the true-image condition recovers the target answer,
- and the dummy-image condition does not.

## Result

From the 50-sample evaluation:

- **19 leak-like cases** were identified

## Interpretation

This subset was used as the main target subset for later:

- channel sensitivity analysis,
- suppression experiments,
- and the first trainable gate experiments.

## Why this mattered

This step prevented the project from being diluted by averaging over many weak or irrelevant cases.
It also made later mitigation results easier to interpret.

---

# 4. Stage III: Channel sensitivity analysis

## Goal

Identify internal channels that are likely associated with the leakage behaviour.

## Layers analysed

- Layer 28
- Layer 31

## Method intuition

For each sample, compare hidden activations under:

- true image
- dummy image

Then rank channels by sensitivity, using activation differences.

## Main results

### Layer 28 top channels

- indices: [1512, 2533, 1076, 1415, 2158, 257, 2789, 490, 2883, 2298]
- scores: [14.9423, 6.0889, 3.8148, 3.4691, 3.3624, 3.0264, 2.6814, 2.5586, 2.4309, 2.3110]

### Layer 31 top channels

- indices: [1512, 3241, 1360, 3556, 2660, 1415, 2042, 1839, 1927, 2533]
- scores: [30.0877, 14.8942, 10.0426, 9.2455, 7.8007, 7.4017, 7.3410, 7.1741, 6.9909, 6.8640]

## Interpretation

Three conclusions were important here:

1. **Layer 31 showed a stronger head-channel signal than Layer 28.**
2. **Channel 1512 was prominent in both layers.**
3. **Late layers became the first practical intervention target.**

## Why this mattered

This analysis gave an evidence-based reason to intervene at **layer 31**, rather than choosing a layer arbitrarily.

This also supported the later design choice:

- first try fixed suppression at layer 31,
- then try learned gating at the same layer,
- so the learned module can be compared against an already-motivated baseline.

---

# 5. Stage IV: Fixed suppression baseline

## Goal

Test whether directly suppressing the identified sensitive channels can reduce leakage.

## Design

A simple intervention was implemented:

- choose a target layer,
- choose top-k sensitive channels,
- scale them by a fixed scalar `alpha`,
- evaluate the effect on the leak-like subset and full set.

Formally, this was equivalent to:

`h[:, :, top_k_channels] *= alpha`

where:

- `alpha = 1.0` means no suppression
- `alpha = 0.0` means full suppression on the selected channels

## Sweep

- layer: 31
- top-k ∈ {1, 3, 5, 10}
- alpha ∈ {0.0, 0.2, 0.5, 1.0}

## Sanity result

At `alpha = 1.0`, the “suppressed” result matched the unsuppressed baseline.
This verified that:

- the hook was attached correctly,
- the evaluation logic was consistent,
- and the intervention path itself was not broken.

## Key result

Most representative point:

- layer 31
- top-k = 5
- alpha = 0.0

Leak-like subset result:

- true hit rate: **1.000**
- suppressed hit rate: **0.632**
- suppression delta: **-0.3684**
- good flips: **7**
- bad flips: **0**

## Interpretation

This was the strongest preliminary mitigation signal in the project so far.

However, the full-set aggregate remained mixed.
Therefore the correct conclusion was:

> Fixed suppression provides preliminary but meaningful evidence on the leak-like subset, but does not establish a full solution.

## Why this mattered

This stage showed that:

- the channel analysis was not meaningless,
- targeted intervention can change behaviour,
- and the leak-like subset is a valid place to test lightweight mitigation.

At the same time, this stage also showed a limitation:

- a fixed scalar suppression is blunt,
- and it does not adapt to different inputs.

That limitation motivated the next stage.

---

# 6. Why move from fixed suppression to a learned gate?

## Problem with fixed suppression

Fixed suppression applies the same strength to every sample:

`g(x) = alpha`

This is simple, but crude.

It cannot distinguish:

- forget-like inputs that should be suppressed,
- from benign or retain-side inputs that should remain mostly unchanged.

## Desired improvement

The natural upgrade was:

> Instead of using a constant `alpha`, learn a small module that outputs different suppression strengths for different inputs.

This led to the design shift from:

- **fixed suppression**
  to
- **input-conditioned gating**

## Important design principle

The model weights themselves should remain frozen as much as possible.
The trainable component should remain small and interpretable.

This preserves the project’s original lightweight scope.

---

# 7. Stage V: First gate design attempts

## Early design idea that was rejected

An early idea was to learn a single global per-channel gate vector:

`h[:, :, c] *= sigmoid(gate_c)`

with one scalar parameter per channel.

## Why it was rejected

This design had a fundamental conflict.

The same global gate would be used for:

- forget samples
- retain samples

But the intended objectives were opposite:

- suppress on forget
- preserve on retain

A single shared gate cannot reasonably do both in an input-specific way.
It can only learn a global compromise.

## Conclusion

This design was rejected before formal training.

---

# 8. Stage VI: Input-conditioned learned gate

## Goal

Replace fixed suppression with a lightweight trainable module that:

- reads the current hidden state,
- predicts a per-sample gate,
- and scales only the selected sensitive channels.

## Insertion point

First implementation choice:

- **layer 31**

## Why layer 31 was chosen

- already supported by channel sensitivity evidence
- already supported by fixed suppression results
- direct comparability with earlier experiments
- lower implementation risk than moving upstream immediately

## Controlled channels

First implementation:

- **top-k sensitive channels only**
- initial setting: `top-k = 5`

## Why top-k = 5 was chosen

- directly matches the strongest fixed suppression point
- keeps the module very small
- easier to interpret in the thesis
- reduces overfitting risk on a small dataset

## Gate formula

Let `h` be the hidden state at layer 31:

- `h` shape: `[B, T, 4096]`
- extract selected channels:
  - `h_k = h[:, :, top_k_indices]`
- pool channel activity:
  - `z = mean(abs(h_k), dim=token)`
- compute gate:
  - `g(x) = sigmoid(W2 * ReLU(W1 * z + b1) + b2)`
- apply gate:
  - `h[:, :, top_k_indices] *= g(x)`

## Important design decisions

### 1. Use `mean(abs(h_k))`, not plain mean

Reason:

- plain mean may cancel positive and negative activations
- the sensitivity analysis itself was based on magnitude differences

### 2. Conservative initialization

The gate should begin mostly open, not half-closed.

Reason:

- avoid immediate collapse in retain behaviour
- allow the gate to learn suppression gradually

Observed initial gate mean in smoke test:

- approximately **0.88**

### 3. Only train the gate

The main model remains frozen.
This keeps the method lightweight.

---

# 9. Stage VII: Training objective revision

## Initial training objective issue

The first training version had two major problems:

### Problem A: retain loss was wrong

It originally used the prompt tokens themselves as labels.
This would only encourage prompt reproduction, not answer preservation.

### Problem B: forget loss mixed prompt and answer tokens

This polluted the signal by optimizing over the prompt span as well.

## Fix

A revised masking strategy was introduced.

### Retain loss

- load `true_answer` from the ungated baseline JSONL
- use it as `teacher_answer`
- build `prompt + teacher_answer`
- mask all prompt tokens with `-100`
- compute CE only on answer tokens

### Forget loss

- build `prompt + target_answer`
- mask all prompt tokens with `-100`
- apply negative CE only to the target answer tokens

## Interpretation

After this revision:

- retain training was aligned with preserving baseline answer behaviour
- forget training was aligned with reducing target-answer recovery
- prompt modelling was no longer mixed into the objective

---

# 10. Stage VIII: Data split refinement for gate training

## Original issue

The retain side originally included everything that was not leak-like.
This was too broad.

Some of those samples were not reliable examples of “behaviour worth preserving”.

## Fix

Retain training was restricted to:

- `image_found == True`
- `true_match_soft == True`
- not leak-like

## Resulting split

- Forget set: **19**
- Retain set: **8**

## Interpretation

This retain set is small, but higher quality.
It is better suited for a first controlled gate experiment.

---

# 11. Stage IX: First gate training run

## Setup

- insertion layer: 31
- top-k: 5
- trainable parameters: 60
- epochs: 5
- forget set: 19
- retain set: 8

Selected channels:

- [1512, 3241, 1360, 3556, 2660]

## Training log

- Epoch 1: forget_loss = -9.0043, retain_loss = 0.4377
- Epoch 2: forget_loss = -9.1275, retain_loss = 0.4352
- Epoch 3: forget_loss = -9.2791, retain_loss = 0.4311
- Epoch 4: forget_loss = -9.4652, retain_loss = 0.4291
- Epoch 5: forget_loss = -9.6870, retain_loss = 0.4283

## Immediate observation

The optimization behaved stably:

- forget loss became increasingly negative
- retain loss decreased slightly
- training did not diverge

This suggests that the gate was trainable and the objective was not broken.

---

# 12. Stage X: First post-training evaluation of the gate

## Forget set result

- Ungated hit rate: **1.000**
- Gated hit rate: **0.895**
- Good flips: **2**
- Bad flips: **0**

## Retain set result

- Ungated hit rate: **1.000**
- Gated hit rate: **1.000**

## Interpretation

This result shows that:

- the learned gate is feasible,
- the gate can be trained stably,
- and it does have some leakage suppression effect.

However, the effect is currently **modest**:

- only 2 good flips on the forget set
- much weaker than the strongest fixed suppression baseline

## Comparison with fixed suppression

Representative fixed suppression result:

- leak-like subset: **1.000 → 0.632**
- good flips: **7**
- bad flips: **0**

First learned gate result:

- forget subset: **1.000 → 0.895**
- good flips: **2**
- bad flips: **0**

## Current conclusion

At this stage, the correct interpretation is:

> The learned gate is trainable and stable, and it provides preliminary suppression signal without visible collapse on the small retain set. However, in its current form, it is still weaker than the strongest fixed suppression baseline.

---

# 13. What has been learned so far

## Things that now seem well-supported

1. Cross-modal leakage is real in the current benchmark setting.
2. Late-layer channel sensitivity is meaningful.
3. Fixed suppression on selected channels can reduce leakage on leak-like cases.
4. A lightweight input-conditioned gate can be trained stably.
5. The gate does not appear to destroy performance immediately on the small high-quality retain subset.

## Things that remain uncertain

1. Whether the learned gate can outperform fixed suppression.
2. Whether the current retain result generalizes beyond the small retain subset.
3. Whether layer 31 is the best insertion point for a trainable module.
4. Whether gate initialization or loss balance is making the current gate too conservative.
5. Whether a different retain weight or a slightly stronger forget objective would improve the trade-off.

---

# 14. Why specific parameters were chosen

This section records the rationale behind the current choices.

## Why layer 31?

- strongest available channel sensitivity signal
- most direct continuity with earlier suppression experiments
- easiest to compare with the fixed baseline

## Why top-k = 5?

- matches the strongest fixed suppression result
- reduces overfitting risk
- keeps the gate interpretable and lightweight

## Why use `mean(abs(h_k))`?

- avoids sign cancellation
- better matches the magnitude-based sensitivity analysis

## Why conservative gate initialization?

- avoids collapsing retain behaviour at the start
- allows the gate to learn selective suppression gradually

## Why freeze the main model?

- keeps the method lightweight
- matches project scope
- makes the intervention more interpretable

## Why use baseline `true_answer` as retain teacher?

- gives a concrete answer-preservation target
- is more meaningful than reproducing the prompt
- better aligns with “preserve normal behaviour”

---

# 15. Current limitations

1. The learned gate currently underperforms the strongest fixed suppression baseline.
2. The retain set is small (8 samples), so utility evidence is still weak.
3. The current gate pools over tokens in a coarse way.
4. The current forget objective is still a simple negative target-answer objective, not a richer safe-response objective.
5. The method has not yet been tested on a broader utility set.
6. The current conclusions are still preliminary and should be described conservatively.

---

# 16. Immediate next steps

The next step should **not** be a full redesign.

The most sensible short-term next steps are:

1. record gate statistics after training
   
   - average gate value on forget samples
   - average gate value on retain samples
   - per-channel gate values
2. run a very small diagnostic sweep
   
   - slightly lower initial gate openness
   - slightly adjust retain weight
   - keep architecture fixed
3. compare learned gate and fixed suppression under the same evaluation protocol

The purpose of the next round is not to expand scope, but to answer:

> Why is the learned gate currently more conservative and weaker than the fixed suppression baseline?

---

# 17. Thesis writing value of this journey

This implementation path is useful for the thesis because it provides a concrete development narrative:

1. **Baseline evaluation** established leakage.
2. **Leak-like filtering** isolated the most relevant cases.
3. **Channel analysis** identified sensitive internal channels.
4. **Fixed suppression** showed that targeted intervention could matter.
5. **Learned gating** was introduced as a more selective upgrade over fixed suppression.
6. The first learned gate result showed feasibility and stability, but also revealed that improved selectivity does not automatically outperform a stronger fixed intervention.

This gives the thesis a more credible methodology section because the final design is not arbitrary.
It is motivated by intermediate evidence and explicit implementation decisions.

---

# 18. Notes to update later

The following items should be updated after each new experiment:

- date
- code version / commit
- changed hyperparameters
- gate statistics
- evaluation results
- short interpretation
- whether the result supports or weakens the current method hypothesis

---

# 19. Short current project summary

Current project state can be summarized as follows:

- baseline leakage evidence: **clear**
- channel sensitivity evidence: **meaningful**
- fixed suppression signal: **strong on leak-like subset**
- learned gate feasibility: **established**
- learned gate performance: **currently modest**
- utility evidence: **still limited**
- thesis framing: **preliminary but meaningful evidence, not full success**

