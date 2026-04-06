# THESIS_SOURCE_PACKET.md (v1 — core set + UnLOK-VQA benchmark + current results)

## 1. Project summary

**Working topic**
Multimodal privacy unlearning in multimodal large language models (MLLMs), with a focus on **cross-modal residual leakage / backpath leakage**.

**Core problem**
Even if private knowledge appears to be removed on the text side, it may still be recoverable through image-conditioned inputs, prompt variants, or nearby multimodal routes.

**Current method direction**A lightweight, inference-time **backpath-aware selective suppression** approach:

1. Run baseline evaluation with dummy image vs. true image.
2. Filter leak-like cases.
3. Identify sensitive internal channels through hidden-state analysis.
4. Suppress selected channels at inference time.
5. Compare baseline vs. suppression.

## 2. Current experimental status

### Baseline status

- Total samples: 50
- Images found: 50/50
- Errors: 0
- Dummy soft hit rate: 0.260
- True soft hit rate: 0.540
- Leakage gap: +0.280

### Interpretation

This is preliminary evidence that real images can help recover target information that is less accessible under dummy-image inputs.

### Channel analysis status

- Leak-like cases analyzed: 18
- Layers analyzed: 28 and 31
- Layer 31 shows stronger head-channel scores than layer 28
- Example high-sensitivity channels:
  - Layer 28: 1512, 2533, 1076, 1415, 2158, ...
  - Layer 31: 1512, 3241, 1360, 3556, 2660, ...

### Suppression status (first pass)

- Setting tested: layer 31, top-k = 5, alpha = 0.0
- Suppressed soft hit rate: 0.560
- Leakage gap after suppression: +0.300
- Suppression delta vs. true image: +0.020

### Current interpretation

The first suppression attempt did **not** reduce leakage. On this 50-sample run, suppression slightly increased the hit rate from 0.540 to 0.560. This is only a +0.020 difference (roughly one additional hit out of 50), so it is best treated as an inconclusive or negative first-pass result rather than evidence that the method works.

### Immediate next step

1. Verify the suppression hook is active and deterministic (e.g., alpha=1.0 should reproduce the unsuppressed baseline).
2. Run a sweep over alpha and top-k on layer 31.
3. Evaluate changes specifically on the leak-like subset, not only the full 50-sample average.
4. If needed, compare layer 31 vs. layer 28 and then a small two-layer variant.

## 3. Draft contribution statement

This thesis aims to:

1. Define and measure cross-modal residual leakage in multimodal unlearning.
2. Show that image-conditioned inputs can partially recover forgotten information.
3. Identify sensitive internal channels associated with this recovery behavior.
4. Test whether lightweight inference-time channel suppression can reduce leakage while retaining basic utility.

## 4. Core references verified so far

### A. Core unlearning background

#### [R1] Large Language Model Unlearning (NeurIPS 2024)

**Why it matters**
A strong general LLM unlearning paper for framing unlearning as a practical post-training alignment tool.

**Use for section**
Introduction / Related Work.

**What to cite it for**

- Why unlearning matters in practice.
- Standard motivations such as harmful content removal, copyright-related removal, and privacy-related forgetting.

**How it relates to this thesis**
Useful as the broad unlearning backdrop, but it is not multimodal.

#### [R2] TOFU: A Task of Fictitious Unlearning for LLMs (arXiv:2401.06121)

**Why it matters**
A benchmark-oriented paper for LLM unlearning.

**Use for section**
Related Work / Evaluation background.

**What to cite it for**

- Benchmarking unlearning quality.
- The idea that unlearning should be evaluated against stronger notions than surface forgetting.

**How it relates to this thesis**
Useful as an evaluation reference, especially when motivating why simple post-unlearning accuracy is not enough.

#### [R3] Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning (arXiv:2406.13356)

**Why it matters**
Directly relevant to the concern that apparent forgetting may not be robust and can be reversed or recovered.

**Use for section**
Introduction / Related Work / Motivation.

**What to cite it for**

- Unlearning may look successful while remaining vulnerable to relearning.
- Evaluation should consider recovery-style attacks, not just immediate forgetting scores.

**How it relates to this thesis**
Conceptually very close to your “backpath leakage / recovery” motivation, even though your setting is multimodal rather than text-only.

### B. Direct multimodal unlearning references

#### [R4] MultiDelete for Multimodal Machine Unlearning (arXiv:2311.12047)

**Why it matters**
One of the earliest direct multimodal unlearning papers.

**Use for section**
Related Work.

**What to cite it for**

- Multimodal unlearning is qualitatively harder because modalities are coupled.
- Early method framing for multimodal machine unlearning.

**How it relates to this thesis**
Important direct baseline/precedent paper. Very likely a must-cite.

#### [R5] Single Image Unlearning: Efficient Machine Unlearning in Multimodal Large Language Models (arXiv:2405.12523)

**Why it matters**
A direct MLLM unlearning paper focused on efficient visual concept forgetting.

**Use for section**
Related Work / Experimental comparison background.

**What to cite it for**

- MLLM-specific unlearning.
- Visual concept forgetting via efficient fine-tuning.
- The existence of a benchmark/metric-oriented MLLM unlearning setup.

**How it relates to this thesis**
A key direct comparator because your work also sits in the MLLM unlearning space.

#### [R6] MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models (arXiv:2502.11051)

**Why it matters**
A recent paper explicitly reformulating multimodal unlearning for MLLMs.

**Use for section**
Related Work / Problem Definition.

**What to cite it for**

- Recent reframing of multimodal unlearning specifically for MLLMs.
- The distinction between forgetting visual patterns while preserving textual knowledge.

**How it relates to this thesis**
Very relevant to your problem framing because your work also cares about selective forgetting rather than simply destroying all associated knowledge.

#### [R7] CLIPErase: Efficient Unlearning of Visual-Textual Associations in CLIP (arXiv:2410.23330)

**Why it matters**
A multimodal unlearning paper on CLIP-style vision-language representations.

**Use for section**
Related Work.

**What to cite it for**

- Unlearning in aligned vision-language representation models.
- Association-level forgetting in multimodal embedding models.

**How it relates to this thesis**
Not an MLLM paper, but highly relevant for the idea that aligned cross-modal associations can be selectively weakened.

#### [R7b] Unlearning Sensitive Information in Multimodal LLMs: Benchmark and Attack-Defense Evaluation (arXiv:2505.01456)

**Why it matters**
This is the paper associated with the **UnLOK-VQA** benchmark and dataset you are currently using. It introduces UnLOK-VQA as a multimodal unlearning benchmark and evaluates attack-defense settings for deleting specific multimodal knowledge from MLLMs.

**Use for section**
Experimental Setup / Related Work / Preliminary Results.

**What to cite it for**

- The definition and purpose of the **UnLOK-VQA** benchmark.
- The dataset structure (target question-answer pair, neighborhood questions, image IDs, and related evaluation assets).
- The broader attack-defense framing for multimodal unlearning evaluation.

**How it relates to this thesis**
This is a **must-cite** reference for your current experiments because your baseline and channel-analysis pipeline are built on the UnLOK-VQA data format and benchmark setting.

### C. Multimodal representation background

#### [R8] ImageBind: One Embedding Space to Bind Them All (arXiv:2305.05665)

**Why it matters**
A strong reference for the idea of shared multimodal representation spaces.

**Use for section**
Introduction / Problem Definition.

**What to cite it for**

- Modern multimodal systems often rely on shared or aligned representation spaces.
- Cross-modal coupling makes selective forgetting harder.

**How it relates to this thesis**
Useful for motivating why knowledge can remain reachable through another modality.

#### [R9] CLIP: Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)

**Why it matters**
A foundational vision-language alignment paper.

**Use for section**
Background only.

**What to cite it for**

- Large-scale visual-text alignment.
- The basis of many later multimodal alignment ideas.

**How it relates to this thesis**
Useful as background, but not central to your direct thesis argument.

### D. Method-style inspiration (optional)

#### [R10] Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection (arXiv:2004.07667)

**Why it matters**
Not an unlearning paper in your exact setting, but relevant as inspiration for representation-level suppression/removal.

**Use for section**
Optional Method Motivation.

**What to cite it for**

- Representation-level removal or suppression of unwanted information.

**How it relates to this thesis**
Potentially useful if you want one citation supporting the idea of removing sensitive signals in hidden representations.

## 5. References verified but probably low priority for this thesis draft

#### [L1] ViT: An Image is Worth 16x16 Words (arXiv:2010.11929)

This is a major vision backbone paper, but it is probably unnecessary for your current thesis draft unless you end up discussing architectural background in more detail.

#### [L2] Medium article on self-attention

Not suitable as a thesis citation.

## 6. Initial section-to-reference map

### Introduction

Use first:

- [R1] Large Language Model Unlearning
- [R3] Unlearning or Obfuscating?
- [R4] MultiDelete
- [R5] Single Image Unlearning
- [R8] ImageBind

### Related Work

Group into:

1. **LLM unlearning**: [R1], [R2], [R3]
2. **Multimodal / MLLM unlearning**: [R4], [R5], [R6], [R7]
3. **Multimodal alignment background**: [R8], [R9]
4. **Representation-level removal inspiration (optional)**: [R10]

### Problem Definition / Threat Model

Use first:

- [R3] for recovery / relearning concerns
- [R6] for selective multimodal forgetting framing
- [R8] and optionally [R9] for shared/aligned representation motivation

### Experimental Setup

Use first:

- [R7b] for the UnLOK-VQA benchmark and dataset
- [R5] if you need MLLM unlearning benchmark framing beyond your current dataset
- Your own current setup and code status

## 7. Duplicates / cleanup already visible in the link list

The following IDs appear more than once in your raw list:

- 2502.11051
- 2405.12523
- 2410.23330
- 2506.23145
- 2511.06793

These should be deduplicated before building the final reference list.

## 8. Links not yet screened in this version

I have **not yet verified** the later 2025–2026 arXiv-heavy tail in this file. They should remain in a backlog until checked individually.

Unscreened examples include:

- 2505.02884
- 2506.23145
- 2506.17265
- 2506.05198
- 2503.15166
- 2503.12545
- 2503.12127
- 2503.11832
- 2502.15910
- 2502.12520
- 2411.19939
- 2410.18057
- 2410.15267
- 2507.01271
- 2508.*
- 2509.*
- 2510.*
- 2511.*
- 2512.*
- 2601.*
- 2602.*
- 2603.*

## 9. What to do next

### Good next step

Screen the unscreened papers in batches of **8–12 links** and assign each one to one of these labels:

- direct multimodal unlearning
- MLLM unlearning
- benchmark / evaluation
- attack / recovery / relearning
- representation editing / suppression inspiration
- background only
- drop for now

### Once enough sources are verified

Use this packet to draft:

- Introduction
- Related Work
- Problem Definition / Threat Model
- Method Overview
- Experimental Setup
- Preliminary Results

## 10. Reference links

https://proceedings.neurips.cc/paper_files/paper/2024/hash/be52acf6bccf4a8c0a90fe2f5cfcead3-Abstract-Conference.html
https://arxiv.org/pdf/2401.06121
https://arxiv.org/pdf/2505.02884
https://arxiv.org/pdf/2406.13356
https://arxiv.org/pdf/2311.12047v2
https://arxiv.org/pdf/2502.11051
https://arxiv.org/pdf/2010.11929
https://arxiv.org/pdf/2103.00020
https://medium.com/@manindersingh120996/the-detailed-explanation-of-self-attention-in-simple-words-dec917f83ef3
https://arxiv.org/pdf/2305.05665
https://arxiv.org/pdf/2410.23330
https://arxiv.org/pdf/2004.07667
https://arxiv.org/pdf/2506.23145
https://arxiv.org/pdf/2405.12523
https://arxiv.org/pdf/2511.06793
https://arxiv.org/pdf/2603.21484
https://arxiv.org/pdf/2603.14185
https://arxiv.org/pdf/2602.16197
https://arxiv.org/pdf/2602.16144
https://arxiv.org/pdf/2601.22020
https://arxiv.org/pdf/2601.21794
https://arxiv.org/pdf/2601.16527
https://arxiv.org/pdf/2601.13264
https://arxiv.org/pdf/2512.17911
https://arxiv.org/pdf/2512.14137
https://arxiv.org/pdf/2512.14113
https://arxiv.org/pdf/2512.09867
https://arxiv.org/pdf/2512.02713
https://arxiv.org/pdf/2511.20196
https://arxiv.org/pdf/2511.18444
https://arxiv.org/pdf/2511.11299
https://arxiv.org/pdf/2511.06793
https://arxiv.org/pdf/2510.22535
https://arxiv.org/pdf/2510.04217
https://arxiv.org/pdf/2509.23895
https://arxiv.org/pdf/2508.19554
https://arxiv.org/pdf/2508.04567
https://arxiv.org/pdf/2508.04192
https://arxiv.org/pdf/2508.03091
https://arxiv.org/pdf/2507.01271
https://arxiv.org/pdf/2506.23603
https://arxiv.org/pdf/2506.23145
https://arxiv.org/pdf/2506.17265
https://arxiv.org/pdf/2506.05198
https://arxiv.org/pdf/2505.01456
https://arxiv.org/pdf/2503.15166
https://arxiv.org/pdf/2503.12545
https://arxiv.org/pdf/2503.12127
https://arxiv.org/pdf/2503.11832
https://arxiv.org/pdf/2502.15910
https://arxiv.org/pdf/2502.12520
https://arxiv.org/pdf/2502.11051
https://arxiv.org/pdf/2411.19939
https://arxiv.org/pdf/2410.23330
https://arxiv.org/pdf/2410.18057
https://arxiv.org/pdf/2410.15267
https://arxiv.org/pdf/2405.12523
