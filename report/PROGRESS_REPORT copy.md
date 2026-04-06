# Honours Thesis Progress Report
---

## Topic

Cross-Modal Residual Leakage in Multimodal Unlearning

---

## Current Focus

My project is currently focused on whether information that appears to be forgotten on the text side can still be recovered through image-conditioned inputs in a multimodal model.

The current study uses LLaVA-1.5-7B and examines cross-modal residual leakage through a small controlled evaluation pipeline.

---

## Completed Work

### 1. Baseline evaluation

I built a baseline evaluation pipeline on 50 samples from the UnLOK-VQA setting.

The results show a clear leakage signal:

- Dummy-image soft hit rate: 0.260
- True-image soft hit rate: 0.540
- Leakage gap: +0.280

This suggests that the real image can substantially increase target recovery compared with a blank image.

### 2. Channel sensitivity analysis

I then analysed internal channel activations on the leak-like subset, defined as cases where the dummy image fails but the true image succeeds.

The analysis was run on layers 28 and 31. Layer 31 showed stronger sensitivity than layer 28, which made it the main target for the next stage.

### 3. Inference-time channel suppression

Based on the sensitivity analysis, I implemented a lightweight inference-time selective suppression module.

This method does not modify model weights and does not require retraining. Instead, it suppresses a small set of selected internal channels during inference.

### 4. Preliminary suppression results

A parameter sweep was run on layer 31.

At the full-set level, the results are mixed and do not show an overall improvement.

However, on the leak-like subset, the suppression shows a meaningful preliminary effect. For example, with top-k = 5 and alpha = 0.0, the subset hit rate drops from 1.000 to 0.632, corresponding to 7 successful true-to-false flips.

--- 
## Current Interpretation

The current evidence suggests that:

1. Cross-modal leakage is observable in the present setting.
2. A small set of late-layer channels appears to be associated with this leakage.
3. Targeted suppression can partially disrupt recovery on leakage-prone cases.

At the same time, the current results do not support claiming that the method improves performance on the full set, and utility preservation has not yet been established.

---
## Next Steps

1. Add a small retain-side or utility check
2. Organise the current results into a cleaner summary table
3. Continue extending the thesis draft, especially the method and preliminary results sections

---

## Questions

I would appreciate feedback on the following points:

1. I am currently planning to focus the thesis mainly on cross-modal leakage between the text and image sides. Do you think this is a reasonable way to frame the problem?
2. Based on the current results, I feel that this work is better described as first measuring the leakage phenomenon and then exploring a preliminary suppression attempt, rather than presenting a complete method. Do you think this is an appropriate positioning for an honours thesis?
3. For the next step, I am unsure whether it would be more useful to first add a utility or retain-side check, to see whether suppression hurts normal performance, or to move to a better GPU environment and expand the experiment scale further. Which direction do you think would be more worthwhile to start first?
