# Honours Timeline

## Week 1 (Apr 1-7)

Main thing this week: pin down the gating design before touching any more code. The current suppression results are good enough to motivate the method but I need something trainable to have a real contribution.

**Tasks**
- Decide gate insertion point - projector output is the first choice, early LLM layer as fallback if the projector route doesn't work
- Split the data properly (forget / retain)
- Write a minimal training script, doesn't need to be clean yet, just needs to run
- Move everything to a proper remote environment, Colab is getting annoying for longer runs
- Start writing - Introduction and Related Work at minimum, ideally also get a rough Problem Definition done

**By end of week:** gate design is decided (not just "probably this"), something trains without crashing, intro + related work in some form

---

## Week 2 (Apr 8-14)

First real training run. Expecting things to break.

**Tasks**
- Train the first version on the forget/retain setup
- Evaluate three things: does it forget on text-only? does image-conditioned leakage change? does retain-side accuracy hold?
- Frame the suppression sweep results properly in the thesis - they're diagnostic evidence, not the main method
- Experimental Setup section + start Preliminary Results

**By end of week:** one result table, even if the numbers aren't good yet

---

## Week 3 (Apr 15-21)

Make the trained module actually defensible, not just runnable.

**Tasks**
- Run a proper comparison: trainable gate vs inference-time suppression vs baseline (no mitigation)
- Small hyperparameter sweep - gate strength at minimum, insertion point if still undecided
- Write the Method section properly and get a rough Discussion going

The comparison table is the most important output this week. Without it the thesis doesn't have a story.

**By end of week:** comparison table, Method + Results in rough draft form

---

## Week 4 (Apr 22-28)

Utility evaluation and finishing the writing.

**Tasks**
- Add a retain-set utility check - this is the one thing missing from the current results and reviewers will ask about it
- Check the leakage-utility tradeoff between the trainable module and the suppression baseline
- Write Discussion, Limitations, Future Work

**By end of week:** utility table, near-complete draft

---

## Week 5 (Apr 29 - May 5)

Assemble everything into one document.

- Pull all sections together
- Make sure the story is coherent: diagnostic pipeline -> suppression evidence -> trainable module as main method
- Tables and figures in place
- Tone check - claims need to be modest given sample sizes

**By end of week:** rough draft v1, full document

---

## Week 6 (May 6-12)

Revision + code cleanup.

- Revise based on supervisor feedback if any comes in
- Fix wording, check references, caption all figures
- Make sure the pipeline actually runs end-to-end for someone else (reproducibility)
- README update

**By end of week:** draft v2

---

## Week 7 (May 13-19)

Shift focus to defence.

- Abstract and Conclusion (easier to write once everything else is done)
- Slide deck v1 - structure first, content second
- Think through likely questions

---

## Week 8 (May 20-31)

Final stretch.

- Fix whatever comes up
- Rerun any small experiments if needed
- Format and polish
- Rehearse

**Submission + defence ready by May 31**
