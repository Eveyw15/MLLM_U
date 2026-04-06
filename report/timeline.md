# Honours Timeline

## Week 1 (Apr 1-7)

**Tasks**
- Decide gate insertion point (first choice - projector output, early LLM layer as fallback if the projector route doesn't work)
- Split the data properly (forget / retain)
- Write a minimal training script, doesn't need to be clean yet, just needs to run
- try new remote environment, Colab's running is too long
- thesis draft - Introduction and Related Work first, Problem Definition can be an extension

Gate design decided **by the end of week**

---

## Week 2 (Apr 8-14)

**Tasks**
- Train the first version on the forget/retain setup
- Evaluate three things: 
  - does it forget on text-only? 
  - does image-conditioned leakage change? 
  - does retain-side accuracy hold?
- Frame the suppression sweep results properly in the thesis - they're diagnostic evidence, not the main method
- Experimental Setup section + start Preliminary Results in thesis

One result table and initial comparison **by the end of week**

---

## Week 3 (Apr 15-21)

**Tasks**
- Run a proper comparison: trainable gate vs inference-time suppression vs baseline (no mitigation)
- Small hyperparameter sweep - gate strength at minimum, insertion point if still undecided
- Write the Method section properly and get a rough Discussion going

The comparison table is the most important output this week.

---

## Week 4 (Apr 22-28)

**Tasks**
- Add a retain-set utility check
- Check the leakage-utility tradeoff between the trainable module and the suppression baseline
- Write Discussion, Limitations, Future Work

Utility table, near-complete draft **By end of week**

---

## Week 5 (Apr 29 - May 5)

Assemble everything into one document.

- Pull all sections together
- Make sure the story is coherent: diagnostic pipeline -> suppression evidence -> trainable module as main method
- Tables and figures in place

Rough draft v1, full document **By end of week**

---

## Week 6 (May 6-12)

Revision + code cleanup.

- Fix wording, check references, caption all figures
- Make sure the pipeline actually runs end-to-end for someone else (reproducibility)
- README update

draft v2 **By end of week**

---

## Week 7 (May 13-19)

- Write Abstract and Conclusion
- Slide v1 - structure first
- Think through likely questions

---

## Week 8 (May 20-31)

Final stretch.

- Fix whatever comes up
- Rerun any small experiments if needed
- Format and polish
- Rehearse

