
---

# Statistical Tests: Wilcoxon Signed-Rank Test

This folder documents our use of the **Wilcoxon signed-rank test** for statistical analysis of paired model results throughout our experiments.

---

## What is the Wilcoxon Signed-Rank Test?

The Wilcoxon signed-rank test is a non-parametric statistical test for comparing two related samples, such as model predictions before and after using our method, or accuracy scores for two models evaluated on the same datasets. The test determines whether the median of the differences between paired observations is significantly different from zero—indicating consistent improvement or decline.

---

### Hypotheses

- **Null Hypothesis (H₀):** There is no median difference between paired samples (the methods perform similarly).
- **Alternative Hypothesis (H₁):** There is a median difference (or one-sided difference) between paired samples.

---

### Statistical Significance

A difference is considered **statistically significant** if the calculated *p*-value is less than your chosen significance threshold (typically 0.05):

- If *p*-value < 0.05 → **Significant:** Reject H₀; a real difference exists.
- If *p*-value ≥ 0.05 → **Not significant:** Cannot reject H₀; insufficient evidence of a difference.

---

## Example Output

> The figure below illustrates an example output of the Wilcoxon signed-rank test comparing paired model results.

![Wilcoxon Test Example](https://github.com/user-attachments/assets/f1eb2ea6-95c1-4138-babe-4ccf15d68ddc)

---

*For additional analysis details, see the included scripts and result files in this directory.*

---
