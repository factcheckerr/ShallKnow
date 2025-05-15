The Wilcoxon signed-rank test is a non-parametric test used to compare two related samples (like before/after measurements, or accuracy scores of two models on the same datasets). It tells you if the median difference between pairs is different from zero—in simpler terms, if one method consistently outperforms the other. 
# Key Concepts: 

     Null Hypothesis (H₀): There is no difference in medians between paired samples.
     Alternative Hypothesis (H₁): There is a median difference between pairs (two-sided), or one is greater/less (one-sided).
     

# Significance: 

A difference is considered “significant” if the p-value calculated by the test is less than your chosen significance level (often 0.05). 

     If p-value < 0.05 → Significant: You can reject the null hypothesis; the difference is unlikely due to random chance.
     If p-value ≥ 0.05 → Not significant: You do not reject the null hypothesis; there’s not enough evidence to support a difference.
     

wilcoxen tests
![Screenshot from 2025-05-14 19-13-04](https://github.com/user-attachments/assets/f1eb2ea6-95c1-4138-babe-4ccf15d68ddc)
