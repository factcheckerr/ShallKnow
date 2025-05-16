# Extracted Predicate List

This folder contains files listing all unique predicates (properties) extracted from the knowledge graph or raw triple datasets used in our experiments.

- Each file provides a summary of predicates either in original format (full URI) or with shortened/prefixed forms (e.g., `wdt:P17`, `P-Located_in`).
- Counts are provided to indicate how many times each predicate appears in the extracted data.
- These lists can be used for further statistical analysis, coverage evaluation, or to guide post-processing/filtering of triples.

**Example usage:**
- Identify the most frequently extracted properties.
- Compare predicate usage before and after shallow knowledge augmentation.

For additional analysis (e.g., histograms of predicate frequency), see our `/predicates` folder or associated scripts.

![predicate_hist_S'](https://github.com/user-attachments/assets/380c7713-ba78-4be0-ac96-7145c9ac6885)

![predicate_hist_S](https://github.com/user-attachments/assets/d687ef79-39f9-4899-9df2-6750860a2619)




# Extracted Predicate and Assertion Statistics

This folder contains statistics about the triples and predicates extracted by **ShallKnow** ($\mathcal{S}$ for primary, $\mathcal{S}'$ for secondary extraction).

|                      | **$\mathcal{S}$**    |         | **$\mathcal{S}'$**        |         |
|----------------------|----------------------|---------|----------------------------|---------|
| **Triples**          |                      | 101,477 |                            | 58,975  |
| **Properties**       |                      |   219   |                            | 11,620  |
| **Top-5 Properties** | Property             | Count   | Property                   | Count   |
| 1.                   | wdt:P17 [country]    | 21,143  | :P-Located\_in             | 1,407   |
| 2.                   | wdt:P276 [location]  | 8,028   | :P-located\_in             | 1,095   |
| 3.                   | wdt:P159 [HQ loc.]   | 4,631   | :P-Part\_of                |   896   |
| 4.                   | wdt:P26 [spouse]     | 4,522   | :P-Nationality             |   844   |
| 5.                   | wdt:P50 [author]     | 3,800   | :P-Associated\_with        |   747   |

<sub><b>Table:</b> Statistics about triples and properties extracted by ShallKnow ($\mathcal{S}$: primary extraction, $\mathcal{S}'$: secondary extraction).</sub>

---

## Description

- **$\mathcal{S}$:** Triples (RDF statements) extracted using the primary triple extraction module.
- **$\mathcal{S}'$:** Additional triples from secondary (LLM-based) triple extraction.
- **Counts** show the total extracted triples and unique properties.
- **Top-5 properties** give an overview of the most frequent predicates in both sets.
