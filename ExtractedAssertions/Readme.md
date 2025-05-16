# Extracted Assertions List

This folder contains files with the full list of extracted RDF triples (assertions) produced by our pipeline.

- Each file contains subject–predicate–object triples, typically in N-Triples (`.nt`) or tabular (`.csv`) format.
- These assertions represent factual statements either extracted from the original knowledge graph or generated from unstructured textual sources via shallow knowledge augmentation.

## Example: Sample Extracted Assertions

| Subject             | Predicate    | Object        |
|---------------------|-------------|--------------|
| wd:Q104542980       | wdt:P26     | wd:Q89925    |
| wd:Q104542980       | wdt:P40     | wd:Q4583     |
| wd:Q89925           | wdt:P40     | wd:Q7339     |

**See `assertions/` for the complete assertion files.**

For property usage statistics, refer to the `predicates/` folder.     


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
