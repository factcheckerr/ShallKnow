# No Need to Be a Know-It-All: Fact Checking Approach with Shallow Knowledge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/...)

This repository contains the official implementation of **ShallKnow**, our fact-checking framework for knowledge graphs, as described in the paper:

**No Need to Be a Know-It-All: Fact Checking with Shallow Knowledge.** [arXiv link here]

ShallKnow augments knowledge graphs (KGs) with shallow knowledge—RDF triples extracted automatically from unstructured text—enabling more effective support or refutation of factual claims.

---

## 🚀 Quick Start

| **Step**                                | **Command / Notes**                                          |
|------------------------------------------|--------------------------------------------------------------|
| Clone the repo & create env              | `git clone https://github.com/factcheckerr/ShallKnow.git`<br>`cd ShallKnow`<br>`python3 -m venv venv`<br>`source venv/bin/activate`<br>`pip install -r requirements.txt` |
| Install Ollama (for LLMs)                | See [Ollama website](https://ollama.com/download)            |
| Run DeepSeek LLM                         | `ollama pull deepseek-r1:14b`<br>`ollama run deepseek-r1:14b`|
| Start extraction and KG augmentation     | `python scripts/wikipedia_extractor_final.py deepseek-r1:14b`|
| (Advanced) Triple Extraction API (Docker)| See section below                                            |

---

## 💻 Hardware Requirements

All experiments were executed on a server with **128 CPU cores, 1 TB RAM, and 2×NVIDIA RTX 6000 Ada GPUs**. A GPU is essential for running LLM and Relik models.

---

## 🔧 Installation

<details id="__DETAIL_0__"/>

---

## 🧪 Running Experiments

### 1. Start LLM (DeepSeek) with Ollama

- [Install Ollama](https://ollama.com/download) if not yet done.
- Pull and run the DeepSeek model:

```bash
ollama pull deepseek-r1:14b
ollama run deepseek-r1:14b
```

### 2. Run Main Extraction

Once the LLM is running, start the extraction:

```bash
python scripts/wikipedia_extractor_final.py deepseek-r1:14b
```

---

## 🔄 Triple Extraction API (Optional - Advanced)

To extract new triples from unstructured text via API:

```bash
cd TripleExtraction
sudo docker compose up
```

Then, run DeepSeek inside the Ollama container:
```bash
sudo docker ps  # Find the container ID for Ollama
sudo docker exec -it <container_id> bash
```

Point your extraction script to the API endpoint (`extract_triples.py`, default `http://localhost:5000/extract`):

```bash
python scripts/extract_triples.py
```
**Example output:**  
![Overview](utils/triples_extraction.png)

---

## 📊 Example Output

Here’s a snapshot of the top properties in our extracted triples (trimmed):

| Property         | Count    |
|------------------|----------|
| wdt:P17          | 21,143   |
| wdt:P276         | 8,028    |
| P-Located_in     | 1,407    |
| P-Nationality    | 844      |

(see `/analysis` for full CSVs and charts for your own datasets)

---

## 📚 Additional Resources

### Datasets

All datasets and benchmark splits are described in [this Zenodo record](https://zenodo.org/records/15390036).

### Supporting Tools

- [KnowledgeStream](https://github.com/saschaTrippel/knowledgestream): Path-based scoring for RDF triples.
- [FAVEL](https://github.com/dice-group/favel): Fact-checking evaluation.
- [GERBIL](https://gerbil-kbc.aksw.org/gerbil/config): Standardized KG benchmarking.

---

## 📜 Citation

If you use ShallKnow in your research, please cite:

```bibtex
@article{your_shallknow_bibtex_entry,
  title={No Need to Be a Know-It-All: Fact Checking with Shallow Knowledge},
  author={...},
  year={2024},
  journal={...},
  note={\url{https://github.com/factcheckerr/ShallKnow}}
}
```

---

## 🤝 Contributing and Support

We welcome pull requests and issue reports! For questions and collaboration, please [open an issue](https://github.com/factcheckerr/ShallKnow/issues) or email us.

---

**Copy the text above for your README.md—do not wrap it in triple backticks!**  
If you paste it into GitHub, it will render as intended.  
Let me know if you’d like to customize further!


# No Need to Be a Know-It-All:Fact Checking Approach with Shallow Knowledge
This open-source project contains the Python implementation of our approach, ShallKnow. It is designed to improve fact-checking over knowledge graphs by augmenting them with shallow knowledge, automatically extracted RDF triples from unstructured sources. ShallKnow improves KG coverage, enabling more effective support or refutation of claims. 

## Pre-requisites
All experiments in paper are executed on a server with 128 CPU cores, 1 TB RAM, and 2×NVIDIA RTX 6000 Ada Generation GPUs.  GPU is important for running LLMs and Relik models.


## Installation

First, clone the repository:
<details><summary> </summary>
 
```bash
git clone https://github.com/factcheckerr/ShallKnow.git
cd ShallKnow
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
or 

```bash
git clone https://github.com/factcheckerr/ShallKnow.git
cd ShallKnow
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
</details>



## Running experiments

To run this experiment, it is essential that Ollama is installed on your system. If it is not already installed, you can find more information and download it from the official website [here](https://ollama.com/download)

Then, pull and run the required model:
<details><summary> </summary>

```bash
ollama pull deepseek-r1:14b
ollama run deepseek-r1:14b
```
</details>

Once the model is running, execute the main script:

```bash
python scripts/wikipedia_extractor_final.py deepseek-r1:14b
```


## Starting Triple Extraction API
If you want to extract new triples from an unstructured textual source. You need to start triple extraction API using [docker image](https://github.com/factcheckerr/ShallKnow/blob/main/TripleExtraction/docker-compose.yml) file.

```bash
cd TripleExtraction
sudo docker up
```
and then you have to run deepseek-r1:14b inside docker container for secondary triple extraction step.

```bash
sudo docker ps
```

it will return the IDs of 3 container. Note down the ID of Ollama container and then execute the following command, replacing the IDs.

```bash
sudo docker exec -it <pid> bash
```

After that, the output files from this experiment will serve as the input for the triple extraction process. Update the input folder name accordingly and proceed to run the next experiment.
Also, change the endpoint of the api [here](https://github.com/factcheckerr/ShallKnow/blob/main/scripts/extract_triples.py) if hosted on different port. `endpoint = "http://localhost:5000/extract"`


```bash
python scripts/extract_triples.py
```

<i>The following is an example of triple extraction:<i>


![Overview](utils/triples_extraction.png)



## Additional Resources

### Datasets

More informations about included datasets [here](https://zenodo.org/records/15390036)

### Supporting Tools


To compute fact-checking scores and evaluate our approach in ShallKnow, we use the following tools:

- [KnowledgeStream](https://github.com/saschaTrippel/knowledgestream): Used to compute plausibility scores for RDF triples based on path-based reasoning over the knowledge graph.
- [FAVEL](https://github.com/dice-group/favel): Used for fact-checking evaluation.
- [GERBIL](https://gerbil-kbc.aksw.org/gerbil/config): Used for standardized benchmarking.







