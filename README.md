
# No Need to Be a Know-It-All: Fact Checking Approach with Shallow Knowledge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

### 2. Run Entity-Centric Paragraph Simplification

Once the LLM is running, start the extraction:

```bash
python scripts/wikipedia_extractor_final.py deepseek-r1:14b
```

Once all paragraphs are simplified. Run the triple extraction API for triple extraction from them.

---

## 🔄 Triple Extraction API 

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
#TODO 
}
```

---

## 🤝 Contributing and Support

We welcome pull requests and issue reports! For questions and collaboration, please [open an issue](https://github.com/factcheckerr/ShallKnow/issues) or email us.
