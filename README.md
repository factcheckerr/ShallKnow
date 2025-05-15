
# No Need to Be a Know-It-All: Fact Checking Approach with Shallow Knowledge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/factcheckerr/ShallKnow/ci.yml?branch=main)](https://github.com/factcheckerr/ShallKnow/actions)
[![Docs](https://img.shields.io/badge/docs-complete-brightgreen.svg)](#)


This repository contains the official implementation of **ShallKnow**—a framework for improving fact-checking over knowledge graphs by augmenting them with automatically extracted RDF triples ("shallow knowledge") from unstructured text.

ShallKnow enables more effective support or refutation of factual claims by increasing KG coverage with high-utility, external information.

---

## 🚀 Quick Start

| **Step**                   | **Command / Notes**                                                               |
|----------------------------|-----------------------------------------------------------------------------------|
| Clone repo & create env    | `git clone https://github.com/factcheckerr/ShallKnow.git`<br>`cd ShallKnow`       |
|                            | `python3 -m venv venv`<br>`source venv/bin/activate`<br>`pip install -r requirements.txt` |
| Install Ollama (for LLMs)  | [Ollama download & docs](https://ollama.com/download)                             |
| Run DeepSeek LLM           | `ollama pull deepseek-r1:14b`<br>`ollama run deepseek-r1:14b`                     |
| Run Entity-Centric Paragraph Simplification         | `python scripts/wikipedia_extractor_final.py deepseek-r1:14b`                     |
| (Advanced) Triple Extraction API | See below for Docker-based API extraction and example `curl` calls         |

---

## 💻 Hardware Requirements

- **Recommended:** 128 CPU cores, 1 TB RAM, 2×NVIDIA RTX 6000 Ada GPUs
- **Notes:** A GPU is required for LLM and Relik components.

---

## 🔧 Installation

```bash
 git clone https://github.com/factcheckerr/ShallKnow.git
 cd ShallKnow  
 python3 -m venv venv
 source venv/bin/activate
 pip install -r requirements.txt
```

---

## 🧪 Running Experiments

### 1. Start LLM (DeepSeek) with Ollama

```bash
ollama pull deepseek-r1:14b
ollama run deepseek-r1:14b
```
(See [Ollama download](https://ollama.com/download) if needed.)

### 2. Entity-Centric Paragraph Simplification and KG Augmentation

Run the Entity-Centric Paragraph Simplification script:

```bash
python scripts/wikipedia_extractor_final.py deepseek-r1:14b
```

---

### 3 🔄 Triple Extraction API (Advanced)

To extract new triples from unstructured text via API:

```bash
cd TripleExtraction
sudo docker compose up
```
Then, run DeepSeek in the Ollama container:
```bash
sudo docker ps  # Find the Ollama container ID
sudo docker exec -it <container_id> bash
# Inside the container:
ollama pull deepseek-r1:14b
ollama run deepseek-r1:14b
```

### Calling the Triple Extraction API

- **For a folder of preprocessed articles:**
  ```bash
  curl --location --request POST 'http://localhost:5000/dextract' \
    --header 'Content-Type: application/x-www-form-urlencoded' \
    --data-urlencode 'query=folder:/your/path/to/preprocessed_folder' \
    --data-urlencode 'components=triple_extraction'
  ```

- **For a single sentence or paragraph:**
  ```bash
  curl --location --request POST 'http://localhost:5000/extract' \
    --header 'Content-Type: application/x-www-form-urlencoded' \
    --data-urlencode 'query=Edith Frank was married to Otto Frank and born in Frankfurt.' \
    --data-urlencode 'components=triple_extraction'
  ```

**Note:** Use `dextract` for batch/folder processing or `extract` for a single text input.

**Example output:**
![Overview](utils/triples_extraction.png)

### 3 Alternate approach
Alternatively, use the script:
```bash
python scripts/extract_triples.py
```
Adjust the API endpoint in the script if needed (default: `http://localhost:5000/extract`).

---

## 📊 Output Stats

A snapshot of the top properties in our extracted triples:

| Property         | Count    |
|------------------|----------|
| wdt:P17          | 21,143   |
| wdt:P276         | 8,028    |
|------------------|----------|
| P-Located_in     | 1,407    |
| P-Nationality    | 844      |
|------------------|----------|

Full CSVs and charts are available in `/Prediction_files_and_AUROC_graphs`.

---

## 📚 Additional Resources

### Datasets

All datasets are provided on [Zenodo](https://zenodo.org/records/15390036).

### Supporting Tools

- [KnowledgeStream](https://github.com/saschaTrippel/knowledgestream): Path-based plausibility scoring for RDF triples
- [FAVEL](https://github.com/dice-group/favel): Benchmark fact-checking evaluation platform
- [GERBIL](https://gerbil-kbc.aksw.org/gerbil/config): Standardized benchmarking of KG tasks

---

## 📜 Citation

If you use ShallKnow in your research, please cite:

```bibtex
# TODO
```
---

## 🙏 Acknowledgements

*To be added later.*

---
---

## 🤝 Contributing and Support

We welcome pull requests and issue reports! For questions and further contributions, please [open an issue](https://github.com/factcheckerr/ShallKnow/issues).
```
