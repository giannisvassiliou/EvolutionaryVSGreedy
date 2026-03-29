# When Does Greedy Suffice?  
### Workload-Aware Knowledge Graph Summarization

This repository contains the implementation and experimental framework for the paper:

> **"When Does Greedy Suffice? A Study of Workload-Aware KG Summarization"**  

---

## 📌 Overview

We study when simple **greedy (frequency-based)** methods are sufficient for workload-aware knowledge graph summarization, and when more expressive methods like **Genetic Algorithms (GA)** provide benefits.

The repository provides a **unified framework** for fair comparison between:
- **WBSum (Greedy baseline)**
- **Genetic Algorithm (multi-objective optimization)**

---

## ⚙️ Features

- Workload-aware schema summarization
- Support for:
  - Real SPARQL workloads (e.g., DBpedia)
  - Synthetic workload generation
  - Rare-query injection
- Structural constraints:
  - Budget limits
  - Connectivity enforcement
- Multi-objective scoring:
  - Coverage
  - Relevance
  - Rare-query coverage

---

## 🧪 Methods

### Greedy (WBSum)
- Selects nodes based on query frequency
- Fast and effective under structured workloads

### Genetic Algorithm (GA)
- Multi-objective optimization (NSGA-II style)
- Diversified initialization:
  - Frequency-based seeds
  - Rare-aware seeds
  - Random seeds
- Mutation:
  - Insert / remove nodes
  - Long-jump replacements

---

## 📊 Key Findings

- Greedy is often **near-optimal** under structured workloads
- GA provides gains when:
  - Workloads include **rare queries**
  - Objectives **conflict**
  - Solution space is less constrained

---

## 🚀 Getting Started

### Requirements
- Python 3.9+
- NetworkX
- NumPy

### Run Experiments
```bash
python run_experiments.py
