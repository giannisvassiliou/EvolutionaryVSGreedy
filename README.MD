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
## 🧪 Experiment Flags

The experiment script supports flexible configuration of **workloads, rare queries, and constraints**.

### 📊 Workload Parameters

- `--n`  
  Number of queries in the workload

- `--qsize`  
  Size of simple queries (number of nodes per query)

- `--min_len`, `--max_len`  
  Path length range for structured (random-walk) queries

- `--structured_ratio`  
  Fraction of structured vs simple queries  
  (0.0 = all simple, 1.0 = all structured)

---

### 🌟 Rare Query Configuration

- `--rare_query_ratio`  
  Fraction of queries that include rare nodes

- `--rare_fraction`  
  Percentage of graph nodes considered "rare" (low-degree)

- `--rare_degree_threshold`  
  Degree cutoff for defining rare nodes

- `--rare_weight`  
  Weight assigned to rare queries in scoring

---

### 🧩 Constraints & Optimization

- `--budget`  
  Maximum number of nodes in the summary

- `--strict_connectivity`  
  Enforce connectivity as a hard constraint (otherwise soft penalty)

---

### ⚙️ Genetic Algorithm Parameters

- `--population_size`  
  Number of individuals in GA population

- `--generations`  
  Number of evolutionary iterations

- `--mutation_rate`  
  Probability of mutation per individual

---

### 📂 Data Input

- `--schema_path`  
  Path to schema graph file

- `--workload_path`  
  Path to SPARQL log (JSON format)

- `--clean_schema`  
  Filter schema to valid nodes only

- `--keep_predicate_nodes`  
  Include predicates as nodes in the graph

---

### 🔍 Misc

- `--seed`  
  Random seed for reproducibility

---

## 🧠 Notes

- Rare queries are generated from **low-degree nodes** and can be weighted more heavily to simulate importance.
- Structured workloads are created via **random walks** over the graph.
- The framework supports both:
  - **Synthetic workloads** (controlled experiments)
  - **Real SPARQL logs**

---
