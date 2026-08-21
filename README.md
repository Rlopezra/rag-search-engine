# Modern Search & RAG Techniques in Python

This repository contains my code for DevOps' **Modern Search Engine Techniques in Python**! The course guides you in creating production-grade search systems from scratch, starting with traditional lexical keyword algorithms all the way up to agentic, multimodal Retrieval-Augmented Generation (RAG) pipelines. I initialized the project via terminal creating a virtual environment, that had a folder structure that seperated my Command Line Interface (CLI) scripts, core classes, helper functions, and data. Updates and changes to the project were tracked through consistent Git commits.

All LLM and embedding integrations are powered by free models hosted on **OpenRouter** using the standard `openai` Python SDK.

---

## 📚 What I Learn

* **Lexical & Keyword Search:** Implement inverted indexes, TF-IDF weighting, and modern BM25 algorithms with metadata boosting.
* **Vector & Semantic Retrieval:** Harness dense embeddings, vector search, cosine/dot-product metrics, and vector database workflows.
* **Pipeline Optimization:** Dynamic document chunking, hybrid search blending (RRF / score normalization), and cross-encoder reranking.
* **LLMs & Orchestration:** Query expansion, intent correction, and combining context with LLMs to generate grounded answers.
* **Advanced RAG Patterns:** Autonomous agentic retrieval loops and multimodal cross-modal search (text + images).
* **Systematic Evaluation:** Quantitative metrics (Precision, Recall, MRR, NDGC) to measure and improve pipeline relevance.

---

## 🗺️ Course Chapter Overview

| Chapter | Module | Highlights |
| :--- | :--- | :--- |
| **01** | **Preprocessing** | Text normalization, tokenization, stop-word removal, and cleaning raw corpora for indexing. |
| **02** | **TF-IDF** | Building inverted indexes, calculating term frequencies, and document scoring schemes. |
| **03** | **Keyword Search** | BM25 rank tuning, parameter optimization ($k_1, b$), and metadata boosting. |
| **04** | **Semantic Search** | Dense embeddings, vector similarity metrics, and vector store integration. |
| **05** | **Chunking** | Fixed-size, sliding window, and semantic chunking strategies for optimal context retention. |
| **06** | **Hybrid Search** | Reciprocal Rank Fusion (RRF) and linear score blending to combine BM25 + Vector scores. |
| **07** | **LLMs** | Prompting for query expansion, rephrasing, and intent classification. |
| **08** | **Reranking** | Cross-encoder rescoring to filter and re-rank candidate document pools. |
| **09** | **Evaluation** | Measuring retrieval performance using Precision@K, Recall@K, and hit rates. |
| **10** | **Augmented Generation** | Synthesizing grounded, context-aware LLM answers while minimizing hallucinations. |
| **11** | **Agentic RAG** | Designing autonomous loops for multi-step reasoning, query refinement, and tool use. |
| **12** | **Multimodal** | Joint image-text embeddings and cross-modal retrieval workflows. |

