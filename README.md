# Athena AI

> **Building a complete Machine Learning Teaching Assistant through Dataset Generation, Fine-Tuning, and Retrieval-Augmented Generation (RAG).**

---

## 📌 Project Overview

**Athena AI** is an end-to-end engineering initiative designed to build a highly specialized Machine Learning Teaching Assistant. Rather than relying on black-box wrappers or high-level abstract frameworks, this project is built entirely from scratch to gain direct hands-on experience with the critical components of the modern Generative AI lifecycle:

- **Source Processing & Extraction**: Extracting and cleaning structure from dense educational documents.
- **Robust Dataset Generation**: Programmatic instruction-response generation using local LLMs.
- **Instruction Tuning**: Formatting and prepping high-quality training pairs.
- **LoRA / QLoRA Fine-Tuning**: Supervised fine-tuning of small language models on domain-specific data.
- **Retrieval-Augmented Generation (RAG)**: Building local embedding pipelines and vector stores for context retrieval.
- **Evaluation Pipelines**: Auditing and scoring retrieval accuracy and generation quality.

---

## 📊 Repository Status & Progress

The project is currently under active development. Below is a detailed view of the dataset generation metrics at the current checkpoint:

| Metric | Value |
| :--- | :--- |
| **Dataset Status** | Paused at Chunk 94 |
| **Processed Chunks** | 94 |
| **Successful Examples** | 92 |
| **Failed Chunks** | 2 |
| **Success Rate** | 97.9% |
| **Unique Concepts** | 82 |
| **Average Response Length** | 4,222 Characters |
| **Beginner Examples** | 6 |
| **Intermediate Examples** | 80 |
| **Advanced Examples** | 6 |

---

## ⚙️ Dataset Generation Details

The dataset generation pipeline extracts educational context blocks from textbooks and processes them into instruction-response datasets for fine-tuning.

- **Source Book**: *Hands-On Machine Learning*
- **Total Chunks Processed**: 94
- **Successful Dataset Records**: 92
- **Failed Chunks**: `chunk_044`, `chunk_086`
- **Architectural Safeguards**:
  - **Checkpointing & Resume**: Fully decoupled progress tracking allows restarting runs instantly without loss.
  - **Deduplication**: Hash-based validation ensures redundant concepts are filtered during generation.
  - **Incremental Saving**: Outputs are appended directly to [JSONL](file:///home/lakshay/ML-Teacher/Data/datasets/v1/ml_teacher_dataset.jsonl) on successful cycles to mitigate corruption risks.

*Thanks to checkpointing, retry logic, and resume capabilities, the generation pipeline successfully survived interruptions, model timeouts, and server restarts during execution.*

---

## 📐 Project Architecture

The codebase follows a modular design layout:

```text
ML-Teacher
│
├── Data                     # Raw data and output datasets
│   ├── books                # Source PDF textbooks
│   ├── chunks               # Intermediate extracted text segments
│   └── datasets             # Versioned JSON and JSONL datasets
│
├── src                      # Core source code
│   ├── dataset_generation   # Pipeline parsing, cleaning, generating, and validation
│   ├── finetuning           # QLoRA fine-tuning modules (SFT)
│   ├── rag                  # Embedding generation and database queries
│   └── common               # Shared configuration and helper modules
│
├── embeddings               # Local text embedding artifacts
├── vector_db                # Local vector store instances
├── models                   # Model base weights and fine-tuned adapter exports
├── notebooks                # Research and analysis files
└── logs                     # Progress and execution log files
```

---

## 🛠️ Implemented Features

- [x] **PDF Processing**: Automatic extraction of text blocks, filtering header/footers and margins.
- [x] **Text Cleaning**: Normalization, whitespace cleanups, and page junction processing.
- [x] **Chunk Generation**: Word-count semantic chunker keeping logical context intact.
- [x] **Dataset Generation**: Automated LLM query loop targeting conceptual curriculum.
- [x] **JSON Validation**: Structure, keys, and difficulty validation.
- [x] **Retry Logic**: Graceful error handling for API timeouts and bad JSON schemas.
- [x] **Deduplication**: Exact-match seen-hash deduplicator preventing duplicate explanations.
- [x] **Incremental Saving**: Real-time line appending to prevent in-flight data loss.
- [x] **Failure Tracking**: Isolated tracking of chunk failures for retry audits.
- [x] **Checkpointing**: Completed chunk state tracking independent of output structure.
- [x] **Resume Support**: Ability to automatically restart execution from the last processed block.
- [x] **Dataset Statistics**: Built-in scripts to track metrics, distributions, and average token sizes.

---

## 🗺️ Upcoming Roadmap

### Phase 1 — Dataset Completion
- Resume generation from Chunk 95
- Process all remaining 117 chunks to achieve complete coverage (211 chunks total)

### Phase 2 — Fine-Tuning
- Format generated JSON/JSONL datasets to match model templates (e.g. ChatML)
- Conduct QLoRA/LoRA parameter-efficient training on a local consumer GPU
- Export adapter weights and merge models for deployment
- Build baseline model evaluation suite

### Phase 3 — RAG System
- Create local embedding generation pipeline for textbook segments
- Establish a local vector database instance for indexing book chunks
- Build retrieval pipeline with reranking capability
- Assemble prompt context compiler

### Phase 4 — ML Teacher Assistant
- Integrate the fine-tuned instructor model with the retriever pipeline
- Implement system evaluation suite (faithfulness, answer relevance)
- Design interactive CLI/web interface for student Q&A

---

## 💡 Lessons Learned

- **Checkpointing is Critical**: In long-running pipelines running on local models, timeouts, transient Out-Of-Memory (OOM) exceptions, and hardware restarts will occur. Separating state tracking from the main data files makes the codebase resilient to these issues.
- **Incremental Appends Over Buffering**: Writing directly to line-oriented JSONL format guarantees that if a process crashes mid-generation, all previously generated records are immediately written to disk and safe.
- **Local Model Limitations**: Small locally hosted language models sometimes deviate from prompt schemas or generate invalid JSON structures. A strong schema validator paired with custom retry parameters makes local LLM execution far more predictable.
- **Pipeline Over Scripts**: Designing standard modular pipelines rather than quick scripts creates a maintainable framework that is easily reused, tested, and optimized for new datasets.

---

## 📈 Repository Status

```text
Status: Active Development
Dataset Generation: 43.6% Complete (92/211 Chunks)
Fine-Tuning: Not Started
RAG Pipeline: Not Started
Evaluation: Planned
```

---

## 🎓 Final Note

This project is a learning-focused engineering exercise. The objective is to understand and demystify the internal machinery of modern Generative AI architectures, exploring the practical software engineering challenges of building robust AI pipelines from data preprocessing through training and deployment.
