# Athena AI — ML Teaching Assistant

> **An end-to-end Machine Learning Teaching Assistant built through Dataset Generation, Supervised Fine-Tuning (QLoRA), and Pedagogical Evaluation.**

---

## 📌 Project Overview

**Athena AI** is an engineering project designed to build a specialized Machine Learning Teaching Assistant. The system transforms dense ML textbooks into high-quality instruction-response pairs, fine-tunes domain-adapted models using QLoRA / Unsloth, evaluates pedagogical teaching performance, and serves interactive explanations.

---

## 📐 Pipeline Architecture (Phase 1 — Hardened Pipeline)

```text
                  ┌───────────────────────────────┐
                  │    Source PDF (Textbooks)     │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │   PDF Cleaning & Extraction   │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │  Word-Count Semantic Chunking │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │   Synthetic Q&A Generation    │
                  │   (Gemma / Ollama + Chunks)   │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ Merge & Deduplicate Datasets  │
                  │ (Gemma + NotebookLM Batches)  │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │   Dataset Validation Gate     │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ ChatML Canonical Formatting   │
                  │   (System Prompt + Llama 3)   │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ Deterministic Train/Val Split │
                  └───────┬───────────────┬───────┘
                          │               │
                 Train (90%)         Val (10%)
                          │               │
                          └───────┬───────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │  QLoRA / SFT Training (Unsloth│
                  │   - 4-bit NF4 Quantization    │
                  │   - Validation Evaluation     │
                  │   - Periodic Checkpointing    │
                  └───────────────┬───────────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
             Checkpoints      Evaluation       Metadata
                  │               │               │
                  └───────────────┼───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │      Saved LoRA Adapter       │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │    Model Merging & Export     │
                  │    (16-bit / 4-bit Standalone)│
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │    Inference & Student Chat   │
                  └───────────────────────────────┘
```

---

## 📁 Project Structure

```text
Athena-Ai/
│
├── Data/
│   ├── books/                     # Source PDF textbooks
│   ├── datasets/                  # Versioned JSON and JSONL datasets
│   │   └── v1/
│   │       ├── notebooklm/        # Batch JSON datasets (batch_01 - batch_04)
│   │       ├── merged/            # Consolidated and formatted datasets
│   │       ├── ml_teacher_dataset.jsonl
│   │       └── checkpoint.json
│   └── evaluation/
│       └── ml_teaching_eval.jsonl # Isolated benchmark evaluation questions
│
├── outputs/
│   ├── final_model/               # Latest trained LoRA adapter
│   ├── evaluation/                # Benchmark inference outputs & metrics
│   └── runs/                      # Isolated training run directories
│       └── run_YYYYMMDD_HHMMSS/
│           ├── adapter/
│           ├── checkpoints/
│           ├── evaluation/
│           └── metadata.json
│
├── src/
│   ├── common/                    # Centralized configs and utilities
│   │   ├── config.py              # Central path & hyperparameter registry
│   │   ├── logger.py              # Structured logging
│   │   └── utils.py               # JSON/JSONL, hashing, and file helpers
│   │
│   ├── dataset_generation/        # Ingestion & synthetic generation
│   │   ├── pdf_processor.py       # Header/footer trimming & block extraction
│   │   ├── text_cleaner.py        # Ligatures, OCR hyphenation & code preservation
│   │   ├── chunker.py             # Word-count semantic chunker
│   │   ├── gemma_generator.py     # Ollama generation client
│   │   ├── json_validator.py      # Output schema validator
│   │   ├── deduplicator.py        # Real-time instruction deduplication
│   │   ├── dataset_writer.py      # Atomic JSONL saves & checkpoint tracking
│   │   └── main.py                # Dataset generation runner & dry-run
│   │
│   └── finetuning/                # SFT & LoRA Pipeline
│       ├── data_opr/              # Dataset operations
│       │   ├── dataset_merger.py  # Gemma + NotebookLM merge & deduplication
│       │   ├── dataset_validator.py# Pre-train dataset validation gate
│       │   ├── formatter.py       # ChatML tokenization & formatting
│       │   └── dataset_loader.py  # Deterministic train/val split
│       │
│       ├── core_finetuning/       # Training & inference engine
│       │   ├── model_loader.py    # Unsloth 4-bit model loader
│       │   ├── lora_config.py     # LoRA target projections & config
│       │   ├── trainer.py         # SFTTrainer with eval & checkpointing
│       │   ├── train.py           # Run-orchestrator with metadata logging
│       │   ├── evaluator.py       # Fixed test set evaluator & leakage audit
│       │   ├── merge_model.py     # LoRA weight merging & standalone export
│       │   └── inference.py       # Single prompt & interactive teacher chat
│       │
│       └── test_finetuning/       # Smoke tests for model loading & inference
│           ├── test_model_load.py
│           ├── test_lora.py
│           └── test_inference.py
│
├── tests/                         # Comprehensive unit & integration tests
├── pyproject.toml                 # Dependencies (Python 3.12)
├── uv.lock                        # Deterministic dependency lock
├── .python-version                # Python 3.12 pin
└── README.md
```

---

## 🚀 Getting Started

### 1. Environment Setup

Athena AI uses **Python 3.12** and `uv` for fast, deterministic dependency resolution:

```bash
# Sync dependencies
uv sync
```

---

## 🛠️ Unified CLI Usage

Athena AI provides a unified CLI via `python main.py <command>` or via direct module invocation:

### 1. Dataset Generation (Synthetic Extraction)
```bash
# Dry-run to test PDF extraction and Ollama reachability
python main.py dataset-generate --dry-run

# Run dataset generation (with auto-resume support)
python main.py dataset-generate --resume
```

### 2. Dataset Merging & Deduplication
Merges local Gemma outputs and NotebookLM batch files with greeting filtering and deterministic hashing:
```bash
python main.py dataset-merge
```

### 3. Dataset Validation Gate
Validates required keys, non-empty responses, length constraints, and duplicate absence:
```bash
python main.py dataset-validate
```

### 4. ChatML Formatting
Formats records into canonical ChatML representation with system prompt:
```bash
python main.py dataset-format
```

### 5. Supervised Fine-Tuning (SFT / QLoRA)
Runs training with validation loss evaluation and checkpoints:
```bash
python main.py train
```

### 6. Benchmark Evaluation & Leakage Check
Runs inference against the fixed, isolated evaluation questions (`Data/evaluation/ml_teaching_eval.jsonl`) and verifies no data leakage:
```bash
python main.py evaluate
```

### 7. Model Merging (Export Standalone Weights)
Merges LoRA adapter back into base model 16-bit float weights:
```bash
python main.py merge-model --format 16bit
```

### 8. Interactive Teacher Assistant
Start an interactive teaching session in the terminal:
```bash
# Interactive Chat
python main.py inference --interactive

# Single prompt query
python main.py inference --prompt "Explain the bias-variance tradeoff in machine learning."
```

---

## 🧪 Testing Suite

Execute the full suite of unit and integration tests:

```bash
uv run python -m unittest discover -s tests
```

---

## ⚙️ Reproducibility & Run Metadata

Every training run automatically generates a timestamped directory under `outputs/runs/run_YYYYMMDD_HHMMSS/` containing:
- `adapter/`: LoRA adapter weights and tokenizer configs.
- `checkpoints/`: Intermediate training checkpoints with validation checkpoints.
- `metadata.json`: Exact Python version, package hashes, dataset SHA-256 hash, dataset sizes, seed, and hyperparameters for full reproducibility.
