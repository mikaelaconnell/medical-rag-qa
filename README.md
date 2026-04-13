# RAG for Domain-Specific QA: Medical (Hormonal Health)

A Retrieval-Augmented Generation (RAG) system for answering medical questions about hormonal and reproductive health, built on PubMedQA.

## Project Overview

This project builds a QA system that retrieves relevant PubMed abstracts and synthesizes answers to medical questions focused on women's hormonal health (PCOS, endometriosis, menstrual cycle patterns, etc.). It serves as the research foundation for integrating evidence-based medical QA into health platforms.

## Architecture

```
User Question → Bi-Encoder (BiomedBERT) → FAISS Index → Top-K Abstracts → LLM (BioMistral-7B-Q4) → Answer
```

1. **Retrieval**: BiomedBERT bi-encoder embeds questions and PubMed abstracts; FAISS finds top-K relevant passages
2. **Generation**: BioMistral-7B (4-bit quantized) generates yes/no/maybe answers with explanations grounded in retrieved context

## Dataset

- **Source**: [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) (`qiaojin/PubMedQA`)
- **Size**: ~211K QA pairs (1K expert-annotated)
- **Inputs**: Natural language medical question + PubMed abstract context
- **Outputs**: yes/no/maybe label + long-form explanation
- **Domain Focus**: Filtered to hormonal health, reproductive health, women's health

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Hardware

- Google Colab free tier (T4 GPU)
- ~15GB GPU RAM sufficient for 4-bit quantized BioMistral-7B + FAISS index

### Reproduction Steps

1. Open `notebooks/01_data_exploration.ipynb` in Google Colab
2. Run all cells — dataset downloads automatically from HuggingFace
3. Open `notebooks/02_retrieval_pipeline.ipynb` to build the FAISS index
4. Open `notebooks/03_rag_pipeline.ipynb` for the full RAG system
5. Open `notebooks/04_evaluation.ipynb` for metrics and visualizations

## Project Structure

```
medical-rag-qa/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_retrieval_pipeline.ipynb
│   ├── 03_rag_pipeline.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_utils.py        # Dataset loading and filtering
│   ├── retriever.py         # Bi-encoder + FAISS retrieval
│   ├── generator.py         # LLM generation with context
│   └── evaluation.py        # Metrics and visualization
├── data/
│   └── sample/              # Small data samples for submission
└── figures/                  # Generated visualizations
```

## Key Results

_(To be filled after experiments)_

## Course

CSCI E-222: Foundations of Large Language Models — Final Project
