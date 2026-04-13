# Project Proposal

**Topic:** RAG for Domain-Specific QA (Medical), a Dual-Corpus Architecture for Women's Hormonal Health

## What motivated you to work on this problem

I am currently developing a platform called Reen to help women understand their hormonal health. One of the challenges I am trying to solve is helping users understand complex medical information, such as symptoms, cycle patterns, hormonal conditions such as PCOS or endometriosis. A key component I want to integrate in this platform is a tool that will help users get answers to questions that are both medically accurate and understandable to someone who is a non-clinical user. I would like to design a system that will provide an answer grounded in real evidence but also written in language a patient can use and understand. This project will explore using a dual-corpus RAG architecture that retrieves two sources simultaneously, both PubMedQA for clinical evidence and MedQuAD for patient-facing explanations. I want to build a QA system that can retrieve and synthesize relevant medical literature to answer user health questions, a real product feature I am actively working through for Reen.

## Where you will obtain the dataset for training/evaluating the model

- **PubMedQA** (https://huggingface.co/datasets/qiaojin/PubMedQA) — Biomedical research QA from PubMed abstracts, which will be used as the clinical retrieval corpus and primary evaluation benchmark.
- **MedQuAD** (https://huggingface.co/datasets/lavita/MedQuAD) — 47K patient QA pairs sourced from NIH and CDC, which will be used as the plain-language retrieval corpus.

## A short description of the dataset

- **Inputs:** A natural language health question (e.g., "Does metformin improve fertility in women with PCOS?")
- **Outputs:** Retrieved context from both corpora fed to a generator, producing a yes/no/maybe answer with a grounded explanation.
- **PubMedQA:** About 211K QA pairs total, with about 1K expert annotated with yes/no/maybe labels; the data will be filtered to women's/hormonal health domain.
- **MedQuAD:** About 47K QA pairs written for patients, covering conditions, symptoms, and treatments from NIH/CDC sources.

## Approximately how many observations you have (and, if known, how you will split them)

The annotated PubMedQA subset filtered to women's health yields approximately 137 labeled evaluation examples. These will be split 80/10/10 (development/validation/test) for retrieval evaluation. The full PubMedQA unlabeled subset (~61K) and MedQuAD (~47K) serve as the two retrieval corpora and will be used for indexing. This project will include a three-way comparison experiment: clinical-only retrieval, patient-only retrieval, and combined dual-corpus retrieval, evaluated on the same labeled question set.

## What hardware you will be using

Google Colab free tier with T4 GPU. The retrieval component will use a biomedical bi-encoder (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) to embed both corpora into dual FAISS indexes (IndexFlatIP with L2-normalized embeddings for cosine similarity). The generator will be BioMistral-7B loaded with 4-bit NF4 quantization via bitsandbytes. Total pipeline includes: bi-encoder embedding, two FAISS indexes, and quantized LLM generation.
