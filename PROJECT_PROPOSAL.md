# Project Proposal

**Topic:** RAG for Domain-Specific QA (Medical) — Dual-Corpus Architecture for Women's Hormonal Health

## What motivated you to work on this problem

I am the founder of Reen, a femtech platform helping women understand their hormonal health. A core challenge I face is delivering answers that are both medically accurate and understandable to a non-clinical user. For example, a user might ask "Why do I feel anxious before my period?" — and the system needs to provide an answer grounded in real evidence but written in language she can actually use. This project explores a dual-corpus RAG architecture that retrieves from two sources simultaneously: PubMedQA for clinical evidence and MedQuAD for patient-facing explanations. The result is a system that grounds answers in research while keeping the language accessible — a real product decision I am actively working through for Reen.

## Where you will obtain the dataset for training/evaluating the model

- **PubMedQA** (`qiaojin/PubMedQA`): https://huggingface.co/datasets/qiaojin/PubMedQA — Biomedical research QA from PubMed abstracts, used as the clinical retrieval corpus and primary evaluation benchmark.
- **MedQuAD** (`lavita/MedQuAD`): https://huggingface.co/datasets/lavita/MedQuAD — 47K patient-facing QA pairs sourced from NIH and CDC, used as the plain-language retrieval corpus.

## A short description of the dataset

- **Inputs:** A natural language health question (e.g., "Does metformin improve fertility in women with PCOS?")
- **Outputs:** Retrieved context from both corpora fed to a generator, producing a yes/no/maybe answer with a grounded explanation
- **PubMedQA:** ~211K QA pairs total; 1K expert-annotated with gold yes/no/maybe labels; filtered to women's/hormonal health domain
- **MedQuAD:** ~47K QA pairs written for patients, covering conditions, symptoms, and treatments from NIH/CDC sources

## Approximately how many observations you have (and how you will split them)

The expert-annotated PubMedQA subset filtered to women's health yields approximately 100–150 labeled evaluation examples. These will be split 80/10/10 (train/validation/test). The artificial PubMedQA subset (~211K) and MedQuAD (~47K) serve as the two retrieval corpora (not split — used in full for indexing). The project includes a three-way comparison experiment: clinical-only retrieval, patient-only retrieval, and combined dual-corpus retrieval.

## What hardware you will be using

Google Colab free tier with T4 GPU. The retrieval component uses a BiomedBERT bi-encoder (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`) for embedding both corpora into dual FAISS indexes. The generator is BioMistral-7B with 4-bit quantization (NF4 via bitsandbytes), which fits within Colab's ~15GB GPU memory. Total pipeline: bi-encoder embedding + two FAISS `IndexFlatIP` indexes + quantized LLM generation.
