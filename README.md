# Product Question Answering System

This repository contains a multi-milestone NLP project that builds an extractive question answering system over Amazon product data. The system takes structured product information, converts it into natural language context passages, and trains models to answer questions about products by extracting relevant spans from those passages.

The project is organized into three milestones, each building on the previous one.

---

## Table of Contents

- [Dataset](#dataset)
- [Milestone 1 -- Data Analysis and Exploration](#milestone-1----data-analysis-and-exploration)
- [Milestone 2 -- Neural Network QA Model from Scratch](#milestone-2----neural-network-qa-model-from-scratch)
- [Milestone 3 -- Pre-trained Model Fine-tuning](#milestone-3----pre-trained-model-fine-tuning)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)

---

## Dataset

The project uses an Amazon product dataset consisting of two CSV files:

- **amazon_products_cleaned.csv** -- Contains roughly 100,000 products with fields such as title, price, list price, star rating, number of reviews, best seller status, category ID, and units bought in the last month.
- **amazon_categories.csv** -- Maps category IDs to human-readable category names.

Since the dataset does not include real question-answer pairs, all QA samples are generated synthetically from the structured product metadata. A context paragraph is constructed for each product by combining its attributes into natural sentences, and questions are paired with the corresponding answer spans within that context.

---

## Milestone 1 -- Data Analysis and Exploration

**Notebook:** `milestone1_data_analysis.ipynb`

This milestone focuses on understanding the dataset before any modeling work begins. The analysis covers:

- Loading and merging the product and category tables.
- Inspecting data types, missing values, and basic statistics.
- Examining distributions of price, ratings, reviews, and purchase counts.
- Analyzing category-level statistics and class imbalance across categories.
- Exploring product title lengths and word frequencies.
- Computing correlations between numeric features.
- Identifying limitations of the dataset, including the absence of product descriptions, real QA pairs, and review text.

The findings from this milestone directly informed how the synthetic QA pairs were designed in the later milestones.

---

## Milestone 2 -- Neural Network QA Model from Scratch

**Notebook:** `milestone2_nn_from_scratch.ipynb`

In this milestone, an extractive QA model is built entirely from scratch without using any pre-trained weights. The pipeline includes:

1. **Context construction** -- Each product's structured fields are turned into a readable paragraph.
2. **Synthetic QA generation** -- Seven types of questions are generated per product (price, category, rating, popularity, product name, best seller status, and review count), each paired with the exact answer span and its character offset in the context.
3. **Tokenization and vocabulary** -- A simple word-level tokenizer is built from the training data, with special tokens for padding, unknown words, and a separator between the question and context.
4. **Dataset encoding** -- Questions and contexts are concatenated with a separator token and converted to padded integer sequences. The answer span positions are recorded as token-level start and end indices.
5. **Model architecture** -- A Transformer encoder model implemented from scratch with:
   - Learned word embeddings
   - Sinusoidal positional encoding
   - Three Transformer encoder layers (4 attention heads, 128-dimensional hidden states, 256-dimensional feed-forward layers)
   - Two linear heads predicting the start and end positions of the answer span
6. **Training** -- The model is trained for 5 epochs with Adam optimizer, cross-entropy loss on both start and end positions, gradient clipping, and a step learning rate scheduler.
7. **Evaluation** -- The test set is evaluated using token-level F1 score and exact match, with a per-question-type breakdown.

The model is saved as `milestone2_qa_model.pt`.

---

## Milestone 3 -- Pre-trained Model Fine-tuning

**Notebook:** `milestone3_pretrained_finetuning.ipynb`

This milestone replaces the from-scratch model with a pre-trained DistilBERT model fine-tuned on the same product QA task. Key differences and improvements include:

1. **Diverse question templates** -- Each question type now has multiple phrasings (for example, the price question may appear as "What is the price of this product?", "How much does this item cost?", or "Tell me the price.") to encourage the model to generalize beyond fixed patterns.
2. **DistilBERT tokenizer and model** -- The HuggingFace `distilbert-base-uncased` checkpoint is used. Its subword tokenizer handles out-of-vocabulary words naturally, unlike the word-level tokenizer in Milestone 2.
3. **Fine-tuning** -- The full model is fine-tuned with AdamW optimizer and a linear learning rate schedule with warmup.
4. **Evaluation on natural text queries** -- Beyond the standard test set metrics, the fine-tuned model is tested on completely new, naturally phrased questions that differ from any training template. This demonstrates its ability to generalize to real user input.
5. **Comparison with Milestone 2** -- The fine-tuned DistilBERT model achieves higher F1 and exact match scores across all question types, and handles varied phrasings and unseen vocabulary much more gracefully.

The fine-tuned model is saved to the `milestone3_distilbert_qa/` directory.

---

## Requirements

The project relies on the following Python libraries:

- pandas
- numpy
- matplotlib
- torch (PyTorch)
- transformers (HuggingFace, used in Milestone 3)

A CUDA-capable GPU is recommended for training but not strictly required.

---

## How to Run

1. Make sure all dependencies listed above are installed.
2. Place the two CSV files (`amazon_products_cleaned.csv` and `amazon_categories.csv`) in the project root directory.
3. Open and run the notebooks in order:
   - Start with `milestone1_data_analysis.ipynb` to explore the dataset.
   - Then run `milestone2_nn_from_scratch.ipynb` to train and evaluate the from-scratch model.
   - Finally run `milestone3_pretrained_finetuning.ipynb` to fine-tune DistilBERT and compare results.

Each notebook is self-contained and can be executed top to bottom.

---

## Project Structure

```
QA_project/
    amazon_products_cleaned.csv        -- Cleaned product data
    amazon_categories.csv              -- Category ID to name mapping
    milestone1_data_analysis.ipynb     -- Data exploration and analysis
    milestone2_nn_from_scratch.ipynb   -- QA model built from scratch
    milestone3_pretrained_finetuning.ipynb -- Fine-tuned DistilBERT QA model
    README.md                          -- This file
```

After running the notebooks, the following artifacts are produced:

```
    milestone2_qa_model.pt             -- Saved from-scratch model checkpoint
    milestone3_distilbert_qa/          -- Saved fine-tuned DistilBERT model and tokenizer
```
