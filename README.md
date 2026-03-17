# Taylor Series Seq2Seq

## Overview

The goal of this work is to model **symbolic Taylor series expansion as a sequence-to-sequence learning problem**, where a neural network translates a mathematical function into its Taylor expansion.

---

## Key Contributions

* Generated **31,000+ symbolic function–Taylor expansion pairs** using SymPy
* Designed a **custom tokenizer for mathematical expressions** (1,582-token vocabulary)
* Implemented a **Seq2Seq LSTM model** achieving **93.08% validation accuracy**
* Implemented a **Transformer-based Seq2Seq model** achieving **95.22% validation accuracy**
* Performed **comparative analysis** of LSTM vs Transformer for symbolic learning

---

## Dataset Generation

* Functions generated using **SymPy**
* Includes:

  * Polynomial functions
  * Trigonometric functions
  * Exponential functions
* Taylor expansion computed up to **4th order**
* Generated 31259 sample from 33021 attempts
* I had traditional method for the vertification purpose.

![Reprsentation](assests/data.png)


## Tokenization Strategy

A **custom regex-based tokenizer** was implemented to handle symbolic expressions.

![Output](assests/tokenizer.png)
---

## Model Architectures

### 🔹 LSTM Seq2Seq

* Encoder–Decoder architecture
* Hidden size: 256
* Embedding size: 128
* Learns sequential symbolic dependencies

### 🔹 Transformer Seq2Seq

* 4 Encoder + 4 Decoder layers
* 8 attention heads
* Embedding size: 128
* Captures **global symbolic relationships using self-attention**

---

## Training Details

* Optimizer: Adam
* Learning Rate: 1e-3
* Loss: CrossEntropy (ignore padding)
* Techniques used:

  * Label smoothing
  * Gradient clipping
  * Learning rate scheduling

---

## Results

### LSTM Performance

* Validation Accuracy: **93.08%**

![LSTM Training Curve](results/lstm_training_curves.png)

---

### Transformer Performance

* Validation Accuracy: **95.22%**

![Transformer Training Curve](results/transformer_training_curves.png)

---

## Key Observations

* Transformer outperforms LSTM in symbolic prediction tasks
* Self-attention enables better handling of **long-range dependencies**
* Model successfully learns the **"grammar" of Taylor expansions**
* Large vocabulary (1,582 tokens) handled effectively

---

## Repository Structure

```
src/
 ├── dataset/
 │    └── generating_dataset.py
 ├── models/
 │    ├── LSTM_model.py
 │    └── Transformer_model.py

data/
 └── taylor_tokenized_dataset.jsonl

models/
 ├── lstm_taylor_model.pth
 └── transformer_taylor_model.pth

results/
 ├── lstm_training_curves.png
 └── transformer_training_curves.png
```

---

## How to Run

```bash
pip install -r requirements.txt
python src/models/Transformer_model.py
```

---

## Relevance to FASEROH

This work demonstrates that **symbolic mathematical transformations can be learned using neural sequence models**.

It provides a strong foundation for extending the approach to:

👉 Histogram → Symbolic Function translation
👉 Neural symbolic regression
👉 Full FASEROH pipeline integration

---

## Author

**V Kannabiran**
B.Tech CSE, IIIT Kottayam
GSoC 2026 Applicant
